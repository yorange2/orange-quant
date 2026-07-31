#!/usr/bin/env python3
"""
Orange Quant automated trading server (exchange-agnostic).

Runs long-term in Docker, rebalancing on a daily schedule. Driven by an
``ExchangeSpec`` supplied by the per-exchange adapter package, so a single loop
serves every venue. Entrypoint: ``run(spec, argv)``.

    from orange_quant.server import run
    from .spec import SPEC
    run(SPEC)
"""

import os
import sys
import time
import signal
import logging
import argparse
import threading
from pathlib import Path
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv(override=True)

from orange_quant.runner import StrategyRunner
from orange_quant.spec import ExchangeSpec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("orange-quant")

_shutdown = False
# PID of the main server process, captured before any multiprocessing fork so
# on_signal can tell itself apart from qlib's forked worker processes.
_main_pid = None

# Liveness heartbeat. The main loop touches this file on every wait tick and
# around retrain/rebalance; the Docker healthcheck (orange_quant.healthcheck)
# and the in-process watchdog below read it to detect a hung loop. The timeout
# must exceed the longest legitimate blocking step (data refresh + ensemble
# retrain + prediction), so it is deliberately generous.
_HEARTBEAT_FILE = os.environ.get("OQ_HEARTBEAT_FILE", "/tmp/oq_heartbeat")
_WATCHDOG_TIMEOUT = int(os.environ.get("OQ_WATCHDOG_TIMEOUT", "1800"))


def _beat():
    """Record a liveness heartbeat (main loop only)."""
    try:
        with open(_HEARTBEAT_FILE, "w") as f:
            f.write(str(int(time.time())))
    except Exception:
        pass


def _start_watchdog():
    """Force-exit the process if the main loop stops beating past the timeout.

    A deadlock (e.g. a wedged multiprocessing pool) blocks the main thread in a
    syscall that releases the GIL, so this daemon thread still runs and can act.
    Exiting lets Docker's ``restart: unless-stopped`` recover the container —
    native Docker healthchecks only mark 'unhealthy', they don't restart.
    """
    def _watch():
        while True:
            time.sleep(30)
            try:
                last = int(open(_HEARTBEAT_FILE).read().strip())
            except Exception:
                continue
            age = int(time.time()) - last
            if age > _WATCHDOG_TIMEOUT:
                logger.error(
                    f"⚠ Watchdog: no heartbeat for {age}s (> {_WATCHDOG_TIMEOUT}s); "
                    f"the loop looks hung — exiting for restart"
                )
                os._exit(1)
    threading.Thread(target=_watch, daemon=True, name="oq-watchdog").start()


def load_strategy_defaults(model_path: str):
    """Read topk / risk_degree / n_drop / hold_thresh from the model's YAML config,
    so live trading uses the same strategy parameters as the backtest."""
    try:
        import yaml
        cfg_path = Path("config") / f"{Path(model_path).stem}.yaml"
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text())
            kwargs = cfg.get("strategy", {}).get("kwargs", {})
            return (kwargs.get("topk"), kwargs.get("risk_degree"),
                    kwargs.get("n_drop"), kwargs.get("hold_thresh"))
    except Exception as e:
        logger.warning(f"Failed to read strategy config for {model_path}: {e}")
    return None, None, None, None


def on_signal(signum, frame):
    """Graceful-shutdown handler for the main server process.

    qlib forks a multiprocessing pool (Alpha158 feature computation) during
    retrain, and the workers inherit this handler. On pool teardown the parent
    sends them SIGTERM; if a worker runs this handler — which only sets a flag —
    instead of dying, the parent blocks forever in wait(), a hard deadlock. So
    in any process that is not the main server, restore the default disposition
    and terminate, which is exactly what the pool expects.
    """
    global _shutdown
    if _main_pid is not None and os.getpid() != _main_pid:
        signal.signal(signum, signal.SIG_DFL)
        os._exit(0)
    logger.info(f"Received signal {signum}, shutting down safely...")
    _shutdown = True


def retrain_model(spec: ExchangeSpec, model_path: str):
    """Incrementally refresh data + retrain the model.

    Never raises: a retrain failure must not crash the daily loop or skip the
    rebalance. On failure we log and fall back to the previously saved model.
    """
    try:
        logger.info("📥 Refreshing data incrementally...")
        spec.rebuild_data()
        config_name = Path(model_path).stem
        logger.info(f"🚀 Retraining: {config_name}")
        from orange_quant.experiment import run_from_yaml
        run_from_yaml(f"config/{config_name}.yaml")
        logger.info("✅ Model updated")
    except Exception as e:
        logger.error(
            f"Retrain failed ({e}); continuing with the existing model",
            exc_info=True,
        )


def run_rebalance(spec, broker, coins, dry_run, topk, risk_degree, n_drop, hold_thresh,
                  lookback, min_trade, model_path=None):
    """Execute a single rebalance."""
    try:
        runner = StrategyRunner(
            broker=broker,
            coins=coins,
            quote_ccy=spec.quote_ccy,
            provider_uri=spec.provider_uri,
            topk=topk,
            n_drop=n_drop,
            hold_thresh=hold_thresh,
            lookback_days=lookback,
            min_trade=min_trade,
            max_position_pct=spec.max_position_pct,
            risk_degree=risk_degree,
            liquidity_multiple=spec.liquidity_multiple,
            use_blacklist=spec.use_blacklist,
            blacklist_path=spec.blacklist_path,
            model_path=model_path,
            state_path=spec.entry_state_path,
        )
        result = runner.run_once(dry_run=dry_run)

        if result["status"] != "ok":
            return

        balances = broker.get_balances()
        cash = balances.get(spec.quote_ccy, 0)
        positions = {c: a for c, a in balances.items() if c != spec.quote_ccy and a > 0}

        total_value = cash
        if positions:
            prices = broker.get_current_prices(list(positions.keys()))
            for coin, amt in positions.items():
                total_value += amt * prices.get(coin, 0)

        logger.info(f"💰 Total equity: ${total_value:,.2f} | {spec.quote_ccy}: ${cash:,.2f} "
                    f"| Positions: {len(positions)}")
        logger.info(f"📊 Target holdings: {result['target_coins']}")
        if result["trades"]:
            for t in result["trades"]:
                logger.info(f"   {t[0]} {t[1]}")
    except Exception as e:
        logger.error(f"Rebalance failed: {e}", exc_info=True)


def _resolve_rotation(spec, args, yaml_n_drop, yaml_hold):
    """Rotation params, honouring the parity rule.

    Binance (honor_config_rotation=False) forces full rotation (n_drop=None,
    hold_thresh=0) unless explicitly overridden on the CLI; Hyperliquid honours
    the model YAML values.
    """
    if args.n_drop is not None:
        n_drop = args.n_drop
    elif spec.honor_config_rotation:
        n_drop = yaml_n_drop if yaml_n_drop is not None else spec.default_n_drop
    else:
        n_drop = None  # full rotation

    if args.hold_thresh is not None:
        hold_thresh = args.hold_thresh
    elif spec.honor_config_rotation:
        hold_thresh = yaml_hold if yaml_hold is not None else spec.default_hold_thresh
    else:
        hold_thresh = 0
    return n_drop, hold_thresh


def run(spec: ExchangeSpec, argv=None):
    """Server entrypoint. ``spec`` selects the exchange; ``argv`` overrides sys.argv."""
    global logger
    logger = logging.getLogger(spec.logger_name)

    parser = argparse.ArgumentParser(description=f"{spec.server_title} Trading Server")
    parser.add_argument("--hour", type=int, default=0, help="Daily rebalance time (UTC hour)")
    parser.add_argument("--minute", type=int, default=15, help="Daily rebalance time (minute)")
    parser.add_argument("--dry-run", action="store_true", help="Analyze only, no orders placed")
    parser.add_argument("--once", action="store_true", help="Run once then exit")
    parser.add_argument("--topk", type=int, default=None,
                        help="Number of positions to hold (default: strategy topk from the model's yaml config)")
    parser.add_argument("--risk-degree", type=float, default=None,
                        help="Fraction of equity to deploy (default: risk_degree from the model's yaml config)")
    parser.add_argument("--n-drop", type=int, default=None,
                        help="Max positions rotated per rebalance (default: n_drop from the model's yaml config)")
    parser.add_argument("--hold-thresh", type=int, default=None,
                        help="Min days to hold before rotating out (default: hold_thresh from the model's yaml config)")
    parser.add_argument("--lookback", type=int, default=spec.default_lookback, help="Lookback window in days")
    parser.add_argument("--min-trade", type=float, default=spec.default_min_trade,
                        help=f"Minimum trade size in {spec.quote_ccy}")
    parser.add_argument("--model", type=str, default=spec.default_model, help="LightGBM model path")
    parser.add_argument("--retrain", action="store_true", help="Refresh data and retrain the model before rebalancing")
    args = parser.parse_args(argv)

    # Fill strategy params from the model's yaml config unless overridden on the CLI
    yaml_topk, yaml_risk, yaml_n_drop, yaml_hold = (
        load_strategy_defaults(args.model) if args.model else (None, None, None, None)
    )
    topk = args.topk if args.topk is not None else (yaml_topk or spec.default_topk)
    risk_degree = args.risk_degree if args.risk_degree is not None else (yaml_risk or spec.default_risk_degree)
    n_drop, hold_thresh = _resolve_rotation(spec, args, yaml_n_drop, yaml_hold)

    global _main_pid
    _main_pid = os.getpid()
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    # Fall back to momentum signals if no trained model is available (unless --retrain will create one)
    if args.model and not args.retrain and not Path(args.model).exists():
        logger.warning(f"Model {args.model} not found — falling back to momentum signals")
        args.model = None

    coins = spec.load_coins()

    mode = "DRY RUN" if args.dry_run else "LIVE"
    rotation = "full" if n_drop is None else f"n_drop={n_drop}"
    logger.info("=" * 50)
    logger.info(f"🤖 {spec.server_title} trading server starting")
    logger.info(f"   Environment: MAINNET | Mode: {mode}")
    logger.info(f"   Coins: {len(coins)} | TopK: {topk} | Risk degree: {risk_degree}")
    logger.info(f"   Rotation: {rotation} | hold_thresh={hold_thresh}d")
    logger.info(f"   Rebalance time: daily at {args.hour:02d}:{args.minute:02d} UTC")
    logger.info("=" * 50)

    try:
        if args.dry_run:
            broker = spec.paper_broker_factory(coins)
        else:
            broker = spec.live_broker_factory()
            balances = broker.get_balances()
            cash = balances.get(spec.quote_ccy, 0)
            logger.info(f"💰 Current {spec.quote_ccy}: ${cash:,.2f}")
    except Exception as e:
        logger.error(f"Exchange connection failed: {e}")
        sys.exit(1)

    def _rebalance():
        run_rebalance(spec, broker, coins, args.dry_run, topk, risk_degree, n_drop,
                      hold_thresh, args.lookback, args.min_trade, args.model)

    if args.once:
        if args.retrain:
            retrain_model(spec, args.model)
            coins = spec.load_coins()  # rebuild_data may have refreshed the universe
        _rebalance()
        return

    # Liveness: beat once so the heartbeat file exists, then let the watchdog
    # (and the Docker healthcheck) detect a hung loop from a stale heartbeat.
    _beat()
    _start_watchdog()

    while not _shutdown:
        now = datetime.utcnow()
        target = now.replace(hour=args.hour, minute=args.minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)

        wait_seconds = (target - now).total_seconds()
        logger.info(f"⏰ Next rebalance: {target.strftime('%Y-%m-%d %H:%M:%S')} UTC "
                    f"(waiting {wait_seconds/3600:.1f}h)")

        while wait_seconds > 0 and not _shutdown:
            sleep_time = min(wait_seconds, 60)
            time.sleep(sleep_time)
            wait_seconds -= sleep_time
            _beat()

        if _shutdown:
            break

        # Beat immediately before each long blocking step so the watchdog timer
        # starts fresh for retrain and for the rebalance (each gets the full
        # timeout to complete before being judged hung).
        if args.retrain:
            _beat()
            retrain_model(spec, args.model)
            coins = spec.load_coins()
        _beat()
        _rebalance()
        _beat()

    logger.info("👋 Server shut down safely")
