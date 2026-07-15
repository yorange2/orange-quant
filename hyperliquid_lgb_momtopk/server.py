#!/usr/bin/env python3
"""
Orange Quant Hyperliquid spot automated trading server

Runs long-term in Docker, rebalancing on a daily schedule.
Usage:
    python -m hyperliquid_lgb_momtopk.server                 # rebalance daily at 00:15 UTC by default
    python -m hyperliquid_lgb_momtopk.server --dry-run       # analyze only, no orders placed
    python -m hyperliquid_lgb_momtopk.server --once          # run once then exit
"""

import sys
import time
import signal
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv(override=True)

from hyperliquid_lgb_momtopk.data import load_coins, rebuild_data
from hyperliquid_lgb_momtopk.trading.broker import HyperliquidBroker, PaperBroker
from hyperliquid_lgb_momtopk.trading.runner import StrategyRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("orange-quant-hl")

DEFAULT_TOP_K = 5
DEFAULT_LOOKBACK = 160
DEFAULT_MIN_TRADE = 20.0
DEFAULT_RISK_DEGREE = 0.95


def load_strategy_defaults(model_path: str):
    """Read topk / risk_degree from the yaml config matching the model,
    so live trading uses the same strategy parameters as the backtest."""
    try:
        import yaml
        cfg_path = Path("config") / f"{Path(model_path).stem}.yaml"
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text())
            kwargs = cfg.get("strategy", {}).get("kwargs", {})
            return kwargs.get("topk"), kwargs.get("risk_degree")
    except Exception as e:
        logger.warning(f"Failed to read strategy config for {model_path}: {e}")
    return None, None

_shutdown = False


def on_signal(signum, frame):
    global _shutdown
    logger.info(f"Received signal {signum}, shutting down safely...")
    _shutdown = True


def retrain_model(model_path: str):
    """Incrementally refresh data + retrain the model"""
    logger.info("📥 Refreshing data incrementally...")
    rebuild_data()
    config_name = Path(model_path).stem
    logger.info(f"🚀 Retraining: {config_name}")
    from hyperliquid_lgb_momtopk.workflow.experiment import run_from_yaml
    run_from_yaml(f"config/{config_name}.yaml")
    logger.info("✅ Model updated")


def run_rebalance(broker, coins, dry_run, topk, lookback, min_trade, model_path=None,
                  risk_degree=DEFAULT_RISK_DEGREE):
    """Execute a single rebalance"""
    try:
        runner = StrategyRunner(
            broker=broker,
            coins=coins,
            topk=topk,
            lookback_days=lookback,
            min_trade_usdc=min_trade,
            risk_degree=risk_degree,
            model_path=model_path,
        )
        result = runner.run_once(dry_run=dry_run)

        if result["status"] != "ok":
            logger.warning(f"Rebalance issue: {result}")
            return

        balances = broker.get_balances()
        usdc = balances.get("USDC", 0)
        positions = {c: a for c, a in balances.items() if c != "USDC" and a > 0}

        total_value = usdc
        if positions:
            prices = broker.get_current_prices(list(positions.keys()))
            for coin, amt in positions.items():
                p = prices.get(coin, 0)
                total_value += amt * p

        logger.info(f"💰 Total equity: ${total_value:,.2f} | USDC: ${usdc:,.2f} | Positions: {len(positions)}")
        logger.info(f"📊 Target holdings: {result['target_coins']}")
        if result["trades"]:
            for t in result["trades"]:
                logger.info(f"  Filled: {t[0]} {t[1]} ${t[2]:.2f}")
        else:
            logger.info("   No rebalance changes")

    except Exception as e:
        logger.error(f"Rebalance failed: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(description="Orange Quant Hyperliquid Trading Server")
    parser.add_argument("--hour", type=int, default=0, help="Daily rebalance time (UTC hour)")
    parser.add_argument("--minute", type=int, default=15, help="Daily rebalance time (minute)")
    parser.add_argument("--dry-run", action="store_true", help="Analyze only, no orders placed")
    parser.add_argument("--once", action="store_true", help="Run once then exit")
    parser.add_argument("--topk", type=int, default=None,
                        help="Number of positions to hold (default: strategy topk from the model's yaml config)")
    parser.add_argument("--risk-degree", type=float, default=None,
                        help="Fraction of equity to deploy (default: risk_degree from the model's yaml config)")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK, help="Lookback window in days")
    parser.add_argument("--min-trade", type=float, default=DEFAULT_MIN_TRADE, help="Minimum trade size in USDC")
    parser.add_argument("--model", type=str, default="models/hyperliquid-lgb-momtopk.pkl", help="LightGBM model path")
    parser.add_argument("--retrain", action="store_true", help="Refresh data and retrain the model before rebalancing")
    args = parser.parse_args()

    # Fill topk / risk_degree from the model's yaml config unless overridden on the CLI
    yaml_topk, yaml_risk = load_strategy_defaults(args.model) if args.model else (None, None)
    topk = args.topk if args.topk is not None else (yaml_topk or DEFAULT_TOP_K)
    risk_degree = args.risk_degree if args.risk_degree is not None else (yaml_risk or DEFAULT_RISK_DEGREE)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    # Fall back to momentum signals if no trained model is available (unless --retrain will create one)
    if args.model and not args.retrain and not Path(args.model).exists():
        logger.warning(f"Model {args.model} not found — falling back to momentum signals")
        args.model = None

    # Load traded coins (qlib instruments, with fallbacks inside load_coins)
    coins = load_coins()

    mode = "DRY RUN" if args.dry_run else "LIVE"
    logger.info("=" * 50)
    logger.info(f"🤖 Orange Quant Hyperliquid trading server starting")
    logger.info(f"   Environment: MAINNET | Mode: {mode}")
    logger.info(f"   Coins: {len(coins)} | TopK: {topk} | Risk degree: {risk_degree}")
    logger.info(f"   Rebalance time: daily at {args.hour:02d}:{args.minute:02d} UTC")
    logger.info("=" * 50)

    try:
        if args.dry_run:
            broker = PaperBroker(coins=coins)
        else:
            broker = HyperliquidBroker()
            balances = broker.get_balances()
            usdc = balances.get("USDC", 0)
            logger.info(f"💰 Current USDC: ${usdc:,.2f}")
    except Exception as e:
        logger.error(f"Exchange connection failed: {e}")
        sys.exit(1)

    if args.once:
        if args.retrain:
            retrain_model(args.model)
            # rebuild_data() may have refreshed the instruments list
            coins = load_coins()
        run_rebalance(broker, coins, args.dry_run, topk, args.lookback, args.min_trade, args.model,
                      risk_degree=risk_degree)
        return

    while not _shutdown:
        now = datetime.utcnow()
        target = now.replace(hour=args.hour, minute=args.minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)

        wait_seconds = (target - now).total_seconds()
        logger.info(f"⏰ Next rebalance: {target.strftime('%Y-%m-%d %H:%M:%S')} UTC (waiting {wait_seconds/3600:.1f}h)")

        while wait_seconds > 0 and not _shutdown:
            sleep_time = min(wait_seconds, 60)
            time.sleep(sleep_time)
            wait_seconds -= sleep_time

        if _shutdown:
            break

        if args.retrain:
            retrain_model(args.model)
            coins = load_coins()
        run_rebalance(broker, coins, args.dry_run, topk, args.lookback, args.min_trade, args.model,
                      risk_degree=risk_degree)

    logger.info("👋 Server shut down safely")


if __name__ == "__main__":
    main()
