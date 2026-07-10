#!/usr/bin/env python3
"""
Orange Quant automated trading server

Runs long-term in Docker, rebalancing on a daily schedule.
Usage:
    python -m biance_lgb_momtopk.server                    # rebalance daily at 00:15 UTC by default
    python -m biance_lgb_momtopk.server --hour 8 --minute 0  # daily at 08:00 UTC
    python -m biance_lgb_momtopk.server --dry-run           # analyze only, no orders placed
    python -m biance_lgb_momtopk.server --once             # run once then exit
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

from biance_lgb_momtopk.data import load_coins
from biance_lgb_momtopk.trading.broker import BinanceBroker, PaperBroker
from biance_lgb_momtopk.trading.runner import StrategyRunner

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("orange-quant")

# Default rebalance parameters
DEFAULT_TOP_K = 5
DEFAULT_LOOKBACK = 160
DEFAULT_MIN_TRADE = 20.0

_shutdown = False


def on_signal(signum, frame):
    global _shutdown
    logger.info(f"Received signal {signum}, shutting down safely...")
    _shutdown = True


def retrain_model(model_path: str):
    """Incrementally refresh data + retrain the model"""
    logger.info("📥 Refreshing data incrementally...")
    from biance_lgb_momtopk.data import rebuild_data
    rebuild_data()
    config_name = Path(model_path).stem
    logger.info(f"🚀 Retraining: {config_name}")
    from biance_lgb_momtopk.workflow.experiment import run_from_yaml
    run_from_yaml(f"config/{config_name}.yaml")
    logger.info("✅ Model updated")


def run_rebalance(broker, coins, dry_run, topk, lookback, min_trade, model_path=None):
    """Execute a single rebalance"""
    try:
        runner = StrategyRunner(
            broker=broker,
            coins=coins,
            topk=topk,
            lookback_days=lookback,
            min_trade_usdt=min_trade,
            model_path=model_path,
        )
        result = runner.run_once(dry_run=dry_run)

        if result["status"] != "ok":
            logger.warning(f"Rebalance issue: {result}")
            return

        balances = broker.get_balances()
        usdt = balances.get("USDT", 0)
        positions = {c: a for c, a in balances.items() if c != "USDT" and a > 0}

        # Compute total equity
        total_value = usdt
        if positions:
            symbols = [f"{c}/USDT" for c in positions.keys()]
            prices = broker.get_current_prices(symbols)
            for coin, amt in positions.items():
                p = prices.get(f"{coin}/USDT", 0)
                total_value += amt * p

        logger.info(f"💰 Total equity: ${total_value:,.2f} | USDT: ${usdt:,.2f} | Positions: {len(positions)}")
        logger.info(f"📊 Target holdings: {result['target_coins']}")
        if result["trades"]:
            for t in result["trades"]:
                logger.info(f"  Filled: {t[0]} {t[1]} ${t[2]:.2f}")
        else:
            logger.info("   No rebalance changes")

    except Exception as e:
        logger.error(f"Rebalance failed: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(description="Orange Quant Trading Server")
    parser.add_argument("--hour", type=int, default=0, help="Daily rebalance time (UTC hour)")
    parser.add_argument("--minute", type=int, default=15, help="Daily rebalance time (minute)")
    parser.add_argument("--dry-run", action="store_true", help="Analyze only, no orders placed")
    parser.add_argument("--once", action="store_true", help="Run once then exit")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOP_K, help="Number of positions to hold")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK, help="Lookback window in days")
    parser.add_argument("--min-trade", type=float, default=DEFAULT_MIN_TRADE, help="Minimum trade size in USDT")
    parser.add_argument("--model", type=str, default="models/binance-lgb-momtopk.pkl", help="LightGBM model path")
    parser.add_argument("--retrain", action="store_true", help="Refresh data and retrain the model before rebalancing")
    args = parser.parse_args()

    # Signal handling
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    # Load traded coins (from qlib instruments, falling back to the built-in list)
    coins = load_coins()
    if not coins:
        coins = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX",
                 "LINK", "DOT", "LTC", "UNI", "NEAR", "AAVE", "FIL", "INJ",
                 "TRX", "FET", "XLM", "ZEC"]

    mode = "DRY RUN" if args.dry_run else "LIVE"
    logger.info("=" * 50)
    logger.info(f"🤖 Orange Quant trading server starting")
    logger.info(f"   Environment: MAINNET | Mode: {mode}")
    logger.info(f"   Coins: {len(coins)} | TopK: {args.topk}")
    logger.info(f"   Rebalance time: daily at {args.hour:02d}:{args.minute:02d} UTC")
    logger.info("=" * 50)

    # Connect to the exchange
    try:
        if args.dry_run:
            broker = PaperBroker(coins=coins)
        else:
            broker = BinanceBroker()
            balances = broker.get_balances()
            usdt = balances.get("USDT", 0)
            logger.info(f"💰 Current USDT: ${usdt:,.2f}")
    except Exception as e:
        logger.error(f"Exchange connection failed: {e}")
        sys.exit(1)


    # Run once
    if args.once:
        if args.retrain:
            retrain_model(args.model)
        run_rebalance(broker, coins, args.dry_run, args.topk, args.lookback, args.min_trade, args.model)
        return

    # -- Run continuously --
    while not _shutdown:
        now = datetime.utcnow()
        target = now.replace(hour=args.hour, minute=args.minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)

        wait_seconds = (target - now).total_seconds()
        logger.info(f"⏰ Next rebalance: {target.strftime('%Y-%m-%d %H:%M:%S')} UTC (waiting {wait_seconds/3600:.1f}h)")

        # Sleep in chunks to support graceful shutdown
        while wait_seconds > 0 and not _shutdown:
            sleep_time = min(wait_seconds, 60)
            time.sleep(sleep_time)
            wait_seconds -= sleep_time

        if _shutdown:
            break

        if args.retrain:
            retrain_model(args.model)
        run_rebalance(broker, coins, args.dry_run, args.topk, args.lookback, args.min_trade, args.model)

    logger.info("👋 Server shut down safely")


if __name__ == "__main__":
    main()
