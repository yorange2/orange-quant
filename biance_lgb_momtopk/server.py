#!/usr/bin/env python3
"""
Orange Quant 自动交易服务器

在 Docker 中长期运行，每日定时调仓。
用法：
    python -m biance_lgb_momtopk.server                    # 默认每日 00:15 UTC 调仓
    python -m biance_lgb_momtopk.server --hour 8 --minute 0  # 每日 08:00 UTC
    python -m biance_lgb_momtopk.server --dry-run           # 只分析不下单
    python -m biance_lgb_momtopk.server --once             # 执行一次后退出
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

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("orange-quant")

# 默认调仓参数
DEFAULT_TOP_K = 5
DEFAULT_LOOKBACK = 160
DEFAULT_MIN_TRADE = 20.0

_shutdown = False


def on_signal(signum, frame):
    global _shutdown
    logger.info(f"收到信号 {signum}，准备安全退出...")
    _shutdown = True


def retrain_model(model_path: str):
    """增量更新数据 + 重新训练模型"""
    logger.info("📥 增量更新数据...")
    from biance_lgb_momtopk.data import rebuild_data
    rebuild_data()
    config_name = Path(model_path).stem
    logger.info(f"🚀 重新训练: {config_name}")
    from biance_lgb_momtopk.workflow.experiment import run_from_yaml
    run_from_yaml(f"config/{config_name}.yaml")
    logger.info("✅ 模型已更新")


def run_rebalance(broker, coins, dry_run, topk, lookback, min_trade, model_path=None):
    """执行一次调仓"""
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
            logger.warning(f"调仓异常: {result}")
            return

        balances = broker.get_balances()
        usdt = balances.get("USDT", 0)
        positions = {c: a for c, a in balances.items() if c != "USDT" and a > 0}

        # 计算总资产
        total_value = usdt
        if positions:
            symbols = [f"{c}/USDT" for c in positions.keys()]
            prices = broker.get_current_prices(symbols)
            for coin, amt in positions.items():
                p = prices.get(f"{coin}/USDT", 0)
                total_value += amt * p

        logger.info(f"💰 总资产: ${total_value:,.2f} | USDT: ${usdt:,.2f} | 持仓: {len(positions)} 个")
        logger.info(f"📊 目标持仓: {result['target_coins']}")
        if result["trades"]:
            for t in result["trades"]:
                logger.info(f"  成交: {t[0]} {t[1]} ${t[2]:.2f}")
        else:
            logger.info("   无调仓变动")

    except Exception as e:
        logger.error(f"调仓失败: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(description="Orange Quant Trading Server")
    parser.add_argument("--hour", type=int, default=0, help="每日调仓时间 (UTC 小时)")
    parser.add_argument("--minute", type=int, default=15, help="每日调仓时间 (分钟)")
    parser.add_argument("--dry-run", action="store_true", help="只分析不下单")
    parser.add_argument("--once", action="store_true", help="执行一次后退出")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOP_K, help="持仓数量")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK, help="回看天数")
    parser.add_argument("--min-trade", type=float, default=DEFAULT_MIN_TRADE, help="最小交易金额 USDT")
    parser.add_argument("--model", type=str, default="models/binance-lgb-momtopk.pkl", help="LightGBM 模型路径")
    parser.add_argument("--retrain", action="store_true", help="调仓前先更新数据并重新训练模型")
    args = parser.parse_args()

    # 信号处理
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    # 加载交易币种（从 qlib instruments，回退到内置列表）
    coins = load_coins()
    if not coins:
        coins = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX",
                 "LINK", "DOT", "LTC", "UNI", "NEAR", "AAVE", "FIL", "INJ",
                 "TRX", "FET", "XLM", "ZEC"]

    mode = "DRY RUN" if args.dry_run else "LIVE"
    logger.info("=" * 50)
    logger.info(f"🤖 Orange Quant 交易服务器启动")
    logger.info(f"   环境: MAINNET | 模式: {mode}")
    logger.info(f"   币种: {len(coins)} | TopK: {args.topk}")
    logger.info(f"   调仓时间: 每日 {args.hour:02d}:{args.minute:02d} UTC")
    logger.info("=" * 50)

    # 连接交易所
    try:
        if args.dry_run:
            broker = PaperBroker(coins=coins)
        else:
            broker = BinanceBroker()
            balances = broker.get_balances()
            usdt = balances.get("USDT", 0)
            logger.info(f"💰 当前 USDT: ${usdt:,.2f}")
    except Exception as e:
        logger.error(f"交易所连接失败: {e}")
        sys.exit(1)


    # 执行一次
    if args.once:
        if args.retrain:
            retrain_model(args.model)
        run_rebalance(broker, coins, args.dry_run, args.topk, args.lookback, args.min_trade, args.model)
        return

    # ── 持续运行 ──
    while not _shutdown:
        now = datetime.utcnow()
        target = now.replace(hour=args.hour, minute=args.minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)

        wait_seconds = (target - now).total_seconds()
        logger.info(f"⏰ 下次调仓: {target.strftime('%Y-%m-%d %H:%M:%S')} UTC (等待 {wait_seconds/3600:.1f}h)")

        # 分段 sleep，支持 graceful shutdown
        while wait_seconds > 0 and not _shutdown:
            sleep_time = min(wait_seconds, 60)
            time.sleep(sleep_time)
            wait_seconds -= sleep_time

        if _shutdown:
            break

        if args.retrain:
            retrain_model(args.model)
        run_rebalance(broker, coins, args.dry_run, args.topk, args.lookback, args.min_trade, args.model)

    logger.info("👋 服务器已安全退出")


if __name__ == "__main__":
    main()
