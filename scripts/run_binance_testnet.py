#!/usr/bin/env python3
"""
Binance Testnet 自动交易测试

在 Binance 测试网上运行自动交易策略（DRY RUN 模式，不下单），
验证连接、数据获取、信号计算等完整流程。

用法：
    python scripts/run_binance_testnet.py          # 单次分析
    python scripts/run_binance_testnet.py --trade   # 实际下单（测试网！）
    python scripts/run_binance_testnet.py --loop    # 持续运行
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orange_quant.trading.broker import BinanceBroker
from orange_quant.trading.runner import StrategyRunner


def main():
    parser = argparse.ArgumentParser(description="Binance Testnet Auto Trading")
    parser.add_argument("--trade", action="store_true", help="实际下单（默认 DRY RUN）")
    parser.add_argument("--loop", action="store_true", help="持续运行模式")
    parser.add_argument("--topk", type=int, default=8, help="持仓数量")
    args = parser.parse_args()

    # 使用训练中表现好的蓝筹币种
    coins = [
        "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX",
        "LINK", "DOT", "LTC", "UNI", "NEAR", "AAVE", "FIL", "INJ",
        "TRX", "FET", "XLM", "ZEC",
    ]

    print("=" * 60)
    print("🤖 Orange Quant 自动交易系统")
    print("=" * 60)

    # 连接交易所
    broker = BinanceBroker(testnet=True)

    # 显示余额
    balances = broker.get_balances()
    print(f"\n💰 账户余额:")
    for asset, amt in sorted(balances.items()):
        if asset == "USDT":
            print(f"  {asset}: {amt:.2f}")
        elif amt > 0.0001:
            print(f"  {asset}: {amt:.6f}")

    # 创建策略执行器
    runner = StrategyRunner(
        broker=broker,
        coins=coins,
        topk=args.topk,
        lookback_days=30,
        rebalance_interval_hours=24,
    )

    dry_run = not args.trade

    if args.loop:
        runner.run_loop(dry_run=dry_run)
    else:
        result = runner.run_once(dry_run=dry_run)
        if result["status"] == "ok":
            sig = result["signals"]
            print(f"\n📊 完整排名:")
            for _, row in sig.iterrows():
                flag = "✅" if row["coin"] in result["target_coins"] else "  "
                print(f"  {flag} {row['rank']:2.0f}. {row['coin']:8s}  "
                      f"score={row['score']:.4f}  price=\${row['price']:.4f}")


if __name__ == "__main__":
    main()
