---
name: oq-live-trade
description: 本地实盘/模拟自动交易，支持分析和下单
---

# Live Trading

本地实盘/模拟自动交易。支持分析和下单两种模式。

## 触发条件
- "实盘交易" / "下单" / "调仓" / "trade"
- "查看持仓" / "check positions"

## 前置条件

1. `.env` 文件已配置 `BINANCE_API_KEY` 和 `BIANCE_SECRET_KEY`

## 运行

```bash
source .venv/bin/activate

# DRY RUN 模式（只分析不下单，推荐先跑）
python scripts/biance/execute.py --once --dry-run

# 实际下单（主网）
python scripts/biance/execute.py --once

# 使用 LightGBM 模型预测
python scripts/biance/execute.py --once --model models/binance-lgb-momtopk.pkl
```

## 当前持仓

```bash
source .venv/bin/activate
python -c "
from dotenv import load_dotenv; load_dotenv(override=True)
from orange_quant.trading.broker import BinanceBroker
broker = BinanceBroker(testnet=False, paper=False)
balances = broker.get_balances()
for a, amt in sorted(balances.items()):
    if amt > 0.0001:
        p = broker.get_current_prices([f'{a}/USDT']).get(f'{a}/USDT', 0) if a != 'USDT' else 1
        print(f'  {a}: {amt:.6f} (≈\${amt*p:,.2f})')
"
```

## 安全提醒

- 市价单，立即成交
- 最小交易金额 $20 USDT（低于此值不交易）
- 单币种最大仓位 25%
- TRX 等小数位多的币种可能有 dust 残留（自动跳过）
