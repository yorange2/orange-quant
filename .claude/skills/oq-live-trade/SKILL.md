---
name: oq-live-trade
description: Local live/simulated automated trading, supporting analysis and order placement
---

# Live Trading

Local live/simulated automated trading. Supports both analysis-only and order-placement modes.

## Trigger conditions
- "live trading" / "place order" / "rebalance" / "trade"
- "check positions" / "check positions"

## Prerequisites

1. `.env` file configured with `BINANCE_API_KEY` and `BIANCE_SECRET_KEY`

## Running

```bash
source .venv/bin/activate

# DRY RUN mode (analyze only, no orders placed; recommended to run first)
python -m biance_lgb_momtopk.server --once --dry-run

# Place real orders (mainnet)
python -m biance_lgb_momtopk.server --once

# Use LightGBM model predictions
python -m biance_lgb_momtopk.server --once --model models/binance-lgb-momtopk.pkl
```

## Current holdings

```bash
source .venv/bin/activate
python -c "
from dotenv import load_dotenv; load_dotenv(override=True)
from biance_lgb_momtopk.trading.broker import BinanceBroker
broker = BinanceBroker(testnet=False, paper=False)
balances = broker.get_balances()
for a, amt in sorted(balances.items()):
    if amt > 0.0001:
        p = broker.get_current_prices([f'{a}/USDT']).get(f'{a}/USDT', 0) if a != 'USDT' else 1
        print(f'  {a}: {amt:.6f} (≈\${amt*p:,.2f})')
"
```

## Safety notes

- Market orders, fill immediately
- Minimum trade size $20 USDT (no trade below this)
- Maximum position size per coin: 25%
- Coins with many decimal places (e.g. TRX) may leave dust residue (automatically skipped)
