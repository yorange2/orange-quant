---
name: oq-download-data
description: Download the datasets needed for quant experiments, supporting A-shares and Binance spot daily bars
---

# Download Datasets

Download the datasets needed for quant experiments. Supports A-shares and Binance spot daily bars.

## Trigger conditions
- "download data" / "download data"
- "build dataset" / "build dataset"

## Available datasets

### A-shares (official qlib data, skip if already downloaded)

```bash
source .venv/bin/activate
python scripts/csi300/build_data.py
```

Data location: `data/qlib_data/cn_data/` (~1-2 GB)

### Exchange spot daily bars (top 50 by volume)

```bash
source .venv/bin/activate
python -m orange_quant.data.build --exchange binance --top 50     # Binance USDT spot
python -m orange_quant.data.build --exchange hyperliquid --top 50 # Hyperliquid USDC spot
```

Legacy entries still work (thin shims): `python -m biance_lgb_momtopk.data.build --top 50`,
`python -m hyperliquid_lgb_momtopk.data.build --top 50`.

Data location: `data/qlib_data/binance/` or `data/qlib_data/hyperliquid/` (qlib format),
`data/{binance,hyperliquid}_raw/` (raw CSV)

## Notes

- The first download requires internet access; time depends on network speed
- Binance data comes from the public API, no API key required
- The data directory is in `.gitignore` and won't be committed
