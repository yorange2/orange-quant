---
name: oq-download-data
description: Download daily bars — A-shares from the Tencent K-line API, crypto from Binance REST / Hyperliquid ccxt — into per-symbol CSVs
argument-hint: "[--exchange binance|hyperliquid|tencent]"
---

# Download Data

Fetch daily bars into raw CSVs (gitignored). All fetches are resumable
(existing CSVs are skipped) and idempotent.

## A-shares (Tencent)

```bash
# full market (needs a symbol list: TSV/CSV whose first column is SH/SZ codes)
python -m orange_quant.data.tencent --symbols-file <symbols.tsv> --workers 8

# smoke test (first N symbols, force re-fetch)
python -m orange_quant.data.tencent --symbols-file <symbols.tsv> --limit 5 --force

# benchmark index only
python -m orange_quant.data.tencent --index-only
```

Output: `data/cn_raw/{SYMBOL}.csv` with `date,symbol,open,high,low,close,volume,amount`.
hfq (backward-adjusted) prices; volume in shares (lots ×100 except STAR/indices);
amount in CNY. **Pagination is end-anchored** — never request with year loops.

## Crypto

```bash
python -m orange_quant.data.build --exchange binance --top 20 --start 2019-07-01
python -m orange_quant.data.build --exchange hyperliquid --top 20 --start 2019-07-01
```

Output: `data/binance_raw/{COIN}.csv` / `data/hyperliquid_raw/...` with
`date,open,close,high,low,volume`. Incremental: rerunning only fetches new days.
