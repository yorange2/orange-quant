#!/usr/bin/env python3
"""
Hyperliquid data-source hooks.

Only the fetch source (top-symbol ranking + daily-bar download) is Hyperliquid-
specific; the incremental download loop and qlib-binary build live in
``orange_quant.data.pipeline``. ``fetch_daily`` returns uniform rows
``[timestamp_ms, open, high, low, close, volume]`` (closed bars only).

Usage:
    python -m hyperliquid_lgb_momtopk.data.build          # top 50 by default
    python -m hyperliquid_lgb_momtopk.data.build --top 100
    python -m hyperliquid_lgb_momtopk.data.build --force   # force a full re-download
"""

import time
import argparse
from pathlib import Path

from orange_quant.data import pipeline

RAW_DIR = Path("data/hyperliquid_raw")
QLIB_DIR = Path("data/qlib_data/hyperliquid")
_REQUEST_DELAY = 0.3

# Stablecoins / fiat-pegged bases to exclude from the universe
_SKIP_BASES = {"USDT", "USDE", "USDH", "USDHL", "FEUSD", "USR", "DAI", "BUIDL", "USDXL"}
_FALLBACK = ["HYPE", "PURR", "BTC", "ETH", "SOL"]

_exchange = None


def _get_exchange():
    """Lazily create a shared public ccxt.hyperliquid client."""
    global _exchange
    if _exchange is None:
        import ccxt
        _exchange = ccxt.hyperliquid({"enableRateLimit": True, "timeout": 30000})
        _exchange.load_markets()
    return _exchange


def get_top_symbols(n: int = 50) -> list:
    """Get the top-N USDC spot pairs by volume on Hyperliquid.

    Returns [(symbol, coin)] where symbol is a ccxt symbol like "HYPE/USDC"
    and coin is the ccxt base code (UBTC -> BTC, UETH -> ETH, USOL -> SOL).
    """
    ex = _get_exchange()
    tickers = ex.fetch_tickers()

    ranked = []
    for sym, t in tickers.items():
        market = ex.markets.get(sym)
        if not market or not market.get("spot") or market.get("quote") != "USDC":
            continue
        base = market["base"]
        if base in _SKIP_BASES:
            continue
        vol = t.get("quoteVolume")
        if vol is None:
            vol = float(t.get("info", {}).get("dayNtlVlm", 0) or 0)
        ranked.append((sym, base, float(vol)))

    ranked.sort(key=lambda x: x[2], reverse=True)
    return [(sym, base) for sym, base, _ in ranked[:n]]


def fetch_daily(symbol: str, start_ms: int, end_ms: int) -> list:
    """Fetch daily spot candles via ccxt (auto-paginated).

    Returns uniform rows [timestamp_ms, open, high, low, close, volume] (closed bars).
    """
    ex = _get_exchange()
    all_rows = []
    batch_start = start_ms

    while batch_start < end_ms:
        try:
            rows = ex.fetch_ohlcv(symbol, "1d", since=batch_start, limit=1000)
        except Exception as e:
            print(f"  API err: {e}")
            break

        if not rows:
            break

        all_rows.extend(rows)
        last_time = rows[-1][0]
        if last_time <= batch_start:
            break
        batch_start = last_time + 86400000
        time.sleep(_REQUEST_DELAY)

    now_ms = int(time.time() * 1000)
    return [r for r in all_rows if r[0] + 86400000 <= now_ms]


_SOURCE = pipeline.DataSource(
    label="Hyperliquid",
    raw_dir=RAW_DIR,
    qlib_dir=QLIB_DIR,
    get_top_symbols=get_top_symbols,
    fetch_daily=fetch_daily,
    fallback_coins=_FALLBACK,
)


def rebuild_data(top: int = 50, start: str = "2020-01-01", force_download: bool = False):
    """Incrementally download data and rebuild qlib binaries."""
    return pipeline.rebuild_data(_SOURCE, top=top, start=start, force_download=force_download)


def load_coins() -> list:
    """Active coin list (qlib instruments -> raw CSVs -> live top pairs -> static)."""
    return pipeline.load_coins(_SOURCE)


def main():
    parser = argparse.ArgumentParser(description="Build the Hyperliquid spot daily-bar dataset")
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--force", action="store_true", help="Force a full re-download")
    args = parser.parse_args()

    print("=" * 60)
    print(f"📥 Building the Hyperliquid spot daily-bar dataset (Top {args.top})")
    print("=" * 60)

    rebuild_data(top=args.top, start=args.start, force_download=args.force)


if __name__ == "__main__":
    main()
