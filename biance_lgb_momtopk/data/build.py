#!/usr/bin/env python3
"""
Binance data-source hooks.

Only the fetch source (top-symbol ranking + daily-bar download) is Binance-specific;
the incremental download loop and qlib-binary build live in
``orange_quant.data.pipeline``. ``fetch_daily`` returns uniform rows
``[timestamp_ms, open, high, low, close, volume]`` (closed bars only).

Usage:
    python -m biance_lgb_momtopk.data.build          # top 50 by default
    python -m biance_lgb_momtopk.data.build --top 100
    python -m biance_lgb_momtopk.data.build --force   # force a full re-download
"""

import time
import argparse
from pathlib import Path

import requests

from orange_quant.data import hourly, pipeline

_BINANCE_API = "https://api.binance.com/api/v3"
RAW_DIR = Path("data/binance_raw")
QLIB_DIR = Path("data/qlib_data/binance")
HOURLY_DIR = Path("data/binance_hourly")
_REQUEST_DELAY = 0.3

_SKIP = {
    "USDCUSDT", "USDTUSDT", "TUSDUSDT", "BUSDUSDT", "DAIUSDT",
    "PAXUSDT", "USD1USDT", "FDUSDUSDT", "RLUSDUSDT", "EURUSDT",
    "XAUTUSDT", "PAXGUSDT",
    "UUSDT",  # trade-restricted on Binance (reduce-only), orders get rejected
}

_FALLBACK = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX",
             "LINK", "DOT", "LTC", "UNI", "NEAR", "AAVE", "FIL", "INJ",
             "TRX", "FET", "XLM", "ZEC"]


def get_top_symbols(n: int = 50) -> list:
    """Get the top-N USDT spot trading pairs by volume on Binance.

    Returns [(symbol, coin)] where symbol is the Binance REST symbol ("BTCUSDT")
    and coin is the base ("BTC").
    """
    tickers = requests.get(f"{_BINANCE_API}/ticker/24hr", timeout=10).json()
    usdt = [(t["symbol"], float(t["quoteVolume"]))
            for t in tickers if t["symbol"].endswith("USDT")]
    usdt.sort(key=lambda x: x[1], reverse=True)

    result = []
    for symbol, vol in usdt:
        base = symbol.replace("USDT", "")
        if symbol in _SKIP:
            continue
        if any(x in base for x in ("UP", "DOWN", "BULL", "BEAR")):
            continue
        result.append((symbol, base))
        if len(result) >= n:
            break
    return result


def _fetch_klines(symbol: str, interval: str, step_ms: int,
                  start_ms: int, end_ms: int) -> list:
    """Fetch klines from the Binance API (auto-paginated).

    Returns uniform rows [timestamp_ms, open, high, low, close, volume] for bars
    that have already closed (Binance closeTime <= now).
    """
    all_candles = []
    batch_start = start_ms
    while batch_start < end_ms:
        params = {"symbol": symbol, "interval": interval,
                  "startTime": batch_start, "endTime": end_ms, "limit": 1000}
        try:
            resp = requests.get(f"{_BINANCE_API}/klines", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  API err: {e}")
            break
        if not data or not isinstance(data, list):
            break
        all_candles.extend(data)
        last_time = data[-1][0]
        if last_time <= batch_start:
            break
        batch_start = last_time + step_ms
        time.sleep(_REQUEST_DELAY)

    # Keep only closed bars, and reshape kline -> uniform [ts, o, h, l, c, v]
    now_ms = int(time.time() * 1000)
    return [[c[0], c[1], c[2], c[3], c[4], c[5]] for c in all_candles if c[6] <= now_ms]


def fetch_daily(symbol: str, start_ms: int, end_ms: int) -> list:
    """Fetch daily bars (uniform rows, closed bars only)."""
    return _fetch_klines(symbol, "1d", 86400000, start_ms, end_ms)


def fetch_hourly(symbol: str, start_ms: int, end_ms: int) -> list:
    """Fetch 1h bars (uniform rows, closed bars only).

    Binance paginates back to listing, so the full history is available — this
    is what the phase resampler in ``orange_quant.data.hourly`` builds on.
    """
    return _fetch_klines(symbol, "1h", 3600000, start_ms, end_ms)


def resolve_symbols(coins) -> dict:
    """coin -> Binance REST symbol. Every USDT pair is just <coin>USDT."""
    return {coin: f"{coin}USDT" for coin in coins}


_SOURCE = pipeline.DataSource(
    label="Binance",
    raw_dir=RAW_DIR,
    qlib_dir=QLIB_DIR,
    get_top_symbols=get_top_symbols,
    fetch_daily=fetch_daily,
    fallback_coins=_FALLBACK,
)


HOURLY_SOURCE = hourly.HourlySource(
    label="Binance",
    hourly_dir=HOURLY_DIR,
    daily_qlib_dir=QLIB_DIR,
    daily_raw_dir=RAW_DIR,
    phase_raw_tmpl="data/binance_phase{phase:02d}_raw",
    phase_qlib_tmpl="data/qlib_data/binance_h{phase:02d}",
    fetch_hourly=fetch_hourly,
    resolve_symbols=resolve_symbols,
)


def rebuild_data(top: int = 50, start: str = "2020-01-01", force_download: bool = False):
    """Incrementally download data and rebuild qlib binaries."""
    return pipeline.rebuild_data(_SOURCE, top=top, start=start, force_download=force_download)


def load_coins() -> list:
    """Active coin list (qlib instruments -> raw CSVs -> live top pairs -> static)."""
    return pipeline.load_coins(_SOURCE)


def main():
    parser = argparse.ArgumentParser(description="Build the Binance spot daily-bar dataset")
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--force", action="store_true", help="Force a full re-download")
    args = parser.parse_args()

    print("=" * 60)
    print(f"📥 Building the Binance spot daily-bar dataset (Top {args.top})")
    print("=" * 60)

    rebuild_data(top=args.top, start=args.start, force_download=args.force)


if __name__ == "__main__":
    main()
