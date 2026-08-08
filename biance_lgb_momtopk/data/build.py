#!/usr/bin/env python3
"""
Binance data-source hooks — moved to ``orange_quant.data.sources.BinanceSource``.

Re-exported here so existing ``python -m biance_lgb_momtopk.data.build`` and
``from biance_lgb_momtopk.data.build import get_top_symbols`` calls keep working.
"""

import argparse

from orange_quant.data.sources import BinanceSource, FetchIncomplete

__all__ = ["BinanceSource", "FetchIncomplete", "get_top_symbols", "fetch_daily",
           "fetch_hourly", "resolve_symbols", "rebuild_data", "load_coins", "main"]

# Module-level constants kept for back-compat imports
RAW_DIR = BinanceSource.raw_dir
QLIB_DIR = BinanceSource.qlib_dir
HOURLY_DIR = BinanceSource.hourly_dir


def _source() -> BinanceSource:
    return BinanceSource()


def get_top_symbols(n: int = 50) -> list:
    return _source().get_top_symbols(n)


def fetch_daily(symbol: str, start_ms: int, end_ms: int) -> list:
    return _source().fetch_daily(symbol, start_ms, end_ms)


def fetch_hourly(symbol: str, start_ms: int, end_ms: int) -> list:
    return _source().fetch_hourly(symbol, start_ms, end_ms)


def resolve_symbols(coins) -> dict:
    return _source().resolve_symbols(coins)


def rebuild_data(top: int = 50, start: str = "2020-01-01", force_download: bool = False):
    """Incrementally download data and rebuild qlib binaries."""
    from orange_quant.data import pipeline
    return pipeline.rebuild_data(_source().build_source(), top=top, start=start,
                                 force_download=force_download)


def load_coins() -> list:
    from orange_quant.data import pipeline
    return pipeline.load_coins(_source().build_source())


def main():
    parser = argparse.ArgumentParser(description="Build the Binance spot daily-bar dataset")
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--force", action="store_true", help="Force a full re-download")
    args = parser.parse_args()
    rebuild_data(top=args.top, start=args.start, force_download=args.force)


if __name__ == "__main__":
    main()
