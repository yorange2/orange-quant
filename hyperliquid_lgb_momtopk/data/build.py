#!/usr/bin/env python3
"""
Hyperliquid data-source hooks — moved to ``orange_quant.data.sources.HyperliquidSource``.

Re-exported here so existing ``python -m hyperliquid_lgb_momtopk.data.build`` and
``from hyperliquid_lgb_momtopk.data.build import get_top_symbols`` calls keep working.
"""

import argparse

from orange_quant.data.sources import HyperliquidSource

__all__ = ["HyperliquidSource", "get_top_symbols", "fetch_daily", "fetch_hourly",
           "resolve_symbols", "rebuild_data", "load_coins", "main"]

# Module-level constants kept for back-compat imports
RAW_DIR = HyperliquidSource.raw_dir
QLIB_DIR = HyperliquidSource.qlib_dir
HOURLY_DIR = HyperliquidSource.hourly_dir


def _source() -> HyperliquidSource:
    return HyperliquidSource()


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
    return _source().rebuild_data(top=top, start=start, force_download=force_download)


def load_coins() -> list:
    from orange_quant.data import pipeline
    return pipeline.load_coins(_source().build_source())


def main():
    parser = argparse.ArgumentParser(description="Build the Hyperliquid spot daily-bar dataset")
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--force", action="store_true", help="Force a full re-download")
    args = parser.parse_args()
    rebuild_data(top=args.top, start=args.start, force_download=args.force)


if __name__ == "__main__":
    main()
