"""Unified dataset build entry: ``python -m orange_quant.data.build --exchange ...``.

Replaces the per-exchange ``biance_lgb_momtopk.data.build`` /
``hyperliquid_lgb_momtopk.data.build`` entry points (kept as thin shims for
back-compat). Venue-specific fetch hooks live in ``orange_quant.data.sources``.
"""

import argparse
import sys

from orange_quant.data.sources import BinanceSource, DataSourceHooks, HyperliquidSource

_SOURCES = {
    "binance": BinanceSource,
    "hyperliquid": HyperliquidSource,
}


def get_source(exchange: str) -> DataSourceHooks:
    try:
        return _SOURCES[exchange.lower()]()
    except KeyError:
        print(f"❌ Unknown exchange '{exchange}'. Available: {', '.join(_SOURCES)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Build the exchange spot daily-bar dataset")
    parser.add_argument("--exchange", default="binance", choices=list(_SOURCES),
                        help="Which venue to build (default: binance)")
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--force", action="store_true", help="Force a full re-download")
    args = parser.parse_args()

    source = get_source(args.exchange)
    print("=" * 60)
    print(f"📥 Building the {source.label} spot daily-bar dataset (Top {args.top})")
    print("=" * 60)

    if hasattr(source, "rebuild_data"):
        source.rebuild_data(top=args.top, start=args.start, force_download=args.force)
    else:
        from orange_quant.data import pipeline
        pipeline.rebuild_data(source.build_source(), top=args.top, start=args.start,
                              force_download=args.force)


if __name__ == "__main__":
    main()
