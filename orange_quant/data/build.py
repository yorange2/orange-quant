"""Unified dataset build entry: ``python -m orange_quant.data.build --exchange ...``.

Venue-specific fetch hooks live in ``orange_quant.data.sources``.
"""

import argparse

from orange_quant.data.sources import BinanceSource, DataSourceHooks, HyperliquidSource

_SOURCES = {
    "binance": BinanceSource,
    "hyperliquid": HyperliquidSource,
}


def get_source(exchange: str) -> DataSourceHooks:
    """Venue name → data-source hooks. The single venue→source dispatch point."""
    try:
        return _SOURCES[exchange.lower()]()
    except KeyError:
        raise ValueError(
            f"unknown exchange '{exchange}'. Available: {', '.join(_SOURCES)}") from None


def main():
    parser = argparse.ArgumentParser(description="Build the exchange spot bar dataset")
    parser.add_argument("--exchange", default="binance", choices=list(_SOURCES),
                        help="Which venue to build (default: binance)")
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--freq", choices=["1d", "1h"], default="1d",
                        help="Bar frequency (default: 1d; 1h → data/{venue}_h1_raw)")
    parser.add_argument("--force", action="store_true", help="Force a full re-download")
    args = parser.parse_args()

    source = get_source(args.exchange)
    print("=" * 60)
    print(f"📥 Building the {source.label} spot {args.freq}-bar dataset (Top {args.top})")
    print("=" * 60)

    from orange_quant.data import pipeline
    pipeline.rebuild_data(source.build_source(), top=args.top, start=args.start,
                          force_download=args.force, freq=args.freq)


if __name__ == "__main__":
    main()
