#!/usr/bin/env python3
"""
Orange Quant Hyperliquid spot automated trading server.

Thin entrypoint: the daily rebalance loop lives in ``orange_quant.server``; this
module only supplies the Hyperliquid ``ExchangeSpec``. Kept as a module so the
existing ``python -m hyperliquid_lgb_momtopk.server`` entrypoint (Docker) is unchanged.
"""

from orange_quant.server import run
from hyperliquid_lgb_momtopk.spec import SPEC


def main():
    run(SPEC)


if __name__ == "__main__":
    main()
