#!/usr/bin/env python3
"""
Train the LightGBM model — Hyperliquid spot.

Thin entrypoint delegating to ``orange_quant.train``.

Usage:
    python -m hyperliquid_lgb_momtopk.train   # hyperliquid-lgb-momtopk
"""

from orange_quant.train import run


def main():
    run("hyperliquid-lgb-momtopk")


if __name__ == "__main__":
    main()
