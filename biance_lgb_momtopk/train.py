#!/usr/bin/env python3
"""
Train the LightGBM model (Binance default).

Thin entrypoint delegating to ``orange_quant.train``.

Usage:
    python -m biance_lgb_momtopk.train                     # binance-lgb-momtopk
    python -m biance_lgb_momtopk.train csi300-lgb-momtopk  # A-shares
"""

from orange_quant.train import run


def main():
    run("binance-lgb-momtopk")


if __name__ == "__main__":
    main()
