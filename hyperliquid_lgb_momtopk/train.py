#!/usr/bin/env python3
"""
Train the LightGBM model — Hyperliquid perpetuals

Usage:
    python -m hyperliquid_lgb_momtopk.train hyperliquid-lgb-momtopk
"""

import sys
import argparse
from pathlib import Path

from hyperliquid_lgb_momtopk.workflow.experiment import run_from_yaml


def main():
    parser = argparse.ArgumentParser(description="Orange Quant Hyperliquid model training")
    parser.add_argument(
        "config", nargs="?", default="hyperliquid-lgb-momtopk",
        help="Config name (without .yaml)",
    )
    args = parser.parse_args()

    config_path = f"config/{args.config}.yaml"
    if not Path(config_path).exists():
        available = [p.stem for p in Path("config").glob("*.yaml")]
        print(f"❌ Config not found: {config_path}")
        print(f"Available: {', '.join(available)}")
        sys.exit(1)

    print(f"🚀 Training: {args.config}")
    results = run_from_yaml(config_path)
    r = results["recorder"]

    print("\n📊 Training results")
    for k, v in r.list_metrics().items():
        if "IC" in k:
            print(f"  {k}: {v:.4f}")
    for k, v in r.list_metrics().items():
        if "annualized" in k:
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
