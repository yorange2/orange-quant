#!/usr/bin/env python3
"""
Train the LightGBM model (exchange-agnostic).

Entrypoint: ``run(default_config, argv)``. The per-exchange package passes its
default config name; a positional arg overrides it.
"""

import sys
import argparse
from pathlib import Path

from orange_quant.experiment import run_from_yaml


def run(default_config: str = "csi300-lgb-momtopk", argv=None):
    parser = argparse.ArgumentParser(description="Orange Quant model training")
    parser.add_argument(
        "config", nargs="?", default=default_config,
        help="Config name (without .yaml)",
    )
    args = parser.parse_args(argv)

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

    return results
