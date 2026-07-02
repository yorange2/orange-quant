#!/usr/bin/env python3
"""
训练 LightGBM 模型

用法:
    python scripts/biance/train.py                            # 默认 binance-lgb-momtopk
    python scripts/biance/train.py binance-lgb-momtopk        # Binance
    python scripts/biance/train.py csi300-lgb-momtopk         # A 股
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from orange_quant.workflow.experiment import run_from_yaml


def main():
    parser = argparse.ArgumentParser(description="Orange Quant 模型训练")
    parser.add_argument(
        "config", nargs="?", default="binance-lgb-momtopk",
        help="配置名（不含 .yaml）",
    )
    args = parser.parse_args()

    config_path = f"config/{args.config}.yaml"
    if not Path(config_path).exists():
        available = [p.stem for p in Path("config").glob("*.yaml")]
        print(f"❌ 配置不存在: {config_path}")
        print(f"可用: {', '.join(available)}")
        sys.exit(1)

    print(f"🚀 训练: {args.config}")
    results = run_from_yaml(config_path)
    r = results["recorder"]

    print("\n📊 训练结果")
    for k, v in r.list_metrics().items():
        if "IC" in k:
            print(f"  {k}: {v:.4f}")
    for k, v in r.list_metrics().items():
        if "annualized" in k:
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
