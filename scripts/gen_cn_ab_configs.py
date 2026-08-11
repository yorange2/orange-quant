"""Generate the cross-section A/B configs (roadmap C2/C4).

Reads the base ``config/cn-lgb-momtopk.yaml`` and writes:

  * ``config/generated/cn-lgb-momtopk-top{N}.yaml`` for N in (50, 300, 800,
    2000) — identical except ``universe.top_n`` and the model/output/cache
    paths, so the width A/B isolates universe width (C2);
  * ``config/generated/cn-lgb-momtopk-cs{rank,zscore}.yaml`` — identical to
    the base (top-300) except ``features.cs_norm``, the feature
    cross-sectional standardization A/B vs the C1 baseline (C4).

The base config carries the shared ``min_amount`` liquidity floor.

Run from orange-quant/::
    ../.venv/bin/python scripts/gen_cn_ab_configs.py
Then per config (universe scan + features build each time)::
    ../.venv/bin/python -m orange_quant.lgb.dataset cn-lgb-momtopk-top800
    ../.venv/bin/python -m orange_quant.lgb.train    cn-lgb-momtopk-top800
    ../.venv/bin/python -m orange_quant.lgb.backtest cn-lgb-momtopk-top800
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

TIERS = [50, 300, 800, 2000]
CS_VARIANTS = ["rank", "zscore"]
BASE = "config/cn-lgb-momtopk.yaml"
OUT = "config/generated"


def _write(cfg: dict, name: str, out: Path) -> None:
    (out / f"{name}.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True))
    print(f"[gen-cn-ab] {out}/{name}.yaml")


def main() -> None:
    base = yaml.safe_load(Path(BASE).read_text())
    stem = Path(BASE).stem
    out = Path(OUT)
    out.mkdir(parents=True, exist_ok=True)
    for n in TIERS:
        cfg = copy.deepcopy(base)
        cfg["universe"]["top_n"] = n
        suffix = f"top{n}"
        for key in ("model_dir", "output_dir", "cache_dir"):
            cfg["paths"][key] = f"{cfg['paths'][key]}-{suffix}"
        _write(cfg, f"{stem}-{suffix}", out)
    for mode in CS_VARIANTS:
        cfg = copy.deepcopy(base)
        cfg["features"]["cs_norm"] = mode
        suffix = f"cs{mode}"
        for key in ("model_dir", "output_dir", "cache_dir"):
            cfg["paths"][key] = f"{cfg['paths'][key]}-{suffix}"
        _write(cfg, f"{stem}-{suffix}", out)


if __name__ == "__main__":
    main()
