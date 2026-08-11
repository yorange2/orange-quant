"""Generate the cross-section-width A/B configs (roadmap C2).

Reads the base ``config/cn-lgb-momtopk.yaml`` and writes
``config/generated/cn-lgb-momtopk-top{N}.yaml`` for N in (50, 300, 800, 2000)
— identical except ``universe.top_n`` and the model/output/cache paths, so the
A/B isolates universe width. The base config carries the shared
``min_amount`` liquidity floor.

Run from orange-quant/::
    ../.venv/bin/python scripts/gen_cn_ab_configs.py
Then per tier (universe scan + features build each time)::
    ../.venv/bin/python -m orange_quant.lgb.dataset cn-lgb-momtopk-top800
    ../.venv/bin/python -m orange_quant.lgb.train    cn-lgb-momtopk-top800
    ../.venv/bin/python -m orange_quant.lgb.backtest cn-lgb-momtopk-top800
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

TIERS = [50, 300, 800, 2000]
BASE = "config/cn-lgb-momtopk.yaml"
OUT = "config/generated"


def main() -> None:
    base = yaml.safe_load(Path(BASE).read_text())
    out = Path(OUT)
    out.mkdir(parents=True, exist_ok=True)
    for n in TIERS:
        cfg = copy.deepcopy(base)
        cfg["universe"]["top_n"] = n
        suffix = f"top{n}"
        for key in ("model_dir", "output_dir", "cache_dir"):
            cfg["paths"][key] = f"{cfg['paths'][key]}-{suffix}"
        name = f"{Path(BASE).stem}-{suffix}"
        (out / f"{name}.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True))
        print(f"[gen-cn-ab] {out}/{name}.yaml: top_n={n}")


if __name__ == "__main__":
    main()
