"""Generate the 24 hour-of-day A/B configs (one dataset per UTC hour).

Reads the base ``config/binance-lgb-momtopk.yaml`` and writes
``config/generated/binance-lgb-momtopk-h{HH}.yaml`` for H in 0..23 — identical
except:

  * ``data.raw_dir`` → ``data/binance_h1_raw`` (hourly bars);
  * ``data.hour_of_day: H`` — the bar reader keeps only that UTC hour's bars,
    so each config sees the market as a fixed-clock daily series (the label
    becomes the same-hour next-day return; rolling features span days at that
    hour);
  * model/output/cache paths get a ``-h{HH}`` suffix.

The universe freeze (top-50 liquidity pool, 2025-01-01 → 2026-08-07) is
identical across hours, so the 24 runs differ only in the clock hour.

Run from orange-quant/::
    ../.venv/bin/python scripts/gen_binance_hour_ab.py
Then per hour::
    ../.venv/bin/python -m orange_quant.lgb.dataset  binance-lgb-momtopk-h08
    ../.venv/bin/python -m orange_quant.lgb.train     binance-lgb-momtopk-h08
    ../.venv/bin/python -m orange_quant.lgb.backtest  binance-lgb-momtopk-h08
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

BASE = "config/binance-lgb-momtopk.yaml"
OUT = "config/generated"


def main() -> None:
    base = yaml.safe_load(Path(BASE).read_text())
    out = Path(OUT)
    out.mkdir(parents=True, exist_ok=True)
    for h in range(24):
        cfg = copy.deepcopy(base)
        cfg["data"]["raw_dir"] = "data/binance_h1_raw"
        cfg["data"]["hour_of_day"] = h
        cfg["universe"]["raw_dir"] = "data/binance_h1_raw"
        suffix = f"h{h:02d}"
        for key in ("model_dir", "output_dir", "cache_dir"):
            cfg["paths"][key] = f"{cfg['paths'][key]}-{suffix}"
        name = f"{Path(BASE).stem}-{suffix}"
        (out / f"{name}.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True))
        print(f"[gen-binance-hour] {out}/{name}.yaml: hour={h} (UTC)")


if __name__ == "__main__":
    main()
