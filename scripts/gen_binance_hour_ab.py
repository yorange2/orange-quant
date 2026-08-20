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

import argparse
import copy
import json
from pathlib import Path

import yaml

BASE = "config/binance-lgb-momtopk.yaml"
OUT = "config/generated"


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the 24 hour-of-day configs")
    ap.add_argument("--exec-lag", type=int, default=1, choices=[0, 1],
                    help="label.exec_lag: 1 = legacy qlib timing, 0 = live timing")
    ap.add_argument("--tag", default="",
                    help="name/path infix separating this family from the "
                         "default one (e.g. 'lag0'); empty overwrites it")
    ap.add_argument("--feature-lag", type=int, default=0,
                    help="features.lag: bars the feature block is pushed back")
    ap.add_argument("--codes-from", default=None,
                    help="meta.json of an existing cache — pin its universe "
                         "into universe.codes so an A/B against that cache "
                         "differs only in what is being tested")
    args = ap.parse_args()

    base = yaml.safe_load(Path(BASE).read_text())
    pinned = None
    if args.codes_from:
        pinned = json.loads(Path(args.codes_from).read_text())["codes"]
        print(f"[gen-binance-hour] pinning {len(pinned)} codes from "
              f"{args.codes_from}")
    out = Path(OUT)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"-{args.tag}" if args.tag else ""
    for h in range(24):
        cfg = copy.deepcopy(base)
        cfg["data"]["raw_dir"] = "data/binance_h1_raw"
        cfg["data"]["hour_of_day"] = h
        cfg["universe"]["raw_dir"] = "data/binance_h1_raw"
        cfg.setdefault("label", {})["exec_lag"] = args.exec_lag
        cfg.setdefault("features", {})["lag"] = args.feature_lag
        if pinned is not None:
            cfg["universe"]["codes"] = pinned
        suffix = f"h{h:02d}"
        for key in ("model_dir", "output_dir", "cache_dir"):
            cfg["paths"][key] = f"{cfg['paths'][key]}{tag}-{suffix}"
        name = f"{Path(BASE).stem}{tag}-{suffix}"
        (out / f"{name}.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True))
        print(f"[gen-binance-hour] {out}/{name}.yaml: hour={h} (UTC), "
              f"exec_lag={args.exec_lag}, feature_lag={args.feature_lag}")


if __name__ == "__main__":
    main()
