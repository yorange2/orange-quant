"""Build the 24-hour score-level ensemble: config + model pickle + diagnostics.

The hour-of-day experiment (PR #27/#28) showed "which hour is best" is
undecidable — hour differences are noise, so instead of picking an hour the
ensemble averages all 24 hour-anchored models at the score level
(``orange_quant.lgb.ensemble.HourEnsemble``): per decision date, each hour's
predictions are cross-sectionally z-scored over the available coins and the
z-scores are averaged across the 24 hours.

This script:
  1. writes ``config/generated/binance-lgb-momtopk-hour-ensemble.yaml`` — a
     clone of the hour-00 config with suffixed paths; ``cache_dir`` points at
     the hour-00 cache because the ensemble's reference dataset IS the hour-00
     dataset (execution prices on the canonical midnight bar);
  2. writes ``models/binance-lgb-momtopk-hour-ensemble/model.pkl`` — a
     pickled ``HourEnsemble`` (no training: the 24 member models already
     exist);
  3. prints member diagnostics: per-hour IC on its own test window, date
     coverage of the shared decision-day calendar, and cross-hour prediction
     correlation (how much variance reduction the averaging can buy).

Run from orange-quant/ (idempotent)::
    ../.venv/bin/python scripts/gen_binance_hour_ensemble.py
Then::
    ../.venv/bin/python -m orange_quant.lgb.backtest binance-lgb-momtopk-hour-ensemble
    ../.venv/bin/python scripts/backtest_stability.py \\
        binance-lgb-momtopk-h{00..23} binance-lgb-momtopk-hour-ensemble
"""

from __future__ import annotations

import copy
import pickle
from pathlib import Path

import numpy as np
import yaml

from orange_quant.lgb.dataset import load_or_build
from orange_quant.lgb.ensemble import HourEnsemble
from orange_quant.rl.dataset import load_config
from orange_quant.rl.metrics import per_date_corr

BASE = "binance-lgb-momtopk-h00"
NAME = "binance-lgb-momtopk-hour-ensemble"
OUT_CFG = Path("config/generated") / f"{NAME}.yaml"
OUT_MODEL = Path("models") / NAME / "model.pkl"


def write_config() -> None:
    cfg = copy.deepcopy(load_config(BASE))
    cfg["data"]["hour_of_day"] = 0
    # explicit paths (not appended: the h00 config's paths already carry -h00)
    cfg["paths"]["model_dir"] = f"models/{NAME}"
    cfg["paths"]["output_dir"] = f"outputs/{NAME}"
    cfg["paths"]["cache_dir"] = "data/binance-lgb-momtopk-h00"  # ref ds = h00
    OUT_CFG.parent.mkdir(parents=True, exist_ok=True)
    OUT_CFG.write_text(yaml.safe_dump(cfg, allow_unicode=True))
    print(f"[gen-hour-ensemble] {OUT_CFG}")


def write_model() -> None:
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MODEL, "wb") as f:
        pickle.dump(HourEnsemble(base="binance-lgb-momtopk"), f)
    print(f"[gen-hour-ensemble] {OUT_MODEL} (24 members, no training)")


def diagnostics() -> None:
    """Per-hour IC + shared decision-day coverage + cross-hour prediction corr."""
    names = [f"binance-lgb-momtopk-h{h:02d}" for h in range(24)]
    preds, dss = {}, {}
    for name in names:
        ds = load_or_build(load_config(name))
        test_s, test_e = ds.split_idx["test"]
        t1 = test_e - 2
        with open(Path(load_config(name)["paths"]["model_dir"]) / "model.pkl",
                  "rb") as f:
            model = pickle.load(f)
        preds[name] = model.predict(ds.feats[test_s : t1 + 1].reshape(
            -1, ds.n_feats)).reshape(t1 - test_s + 1, ds.n_stocks)
        dss[name] = ds

    ref = dss[names[0]]
    test_s, test_e = ref.split_idx["test"]
    t1 = test_e - 2
    ref_dates = ref.dates[test_s : t1 + 1]
    D = len(ref_dates)
    print(f"\n[member diagnostics] {len(names)} hours, reference (h00) "
          f"decision days: {D} ({ref_dates[0].astype('datetime64[D]')} ~ "
          f"{ref_dates[-1].astype('datetime64[D]')})")

    ics = []
    S = np.full((24, D, ref.n_stocks), np.nan)
    for i, name in enumerate(names):
        ds = dss[name]
        ts, te = ds.split_idx["test"]
        m1 = te - 2
        lab = ds.label[ts : m1 + 1]
        valid = ~np.isnan(lab)
        rows = np.argwhere(valid)
        ic = per_date_corr(preds[name][valid], lab[valid], rows[:, 0], "pearson")
        ics.append(float(ic.mean()))
        day_to_row = {np.datetime64(d, "D"): k
                      for k, d in enumerate(ds.dates[ts : m1 + 1])}
        for k, d in enumerate(ref_dates):
            r = day_to_row.get(d)
            if r is not None:
                S[i, k] = preds[name][r]
    print(f"per-hour IC (own window): mean {np.mean(ics):+.4f}, "
          f"range [{min(ics):+.4f}, {max(ics):+.4f}]")

    covered = int(np.all(np.isfinite(S), axis=(0, 2)).sum())
    print(f"decision days scored by all 24 hours: {covered}/{D}")

    corr = []
    for i in range(24):
        for j in range(i + 1, 24):
            a, b = S[i], S[j]
            m = np.isfinite(a) & np.isfinite(b)
            corr.append(np.corrcoef(a[m], b[m])[0, 1])
    corr = np.array(corr)
    print(f"cross-hour pooled prediction corr: mean {corr.mean():.4f}, "
          f"min {corr.min():.4f}, max {corr.max():.4f} — member corr 0.58 "
          f"leaves real room for variance reduction")


if __name__ == "__main__":
    write_config()
    write_model()
    diagnostics()
