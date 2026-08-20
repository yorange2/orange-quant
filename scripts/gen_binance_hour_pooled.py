"""Pooled-across-hours LGB models: the missing arm of the hour-of-day study.

The existing hour experiments cover two designs:

  * **per-hour models** (``binance-lgb-momtopk-h00..h23``) — one model per
    clock anchor, each trained on that anchor's ~46k rows;
  * **score-level ensemble** (``binance-lgb-momtopk-hour-ensemble``) — the
    24 per-hour models averaged as cross-sectional z-scores.

Both treat the 24 anchors as 24 datasets. This script adds the third design:
pool all 24 anchors into ONE training set (~1.1M rows) and fit a single
model — the "data augmentation" reading of the same 24 series.

Two modes:
  * ``pooled``    — anchors pooled, hour identity discarded;
  * ``pooled-hf`` — anchors pooled plus 3 hour features (sin, cos, raw hour),
    so trees can recover per-hour behaviour where it exists. This nests both
    extremes: zero splits on the hour columns → ``pooled``; splitting on hour
    at every node → the per-hour models.

Purge/embargo: the label spans close[t+1]→close[t+2], so a train row at the
segment boundary peeks 2 days into the next segment. ``--embargo`` (default 2)
drops that many trailing days from train and valid. Tiny here (2/2008 days)
but it makes the pooled fit honest — 24× overlapping rows are exactly the
setting where boundary leakage compounds.

The fitted booster set is written ONCE to
``models/binance-lgb-momtopk-{mode}/model.pkl``; the 24 generated configs get
a thin ``PooledHourModel`` proxy that lazy-loads it and (in ``pooled-hf``)
appends that anchor's hour columns at predict time. So each anchor is scored
by the same pooled model but executed on its own bars — which is what makes
the per-anchor Sharpe spread comparable across all four designs.

Run from orange-quant/ (idempotent)::

    ../.venv/bin/python scripts/gen_binance_hour_pooled.py --mode pooled
    ../.venv/bin/python scripts/gen_binance_hour_pooled.py --mode pooled-hf
    for h in $(seq -w 0 23); do
        ../.venv/bin/python -m orange_quant.lgb.backtest \
            binance-lgb-momtopk-pooled-h$h
    done
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import yaml

from orange_quant.lgb.dataset import load_or_build
from orange_quant.lgb.ensemble import EnsembleLGB
from orange_quant.lgb.pooled import HOUR_FEATURE_COLS, PooledHourModel, hour_columns
from orange_quant.rl.dataset import load_config
from orange_quant.rl.metrics import per_date_corr

BASE = "binance-lgb-momtopk"
HOURS = list(range(24))


# --------------------------------------------------------------------------
# pooled row assembly
# --------------------------------------------------------------------------
def segment_rows(ds, segment: str, embargo: int):
    """(X, y, date_idx) for one anchor's segment, trailing ``embargo`` days cut.

    The cut is applied to the END of the segment only: label[t] reads
    close[t+2], so the last ``embargo`` days are the ones whose labels reach
    into the next segment.
    """
    s, e = ds.split_idx[segment]
    e = e - embargo
    if e <= s:
        raise ValueError(f"embargo {embargo} wipes out segment {segment}")
    label = ds.label[s : e + 1]
    valid = ~np.isnan(label)
    rows = np.argwhere(valid)
    return (ds.feats[s : e + 1][valid],
            label[valid].astype(np.float32),
            (rows[:, 0] + s).astype(np.int64))


def build_pooled(mode: str, embargo: int, base: str = BASE):
    """Stack the 24 anchors' train/valid rows into one design matrix."""
    parts = {"train": [], "valid": []}
    for h in HOURS:
        ds = load_or_build(load_config(f"{base}-h{h:02d}"))
        for seg in parts:
            X, y, date_idx = segment_rows(ds, seg, embargo)
            if mode == "pooled-hf":
                X = np.hstack([X, hour_columns(h, len(X))])
            # date_idx is per-anchor; offset by hour so per-date IC groups
            # stay distinct across anchors (24 anchors × same calendar day are
            # 24 separate cross-sections, not one)
            parts[seg].append((X, y, date_idx * 24 + h))
        print(f"[pooled] h{h:02d}: train={len(parts['train'][-1][1])} "
              f"valid={len(parts['valid'][-1][1])}")

    out = {}
    for seg, chunks in parts.items():
        out[seg] = (np.vstack([c[0] for c in chunks]),
                    np.concatenate([c[1] for c in chunks]),
                    np.concatenate([c[2] for c in chunks]))
        print(f"[pooled] {seg}: {out[seg][0].shape} "
              f"({out[seg][0].nbytes / 1e9:.2f} GB)")
    return out


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def train_pooled(cfg: dict, pooled: dict, mode: str, rounds: int | None):
    lgb_cfg = cfg["lgb"]
    n_seeds = int(cfg.get("ensemble", {}).get("n_seeds", 1) or 1)
    rounds = int(rounds or lgb_cfg["num_boost_round"])
    base_seed = int(lgb_cfg.get("seed", 42))

    X_tr, y_tr, _ = pooled["train"]
    X_va, y_va, va_date = pooled["valid"]

    params = {
        "objective": lgb_cfg.get("loss", "mse"),
        "learning_rate": float(lgb_cfg["learning_rate"]),
        "num_leaves": int(lgb_cfg["num_leaves"]),
        "feature_fraction": float(lgb_cfg["feature_fraction"]),
        "bagging_fraction": float(lgb_cfg["bagging_fraction"]),
        "bagging_freq": int(lgb_cfg["bagging_freq"]),
        "min_data_in_leaf": int(lgb_cfg["min_data_in_leaf"]),
        "lambda_l1": float(lgb_cfg["lambda_l1"]),
        "lambda_l2": float(lgb_cfg["lambda_l2"]),
        "verbosity": -1,
        "num_threads": os.cpu_count() or 4,
    }
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain)

    boosters, best_iters, rmses = [], [], []
    for i in range(n_seeds):
        t0 = time.time()
        bst = lgb.train(
            dict(params, seed=base_seed + i), dtrain, num_boost_round=rounds,
            valid_sets=[dvalid], valid_names=["valid"],
            callbacks=[lgb.early_stopping(int(lgb_cfg["early_stopping_rounds"]),
                                          verbose=False),
                       lgb.log_evaluation(100)],
        )
        boosters.append(bst)
        best_iters.append(bst.best_iteration)
        # objective=mse reports under "l2" (train.py falls back the same way)
        score = bst.best_score["valid"]
        rmses.append(float(score.get("rmse", score.get("l2", float("nan")))))
        print(f"[pooled-train] seed {i}: best_iter={bst.best_iteration} "
              f"rmse={rmses[-1]:.5f} ({time.time() - t0:.0f}s)")

    model = EnsembleLGB(boosters)
    pred = model.predict(X_va)
    ic = per_date_corr(pred, y_va, va_date, "pearson")
    ric = per_date_corr(pred, y_va, va_date, "spearman")
    metrics = {
        "mode": mode,
        "valid_ic": float(ic.mean()),
        "valid_rank_ic": float(ric.mean()),
        "valid_rmse": float(np.mean(rmses)),
        "best_iteration_per_seed": best_iters,
        "n_train_rows": int(len(y_tr)),
        "n_seeds": n_seeds,
    }
    print(f"[pooled-train] valid IC={metrics['valid_ic']:.4f} "
          f"RankIC={metrics['valid_rank_ic']:.4f} RMSE={metrics['valid_rmse']:.5f}")

    if mode == "pooled-hf":
        gain = boosters[0].feature_importance("gain")
        n_base = len(gain) - len(HOUR_FEATURE_COLS)
        hour_gain = gain[n_base:].sum() / max(gain.sum(), 1e-12)
        metrics["hour_feature_gain_share"] = float(hour_gain)
        print(f"[pooled-train] hour features carry {hour_gain:.2%} of split gain "
              f"— the direct read on whether the anchor matters at all")
    return model, metrics


# --------------------------------------------------------------------------
# artifacts
# --------------------------------------------------------------------------
def write_configs(mode: str, shared_path: Path, base: str = BASE) -> None:
    """One config per anchor: pooled model, that anchor's bars for execution."""
    for h in HOURS:
        name = f"{base}-{mode}-h{h:02d}"
        cfg = copy.deepcopy(load_config(f"{base}-h{h:02d}"))
        cfg["paths"]["cache_dir"] = f"data/{base}-h{h:02d}"   # anchor's own bars
        cfg["paths"]["model_dir"] = f"models/{name}"
        cfg["paths"]["output_dir"] = f"outputs/{name}"
        out = Path("config/generated") / f"{name}.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(cfg, allow_unicode=True))

        model_dir = Path("models") / name
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "model.pkl", "wb") as f:
            pickle.dump(PooledHourModel(str(shared_path), h,
                                        with_hour_features=(mode == "pooled-hf")), f)
    print(f"[pooled] wrote 24 configs + proxy models for mode={mode}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the pooled-across-hours LGB")
    ap.add_argument("--mode", choices=["pooled", "pooled-hf"], default="pooled")
    ap.add_argument("--embargo", type=int, default=2,
                    help="trailing days dropped from train/valid (label horizon)")
    ap.add_argument("--num-boost-round", type=int, default=None)
    ap.add_argument("--base", default=BASE,
                    help="config family to pool, e.g. binance-lgb-momtopk-lag0")
    args = ap.parse_args()

    cfg = load_config(f"{args.base}-h00")
    pooled = build_pooled(args.mode, args.embargo, args.base)
    model, metrics = train_pooled(cfg, pooled, args.mode, args.num_boost_round)

    shared_dir = Path("models") / f"{args.base}-{args.mode}"
    shared_dir.mkdir(parents=True, exist_ok=True)
    shared_path = shared_dir / "model.pkl"
    with open(shared_path, "wb") as f:
        pickle.dump(model, f)
    metrics["embargo_days"] = args.embargo
    (shared_dir / "best_metric.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[pooled] saved {shared_path}")

    write_configs(args.mode, shared_path.resolve(), args.base)


if __name__ == "__main__":
    main()
