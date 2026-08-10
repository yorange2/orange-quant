"""Train the LightGBM model (seed-bagged ensemble) and report valid metrics.

Faithful port of the legacy qlib pipeline: MSE on the cross-sectionally
z-scored 1-day-forward return label, raw Alpha158 features (NaN native to
LightGBM), early stopping on valid RMSE, average of ``n_seeds`` Boosters
trained with different seeds.

Usage:
    python -m orange_quant.lgb.train <config> [--num-boost-round N] [--no-mlflow]
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import lightgbm as lgb

from orange_quant.lgb.dataset import LGBDataset, load_or_build, load_config
from orange_quant.lgb.ensemble import EnsembleLGB
from orange_quant.lgb.features import FEATURE_COLS
from orange_quant.rl.metrics import per_date_corr
from orange_quant.rl.tracking import log_run


def _segment_rows(ds: LGBDataset, segment: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(X (R,158) float32, y (R,) float32, date_idx (R,), code_idx (R,)) for a segment."""
    s, e = ds.split_idx[segment]
    feats = ds.feats[s : e + 1]                     # (D, N, F)
    label = ds.label[s : e + 1]                     # (D, N)
    valid = ~np.isnan(label)
    rows = np.argwhere(valid)
    X = feats[valid]
    y = label[valid].astype(np.float32)
    date_idx = (rows[:, 0] + s).astype(np.int64)
    code_idx = rows[:, 1].astype(np.int64)
    return X, y, date_idx, code_idx


def train_model(config: dict, ds: LGBDataset, num_boost_round: int | None = None,
                quiet: bool = False) -> Tuple[EnsembleLGB, Dict]:
    """Fit the seed-bagged ensemble; returns (model, metrics)."""
    lgb_cfg = config["lgb"]
    n_seeds = int(config.get("ensemble", {}).get("n_seeds", 1) or 1)
    rounds = int(num_boost_round or lgb_cfg["num_boost_round"])
    es_rounds = int(lgb_cfg["early_stopping_rounds"])
    base_seed = int(lgb_cfg.get("seed", 42))

    X_tr, y_tr, _, _ = _segment_rows(ds, "train")
    X_va, y_va, va_date, _ = _segment_rows(ds, "valid")
    print(f"[lgb-train] rows: train={len(y_tr)}, valid={len(y_va)}")
    if len(y_tr) < 1000 or len(y_va) < 100:
        raise RuntimeError(f"too few training rows ({len(y_tr)}/{len(y_va)})")

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

    boosters, best_iters, valid_rmses = [], [], []
    for i in range(n_seeds):
        p = dict(params, seed=base_seed + i)
        callbacks = [lgb.early_stopping(es_rounds, verbose=not quiet)]
        if not quiet:
            callbacks.append(lgb.log_evaluation(50))
        bst = lgb.train(
            p, dtrain, num_boost_round=rounds,
            valid_sets=[dvalid], valid_names=["valid"],
            callbacks=callbacks,
        )
        boosters.append(bst)
        best_iters.append(bst.best_iteration)
        score = bst.best_score["valid"]
        valid_rmses.append(float(score.get("rmse", score.get("l2", float("nan")))))

    model = EnsembleLGB(boosters)

    # valid IC / Rank IC: per-date correlation vs the label, mean over dates
    pred_va = model.predict(X_va)
    ic = per_date_corr(pred_va, y_va, va_date, "pearson")
    ric = per_date_corr(pred_va, y_va, va_date, "spearman")
    metrics = {
        "valid_rmse": float(np.mean(valid_rmses)),
        "valid_rmse_per_seed": valid_rmses,
        "best_iteration_per_seed": best_iters,
        "valid_ic": float(ic.mean()) if len(ic) else float("nan"),
        "valid_rank_ic": float(ric.mean()) if len(ric) else float("nan"),
        "n_train_rows": int(len(y_tr)),
        "n_seeds": n_seeds,
    }
    print(f"[lgb-train] valid RMSE={metrics['valid_rmse']:.5f} "
          f"IC={metrics['valid_ic']:.4f} RankIC={metrics['valid_rank_ic']:.4f} "
          f"(best iters {best_iters})")
    return model, metrics


def save_model(config: dict, model: EnsembleLGB, metrics: Dict, codes=None) -> None:
    model_dir = Path(config["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    (model_dir / "best_metric.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2))
    (model_dir / "meta.json").write_text(json.dumps({
        "feature_names": FEATURE_COLS,
        "codes": codes,
        "lgb_params": config["lgb"],
        "ensemble": config.get("ensemble", {}),
        "windows": {
            "train": config["train"], "valid": config["valid"],
            "test": config["test"],
        },
    }, ensure_ascii=False, indent=2))
    print(f"[lgb-train] saved {model_dir}/model.pkl + best_metric.json")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train binance LGB momtopk")
    parser.add_argument("config", help="config name without .yaml")
    parser.add_argument("--num-boost-round", type=int, default=None,
                        help="override num_boost_round (sanity runs)")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = load_or_build(cfg)
    model, metrics = train_model(cfg, ds, num_boost_round=args.num_boost_round)
    save_model(cfg, model, metrics, codes=ds.codes)

    if not args.no_mlflow:
        log_run(
            f"{cfg['market']['venue']}-lgb-momtopk",
            params={**cfg["lgb"], "n_seeds": metrics["n_seeds"],
                    "universe_top_n": cfg["universe"]["top_n"],
                    "n_features": ds.n_feats},
            metrics={k: metrics[k] for k in
                     ("valid_rmse", "valid_ic", "valid_rank_ic")},
            artifacts=[Path(cfg["paths"]["model_dir"]) / "model.pkl"],
            tag="lgb-train",
        )


if __name__ == "__main__":
    main()
