"""Seed-bagged LightGBM ensemble wrapper (picklable, numpy interface).

Replaces the legacy qlib ``EnsembleLGB`` (which wrapped qlib model objects
that required ``predict(dataset, segment)``); this one takes plain feature
matrices and averages member Booster predictions.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import numpy as np

from orange_quant.lgb.dataset import load_or_build
from orange_quant.rl.dataset import load_config


class EnsembleLGB:
    """Mean of several LightGBM Boosters trained with different seeds."""

    def __init__(self, models: List) -> None:
        if not models:
            raise ValueError("EnsembleLGB requires at least one model")
        self.models = models

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Average member predictions; X is (R, n_features) float array."""
        preds = [m.predict(X) for m in self.models]
        out = preds[0].astype(np.float64)
        for p in preds[1:]:
            out = out + p
        return out / len(preds)


class HourEnsemble:
    """Score-level ensemble over the 24 hour-anchored daily-series models.

    Each hour model sees a *different* fixed-clock daily series (same calendar
    day, different anchor hour), so the members share no feature matrix and
    cannot be averaged at the model level — aggregation happens on the scores.
    The hour-of-day experiment showed "which hour is best" is undecidable
    (sub-window hour rankings ≈ 0, bootstrap CI excludes 0 for 0/24 hours),
    so instead of picking an hour we average all 24: per decision date, each
    member's predictions are cross-sectionally z-scored over the available
    coins, then averaged over the hours that have a bar that date (sum and
    mean rank identically while all 24 members are present, which they are on
    every decision date — the 24 calendars differ only in warmup days and one
    final mark date).

    ``predict(X)`` satisfies the standard backtest/stability contracts: X is
    the reference (hour-00) dataset's test features and is ignored — the
    member scores are recomputed from each hour's own cached dataset + model
    and aligned on the shared decision-day calendar. Hour-00 is the reference
    because its calendar is the longest and its closes are the canonical
    midnight daily bar used for execution.

    This is deliberately research-grade: one predict() reloads 24 datasets and
    models, fine for a weekly A/B but not for per-day live trading.
    """

    def __init__(self, base: str = "binance-lgb-momtopk",
                 hours: List[int] | None = None) -> None:
        self.base = base
        self.hours = list(range(24)) if hours is None else list(hours)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # X is ignored — member scores come from each hour's own dataset.
        ref = load_or_build(load_config(f"{self.base}-h00"))
        test_s, test_e = ref.split_idx["test"]
        t1 = test_e - 2                                   # last decision day
        D = t1 - test_s + 1
        if X.shape[0] != D * ref.n_stocks:
            raise ValueError(f"HourEnsemble: X rows {X.shape[0]} != "
                             f"{D} decision days × {ref.n_stocks} coins")
        ref_dates = ref.dates[test_s : t1 + 1]

        zsum = np.zeros((D, ref.n_stocks))
        cnt = np.zeros((D, ref.n_stocks))
        for h in self.hours:
            name = f"{self.base}-h{h:02d}"
            cfg = load_config(name)
            ds = load_or_build(cfg)
            with open(Path(cfg["paths"]["model_dir"]) / "model.pkl", "rb") as f:
                model = pickle.load(f)
            ts, te = ds.split_idx["test"]
            m1 = te - 2
            p = model.predict(ds.feats[ts : m1 + 1].reshape(-1, ds.n_feats)
                              ).reshape(m1 - ts + 1, ds.n_stocks)
            # align on the shared decision-day calendar (dates are identical
            # across hours within the test window; index offsets differ)
            day_to_row = {np.datetime64(d, "D"): k
                          for k, d in enumerate(ds.dates[ts : m1 + 1])}
            rows = np.array([day_to_row.get(d, -1) for d in ref_dates])
            z = np.full((D, ref.n_stocks), np.nan)
            for k in np.flatnonzero(rows >= 0):
                row = p[rows[k]]
                m = np.isfinite(row)
                if m.sum() > 2:
                    std = row[m].std(ddof=1)
                    if std > 0:
                        z[k, m] = (row[m] - row[m].mean()) / std
            have = np.isfinite(z)
            zsum += np.where(have, z, 0.0)
            cnt += have.astype(np.float64)

        out = zsum / np.where(cnt > 0, cnt, 1.0)
        out[cnt == 0] = np.nan
        return out
