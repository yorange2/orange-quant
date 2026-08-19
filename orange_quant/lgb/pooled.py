"""Pooled-across-hours model plumbing: hour features + per-anchor proxy.

The pooled design (``scripts/gen_binance_hour_pooled.py``) fits ONE booster
set on all 24 clock anchors stacked together, then scores each anchor with it.
Two pieces are needed to make that fit the existing backtest contract
(``model.predict(X)`` where X is one anchor's ``(D·N, 158)`` feature block):

  * :func:`hour_columns` — the 3 extra columns that carry the anchor identity
    in ``pooled-hf`` mode;
  * :class:`PooledHourModel` — a picklable proxy that lazy-loads the shared
    booster set and appends its own anchor's hour columns.

The proxy holds a path, not the boosters, so the 24 per-anchor pickles stay a
few hundred bytes each and every anchor provably scores with the identical
model.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

HOUR_FEATURE_COLS = ["HOUR_SIN", "HOUR_COS", "HOUR_RAW"]


def hour_columns(hour: int, n_rows: int) -> np.ndarray:
    """(n_rows, 3) block encoding the clock anchor.

    sin/cos give trees a smooth wrap-around handle (hour 23 and hour 0 are
    adjacent); the raw hour lets a single split isolate an arbitrary contiguous
    block of anchors, so the pooled-hf model can in principle reproduce the
    per-hour models exactly.
    """
    ang = 2.0 * np.pi * hour / 24.0
    col = np.empty((n_rows, 3), dtype=np.float32)
    col[:, 0] = np.sin(ang)
    col[:, 1] = np.cos(ang)
    col[:, 2] = float(hour)
    return col


class PooledHourModel:
    """Scores one clock anchor with the shared pooled booster set.

    ``predict(X)`` takes the anchor's raw ``(R, 158)`` Alpha158 block — exactly
    what ``orange_quant.lgb.backtest.predict_block`` passes — and, when the
    pooled model was fit with hour features, appends this anchor's hour columns
    before delegating to the shared :class:`~orange_quant.lgb.ensemble.EnsembleLGB`.
    """

    def __init__(self, model_path: str, hour: int,
                 with_hour_features: bool = False) -> None:
        self.model_path = str(model_path)
        self.hour = int(hour)
        self.with_hour_features = bool(with_hour_features)
        self._model = None

    def _load(self):
        if self._model is None:
            with open(Path(self.model_path), "rb") as f:
                self._model = pickle.load(f)
        return self._model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.with_hour_features:
            X = np.hstack([X, hour_columns(self.hour, len(X))])
        return self._load().predict(X)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_model"] = None          # never pickle the boosters into a proxy
        return state
