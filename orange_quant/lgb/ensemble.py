"""Seed-bagged LightGBM ensemble wrapper (picklable, numpy interface).

Replaces the legacy qlib ``EnsembleLGB`` (which wrapped qlib model objects
that required ``predict(dataset, segment)``); this one takes plain feature
matrices and averages member Booster predictions.
"""

from __future__ import annotations

from typing import List

import numpy as np


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
