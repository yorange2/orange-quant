"""Seed-bagged LightGBM ensemble (exchange-agnostic).

A thin wrapper that averages the predictions of several LGBModels trained on the
same data with different random seeds. Seed-bagging reduces the variance of the
cross-sectional signal, which — unlike deeper trees or single-seed retuning —
robustly improves Rank IC across out-of-sample windows on the crypto universes.

The wrapper exposes the same ``predict(dataset, segment=...)`` interface as
qlib's ``LGBModel``, so it is a drop-in for both the experiment pipeline
(SignalRecord/PortAnaRecord) and the live ``ModelPredictor``. It must stay a
top-level, importable class so the pickled model round-trips in the live
container (``orange_quant.ensemble.EnsembleLGB``).
"""

from typing import List


class EnsembleLGB:
    """Average the test-segment predictions of several trained LGBModels."""

    def __init__(self, models: List[object]):
        if not models:
            raise ValueError("EnsembleLGB needs at least one model")
        self.models = models

    def predict(self, dataset, segment: str = "test"):
        preds = [m.predict(dataset, segment=segment) for m in self.models]
        out = preds[0].copy()
        for p in preds[1:]:
            out = out + p
        return out / len(preds)
