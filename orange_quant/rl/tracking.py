"""Best-effort mlflow run logging (shared by the RL and LGB training entries).

Tracking must never block training: mlflow is imported lazily and every failure
is swallowed with a printed note.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Optional


def log_run(run_name: str, params: Dict, metrics: Dict[str, float],
            artifacts: Optional[Iterable[str]] = None, tag: str = "train") -> None:
    """Log one run to mlflow; never raises."""
    try:
        # mlflow 3.x blocks the filesystem backend by default
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import mlflow

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            for path in artifacts or ():
                mlflow.log_artifact(str(path))
        print(f"[{tag}] mlflow run logged: {run_name}")
    except Exception as e:  # noqa: BLE001 - tracking is best-effort
        print(f"[{tag}] mlflow logging skipped: {e}")
