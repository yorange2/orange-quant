"""Experiment pipeline moved to orange_quant.experiment; re-exported for back-compat.

Keeps ``from biance_lgb_momtopk.workflow.experiment import run_from_yaml`` working
(used in .AGENT.md, notebooks, and the server's retrain path).
"""

from orange_quant.experiment import (
    QuantExperiment,
    run_from_yaml,
    run_dl_from_yaml,
    KNOWN_DL_MODULES,
)

__all__ = ["QuantExperiment", "run_from_yaml", "run_dl_from_yaml", "KNOWN_DL_MODULES"]
