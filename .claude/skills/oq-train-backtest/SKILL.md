---
name: oq-train-backtest
description: Train a LightGBM model and run a backtest, outputting IC, returns, Sharpe, and other metrics
argument-hint: "[config]"
---

# Train & Backtest

Train a LightGBM model and run a backtest, outputting IC, returns, Sharpe, and other metrics.

## Trigger conditions
- "train model" / "backtest" / "run experiment" / "train" / "backtest"
- "evaluate strategy" / "check performance"

## Experiment configs

| Config | Universe | IC | Notes |
|------|------|-----|------|
| `csi300-lgb-momtopk` | A-share CSI300 (820) | 0.027 | Alpha158 + LightGBM |
| `binance-lgb-momtopk` | Binance top-20 blue chips | ~0.02 | Binance spot costs |

## Running

### Single experiment

```bash
source .venv/bin/activate
python -m biance_lgb_momtopk.train binance-lgb-momtopk
python -m biance_lgb_momtopk.train csi300-lgb-momtopk
```

### Deep learning experiment (LSTM/GRU/Transformer)

```bash
source .venv/bin/activate
python -c "
from biance_lgb_momtopk.workflow.experiment import run_dl_from_yaml
results = run_dl_from_yaml('config/csi300-lstm-momtopk.yaml')
"
```

> Note: DL training needs a GPU; CPU training is extremely slow.

## Output metrics

- **IC** (>0.05 effective factor, >0.1 excellent)
- **ICIR** (>0.5 stable, >1.0 excellent)
- **Rank IC** (rank correlation, more robust)
- **Annualized excess return** (with/without trading costs)
- **Information Ratio** (>1.0 excellent)
- **Max drawdown**

## Model export

After training, the model is **automatically exported** to the `models/` directory:

```
binance-lgb-momtopk  →  models/binance-lgb-momtopk.pkl
csi300-lgb-momtopk   →  models/csi300-lgb-momtopk.pkl
csi300-lstm-momtopk  →  models/csi300-lstm-momtopk.pkl
```

Live trading simply loads the corresponding pkl file.
