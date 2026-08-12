---
name: oq-train-backtest
description: Train the RL rotation policy (tianshou PPO) and run a backtest, outputting NAV, Sharpe, drawdown, turnover, and IR metrics
argument-hint: "[config]"
---

# Train & Backtest

Train the MultiDiscrete PPO policy and backtest it on the test segment.

## Trigger conditions
- "train model" / "backtest" / "run experiment" / "train" / "evaluate strategy"

## Configs

| Config | Market | Universe | Note |
|------|--------|----------|------|
| `csi300-rl-rotation` | A-share (Tencent) | top-50 frozen 2012 | research only |
| `binance-rl-rotation` | Binance spot | top-20 | research + live |
| `hyperliquid-rl-rotation` | Hyperliquid spot | top-20 | research + live |
| `binance-h1-rl-rotation` | Binance spot h1 | top-20 | research + live |

## Running

```bash
cd orange-quant && source ../.venv/bin/activate

# 1. build the dataset (npz cache; --force to rebuild)
python -m orange_quant.rl.dataset csi300-rl-rotation

# 2. quick sanity run (1-5 epochs, ~1-7 min on CPU)
python -m orange_quant.rl.train csi300-rl-rotation --max-epoch 5

# 3. full training (50 epochs, ~8 min CPU); best valid checkpoint → models/
python -m orange_quant.rl.train csi300-rl-rotation

# 4. backtest (test-segment deterministic rollout)
python -m orange_quant.rl.backtest csi300-rl-rotation
# outputs/: nav.csv, metrics.json, nav.png (RL vs equal-weight vs benchmark)
```

## Interpretation

- `metrics.json` keys: annual_return, sharpe, max_drawdown, annual_turnover,
  information_ratio (vs equal-weight), benchmark_annual_return.
- Validation-segment mean reward per epoch is printed by the trainer; best
  checkpoint selection uses *pure* returns (no turnover penalty / baseline).
- First runs often underperform equal-weight — check turnover: >10x/yr usually
  means `turnover_penalty` in the yaml should be raised.
