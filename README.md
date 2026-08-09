# 🍊 Orange Quant

Reinforcement-learning quantitative trading framework — data straight from the
feeds (no qlib), training exclusively on tianshou.

## Architecture

```
orange_quant/
├── data/
│   ├── tencent.py          # A-share daily bars from the Tencent K-line API
│   │                       #   (end-anchored pagination → data/cn_raw/*.csv)
│   ├── universe.py         # liquidity-frozen universe (cn + crypto, zero extra API)
│   ├── sources.py          # Binance (REST) / Hyperliquid (ccxt) fetch hooks
│   └── pipeline.py         # incremental crypto download loop (CSV resumable)
├── rl/                     # market-agnostic RL core (the research engine)
│   ├── dataset.py          # bars → 10 OHLCV factors → npz cache (z-score fit on train)
│   ├── env.py              # RotationEnv: obs = features + current tiers,
│   │                       #   action = MultiDiscrete tiers, next-day-open execution
│   ├── network.py/policy.py# MultiDiscrete PPO (tianshou 0.4.10)
│   ├── train.py            # onpolicy_trainer + valid-segment best checkpoint
│   ├── backtest.py         # test-segment rollout, NAV/turnover metrics, 3-line chart
│   └── metrics.py / smoke_test.py
├── trading/
│   ├── broker.py           # Broker ABC (10 methods)
│   ├── binance_broker.py   # ccxt spot, reduce-only blacklist
│   ├── hyperliquid_broker.py  # ccxt, wallet auth
│   └── paper_broker.py     # simulated account
├── live.py                 # live runner: bars → features → policy → tiers → orders
├── server.py               # daily cron loop (--config/--once/--dry-run, heartbeat)
├── blacklist.py / healthcheck.py
└── config/                 # one yaml per market
    ├── csi300-rl-rotation.yaml       # A-share research
    ├── binance-rl-rotation.yaml      # Binance live + research
    └── hyperliquid-rl-rotation.yaml  # Hyperliquid live + research
```

Market differences live only in the data layer (source/universe/benchmark) and
the execution layer (broker); the RL train/backtest/env core is identical across
markets and is parameterized entirely by the yaml.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                      # pyproject deps: tianshou/gym/torch/ccxt/...
```

### A-share research

```bash
# 1. fetch daily bars (first column of a TSV/CSV = SH/SZ symbols)
python -m orange_quant.data.tencent --symbols-file <symbols.tsv>   # ~4800 names
# 2. build the dataset (freezes top-50 liquid names, caches npz)
python -m orange_quant.rl.dataset csi300-rl-rotation
# 3. train (CPU, ~8 min for 50 epochs) and backtest
python -m orange_quant.rl.train csi300-rl-rotation
python -m orange_quant.rl.backtest csi300-rl-rotation
```

### Crypto live

```bash
python -m orange_quant.data.build --exchange binance --top 20      # daily bars
python -m orange_quant.rl.dataset binance-rl-rotation
python -m orange_quant.rl.train binance-rl-rotation
python -m orange_quant.rl.backtest binance-rl-rotation

# paper trade once, then live (set keys in .env)
python -m orange_quant.server --config binance-rl-rotation --once --dry-run
python -m orange_quant.server --config binance-rl-rotation --once
```

## Experiment results

| Market | Universe | Period | Annual ret (net) | Sharpe | vs benchmark |
|--------|----------|--------|------------------|--------|--------------|
| A-share RL rotation | top-50 liquid (frozen 2012) | test 2023–2026 | −5.0% | −0.31 | equal-weight +3.0%/yr |
| Binance RL rotation | top-20 (frozen 2026) | test 2025–2026 | +70.9% | 0.55 | BTC +50.2%/yr (3-epoch smoke) |

Data-quality note: the new Tencent-only dataset fixes the legacy store's
2022-01~2022-05 data hole (4.5 months of zeros) and uses hfq (correct
ex-dividend returns); validation-segment mean reward improved from +0.87%/day
(legacy data) to +1.57%/day. Strategy alpha vs equal-weight is still an open
research question (see ROADMAP).

## Live retraining (walk-forward schedule)

```bash
# quarterly, from cron (e.g. 1st of Jan/Apr/Jul/Oct 03:00):
# 0 3 1 1,4,7,10 * cd /path/to/orange-quant && .venv/bin/python -m scripts.retrain_live --config binance-rl-rotation
```
Retrains on the trailing 3 years (valid = last 6 months), refreshes the raw
bars/npz, and atomically swaps `models/<cfg>/policy_best.pth` — the live
server loads the checkpoint on every daily run, so the new model takes effect
without a restart. History in `models/<cfg>/retrain_history.json`.

## Notes

- Data is backward-adjusted (hfq); never use qfq for investment series (re-anchored
  by each dividend = look-ahead).
- The Tencent API returns a trailing 640-row window anchored at the request's end
  date — fetching must use end-anchored pagination (the legacy 3-year loop had a
  data hole in 2021).
- Live trading is idempotent per day via the state file; `--force` reruns.
- mlflow file store requires `MLFLOW_ALLOW_FILE_STORE=true` (set in train.py and
  the Docker image).
