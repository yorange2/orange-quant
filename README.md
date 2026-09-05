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
│   ├── industry.py         # SW level-1 industry map (data/cn_industry.csv)
│   ├── sources.py          # Binance (REST) / Hyperliquid (ccxt) fetch hooks
│   └── pipeline.py         # incremental crypto download loop (CSV resumable)
├── lgb/                    # market-agnostic LightGBM core (cross-sectional alpha)
│   ├── dataset.py          # bars → Alpha158 (158 feats) → npz cache; cs_norm,
│   │                       #   industry-neutral label, hour_of_day knobs
│   ├── features.py         # qlib Alpha158 ported to pandas (0 mismatches vs qlib)
│   ├── train.py            # seed-bagged ensemble; mse | lambdarank objectives
│   ├── backtest.py         # TopkDropout portfolio, suspension-safe (ffill valuation)
│   └── report.py           # cross-sectional report: IC/ICIR, deciles, long-short
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
    ├── csi300-rl-rotation.yaml       # A-share research (RL)
    ├── binance-rl-rotation.yaml      # Binance live + research (RL)
    ├── hyperliquid-rl-rotation.yaml  # Hyperliquid live + research (RL)
    ├── binance-lgb-momtopk.yaml      # Binance live + research (LGB)
    └── cn-lgb-momtopk.yaml           # A-share research (LGB, top-300 liquidity pool)
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

### Binance LGB momentum top-k (Alpha158, seed-bagged LightGBM)

The legacy qlib LGB pipeline, rebuilt on this architecture (`orange_quant.lgb`,
config `binance-lgb-momtopk.yaml`, `strategy.type: lgb` — `server.py` dispatches
the runner on it):

```bash
../.venv/bin/python -m orange_quant.lgb.dataset  binance-lgb-momtopk
../.venv/bin/python -m orange_quant.lgb.train     binance-lgb-momtopk
../.venv/bin/python -m orange_quant.lgb.backtest  binance-lgb-momtopk
../.venv/bin/python -m orange_quant.server --config binance-lgb-momtopk --once --dry-run
```

Features are the full 158-feature qlib Alpha158 set, ported to pandas
(`orange_quant.lgb.features`, cross-checked against qlib's expression engine:
0 mismatches on 23 representative ops). Backtest timing matches qlib
(`signal day t → rebalance at close[t+1] → earn [t+1,t+2]`); live does full
daily rotation to top-k on the signal day (legacy deviation).

### A-share LGB momentum top-k (research)

The legacy qlib `csi300-lgb-momtopk` pipeline ported to the new architecture
(top-300 liquidity pool frozen 2017-12-31 instead of hardcoded constituents,
benchmark SH000300 from the local Tencent CSVs):

```bash
../.venv/bin/python -m orange_quant.lgb.dataset  cn-lgb-momtopk
../.venv/bin/python -m orange_quant.lgb.train     cn-lgb-momtopk
../.venv/bin/python -m orange_quant.lgb.backtest  cn-lgb-momtopk
# outputs/ + report.md / report_ic.png / report_deciles.png / positions.csv
```

A-share specifics handled: suspended holdings are valued at the last known
close (ffill) while trading still requires a real bar; `universe.min_amount`
(5000万) filters untradable small caps.

## Cross-sectional research toolkit (A/B)

The lgb pipeline is market-agnostic; every knob below is config-driven, and
generated variant configs live in `config/generated/` (gitignored, addressed
by the same bare name via `load_config` fallback):

| Script | Purpose |
|--------|---------|
| `scripts/gen_cn_ab_configs.py` | width tiers (`top50/300/800/2000`) + `cs_norm` variants + lambdarank + industry-neutral |
| `scripts/gen_binance_hour_ab.py` | 24 hour-of-day datasets (`data.hour_of_day`, one per UTC hour) |
| `scripts/fetch_cn_industry.py` | SW level-1 industry snapshot → `data/cn_industry.csv` (survivorship caveat) |
| `scripts/backtest_stability.py` | **generic stability check** for any config(s): sub-window IC/ICIR/RankIC/excess + block-bootstrap CI + cross-config ranking stability |
| `scripts/hour_of_day_stability.py` | the same machinery for the 24-hour experiment |

A/B discipline observed across all experiments (see ROADMAP for the full
record): one config knob at a time; report **both** the pure-alpha layer
(IC/ICIR, decile spread — from `report.py`) and the portfolio layer (net
excess, turnover) because they repeatedly diverge; validate the winner with
`backtest_stability.py` before adopting — full-window winners that do not
reproduce across sub-windows are regime noise.

## Experiment results

| Market | Universe | Period | Annual ret (net) | Sharpe | vs benchmark |
|--------|----------|--------|------------------|--------|--------------|
| A-share RL rotation | top-50 liquid (frozen 2012) | test 2023–2026 | −5.0% | −0.31 | equal-weight +3.0%/yr |
| Binance RL rotation | top-20 (frozen 2026) | test 2025–2026 | +70.9% | 0.55 | BTC +50.2%/yr (3-epoch smoke) |
| Binance LGB momtopk | top-50 (frozen 2026-08) | test 2026-02–08 | −8.2% (BTC −17.1%) | IC 0.043 | excess +11.2%/yr, IR 0.26 |
| A-share LGB momtopk | top-300 liquid (frozen 2017-12) | test 2025–2026 | +37.8% (SH000300 +14.0%) | 1.68 | excess +23.8%/yr, IC 0.045, ICIR 0.255 |

LGB A/B highlights (test 2025–2026, A-share top-300 pool; full record +
stability verdicts in ROADMAP):
- **Universe width** (C2): IC rises strictly with width — top-50 0.0001 → top-2000
  0.0570, ICIR 0.00 → 0.59. Net excess diverges from IC (top-800: IC up, excess
  down); top-2000's excess edge (+46%) is carried by one 60-day stretch
  (stability check: IC positive in all windows, excess wins 1/3 windows).
  top-300 is the most consistent portfolio performer (excess positive in all
  3 sub-windows).
- **`features.cs_norm: zscore`** (C4): IC above baseline in 3/3 sub-windows
  (0.065/0.041/0.048 vs 0.064/0.031/0.041), ICIR 0.255 → 0.296 — a stable,
  mild alpha-layer win; recommended as the default base for future A/Bs.
  `rank` not adopted (net excess collapses). 
- **lambdarank** (C5): decisively negative (RankIC −0.026 on test) — recorded,
  infrastructure kept.
- **Industry-neutral label** (C6): RankICIR 0.10 → 0.38 (stable across
  windows) but net excess collapses to −9.4%/yr (negative in 2/3 windows) —
  the whole-cross-section model implicitly times industry momentum, which was
  most of the return; not adopted.
- **Hour-of-day** (24 datasets, one per UTC hour): every hour carries positive
  test IC (0.013–0.060) — no clock hour is dead; hour-to-hour differences are
  noise (rankings unstable across sub-windows, best hour +175% ≈ expected
  max of 24 noise draws); the robust claim is "every hour weakly beats BTC".

Data-quality note: the new Tencent-only dataset fixes the legacy store's
2022-01~2022-05 data hole (4.5 months of zeros) and uses hfq (correct
ex-dividend returns); validation-segment mean reward improved from +0.87%/day
(legacy data) to +1.57%/day. Strategy alpha vs equal-weight is still an open
research question (see ROADMAP). Backtesting a freshly-retrained model over
the fixed test segment is NOT strictly out-of-sample (its train window
contains the test dates) — trust the walk-forward OOS numbers
(scripts/walkforward.py) for honest estimates; live deployment itself is
always OOS (the retrain at day T only sees data before T).

## Live retraining (walk-forward schedule)

The `binance-live` Docker profile includes a dedicated retraining service. It
runs on the first day of Jan/Apr/Jul/Oct at 03:00 UTC, retries failures after
24 hours, and stores idempotency state in
`models/binance-rl-rotation/retrain_schedule.json`:

```bash
docker compose --profile binance-live up -d --build
docker compose logs -f orange-quant-binance-retrain
```

For a non-Docker deployment, schedule the same job with cron:

```bash
# quarterly, from cron (e.g. 1st of Jan/Apr/Jul/Oct 03:00):
# 0 3 1 1,4,7,10 * cd /path/to/orange-quant && .venv/bin/python -m scripts.retrain_live --config binance-rl-rotation
# 0 3 1 1,4,7,10 * cd /path/to/orange-quant && .venv/bin/python -m scripts.retrain_live --config csi300-rl-rotation
```
A-share (Tencent) refresh is incremental too (CSVs extended from their last
date; A-share valid windows auto-extend to cover the training horizon on
trading calendars).
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
