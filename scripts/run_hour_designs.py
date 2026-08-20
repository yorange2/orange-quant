"""Backtest all four hour-of-day designs on all 24 clock anchors.

The question this answers: given 24 hourly-shifted "daily" series, is it better
to train one model per anchor, average those 24 models, or pool the anchors
into a single training set? The four designs:

  ``single``     per-anchor model, scored and executed on its own anchor
                 (``binance-lgb-momtopk-h00..h23``)
  ``ens``        the 24 per-anchor models averaged as cross-sectional
                 z-scores, executed on each anchor in turn
  ``pooled``     one model fit on all 24 anchors stacked, hour discarded
  ``pooled-hf``  same, plus hour sin/cos/raw features
  ``pooled-ens`` the two stacked, **calendar-day aligned**: the pooled model
                 scores all 24 anchor feature series for the same calendar day,
                 those scores are cross-sectionally z-scored and averaged. One
                 model, 24 views.
  ``pooled-hf-ens`` the same stack on the hour-feature model, kept only as a
                 noise floor for ``pooled-ens`` (the two base models were shown
                 to be information-equivalent)
  ``pooled-ens-causal`` the same stack, **rolling-24h aligned** — each view is
                 lagged to the target anchor's own information cutoff. The
                 ``-causal`` suffix is kept because output paths already carry
                 it; prose should say "rolling 24h", which describes what the
                 window IS rather than a property relative to an unstated
                 reference point.
  ``ens-causal`` the 24-model ensemble under the same rolling-24h alignment.

Naming note: "causal" here means causal **with respect to the decision moment**
— the calendar-day variant uses views that close up to 23h later than the
target's own cutoff. Measured against the EXECUTION moment (a further ~24h out)
both variants are causal; neither has look-ahead. See ``scripts/
test_causal_alignment.py --explain`` for the clock arithmetic.

Caliber warning for per-anchor spread: the calendar-day blend is
anchor-independent by construction (``build_ens_blend`` returns one date→row
map that all 24 anchors share), so its 24 backtests differ only in execution
bars, not in signal. Its per-anchor σ is therefore not comparable with that of
``single``/``pooled``/the rolling-24h designs, which do produce 24 distinct
signals.

Every design produces 24 backtests on 24 near-identical execution calendars,
so the output is four *distributions* of Sharpe, not four point estimates.
With a 179-day test window the spread within a design is the headline number:
a design whose Sharpe swings by more than the gap between designs has not
demonstrated anything.

Scores are computed once and reused: the ensemble's blend is anchor-independent
(only the execution calendar changes), and the pooled models are shared booster
sets behind per-anchor proxies. Datasets are cached in-process — otherwise the
ensemble alone would reload 24 × 24 npz files.

Run from orange-quant/ (after ``gen_binance_hour_pooled.py`` for both modes)::

    ../.venv/bin/python scripts/run_hour_designs.py
    ../.venv/bin/python scripts/run_hour_designs.py --designs single ens
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from orange_quant.lgb.backtest import benchmark_nav, compute_ic, run_backtest
from orange_quant.lgb.dataset import load_or_build
from orange_quant.rl.dataset import load_config
from orange_quant.rl.metrics import bars_per_year, return_metrics

BASE = "binance-lgb-momtopk"
HOURS = list(range(24))
DESIGNS = ["single", "ens", "pooled", "pooled-hf", "pooled-ens", "pooled-hf-ens",
           "pooled-ens-causal", "ens-causal"]
# design → the per-anchor scorer whose 24 outputs get blended
BLENDED = {"ens": "single", "pooled-ens": "pooled", "pooled-hf-ens": "pooled-hf"}
# same, but each member is lagged to the target anchor's own information cutoff
CAUSAL = {"pooled-ens-causal": "pooled", "ens-causal": "single"}
OUT = Path("outputs/hour-designs")

_DS_CACHE: dict[str, object] = {}
_LAG: int | None = None


def anchor_lag() -> int:
    """``label.exec_lag`` of the active config family (all 24 anchors share it)."""
    global _LAG
    if _LAG is None:
        from orange_quant.lgb.dataset import exec_lag

        _LAG = exec_lag(load_config(f"{BASE}-h00"))
    return _LAG


def ds_for(hour: int):
    """The anchor's dataset — cached, these are 80 MB npz files."""
    name = f"{BASE}-h{hour:02d}"
    if name not in _DS_CACHE:
        _DS_CACHE[name] = load_or_build(load_config(name))
    return _DS_CACHE[name]


def decision_range(ds):
    """[first, last] decision-day indices: the backtest marks out at test_end."""
    test_s, test_e = ds.split_idx["test"]
    return test_s, test_e - 1 - anchor_lag()


def anchor_model(name: str):
    with open(Path("models") / name / "model.pkl", "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------
# scores per design
# --------------------------------------------------------------------------
def scores_single(hour: int) -> np.ndarray:
    ds = ds_for(hour)
    t0, t1 = decision_range(ds)
    model = anchor_model(f"{BASE}-h{hour:02d}")
    block = ds.feats[t0 : t1 + 1]
    return model.predict(block.reshape(-1, ds.n_feats)).reshape(block.shape[:2])


def scores_pooled(hour: int, mode: str) -> np.ndarray:
    ds = ds_for(hour)
    t0, t1 = decision_range(ds)
    model = anchor_model(f"{BASE}-{mode}-h{hour:02d}")
    block = ds.feats[t0 : t1 + 1]
    return model.predict(block.reshape(-1, ds.n_feats)).reshape(block.shape[:2])


def _cs_zscore(row: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score of one date's scores (NaN-safe)."""
    out = np.full_like(row, np.nan, dtype=np.float64)
    m = np.isfinite(row)
    if m.sum() > 2:
        std = row[m].std(ddof=1)
        if std > 0:
            out[m] = (row[m] - row[m].mean()) / std
    return out


def base_scorer(design: str):
    """The per-anchor score function that a blended design averages over."""
    src = {**BLENDED, **CAUSAL}[design]
    return scores_single if src == "single" else (lambda h: scores_pooled(h, src))


def build_ens_blend(score_fn) -> dict:
    """{date → (N,) mean z-score} over the 24 anchors' scores.

    Anchor-independent by construction, so it is built once and then sliced
    onto whichever anchor's decision calendar is being executed. With
    ``scores_single`` the members are 24 different models; with a pooled
    scorer they are 24 views of ONE model — the blend then averages over the
    feature window's phase rather than over model-fitting noise.
    """
    zsum: dict = defaultdict(lambda: None)
    cnt: dict = defaultdict(lambda: None)
    for h in HOURS:
        ds = ds_for(h)
        t0, t1 = decision_range(ds)
        p = score_fn(h)
        for k, d in enumerate(ds.dates[t0 : t1 + 1]):
            d = np.datetime64(d, "D")
            z = _cs_zscore(p[k])
            have = np.isfinite(z)
            if zsum[d] is None:
                zsum[d] = np.zeros(ds.n_stocks)
                cnt[d] = np.zeros(ds.n_stocks)
            zsum[d] += np.where(have, z, 0.0)
            cnt[d] += have
    blend = {}
    for d, s in zsum.items():
        c = cnt[d]
        v = s / np.where(c > 0, c, 1.0)
        v[c == 0] = np.nan
        blend[d] = v
    print(f"[designs] ensemble blend: {len(blend)} dates")
    return blend


def scores_ens(hour: int, blend: dict) -> np.ndarray:
    ds = ds_for(hour)
    t0, t1 = decision_range(ds)
    return np.vstack([blend[np.datetime64(d, "D")]
                      for d in ds.dates[t0 : t1 + 1]])


def build_zmaps(score_fn) -> dict:
    """{anchor → {date → cross-sectionally z-scored score row}}."""
    zmaps = {}
    for h in HOURS:
        ds = ds_for(h)
        t0, t1 = decision_range(ds)
        p = score_fn(h)
        zmaps[h] = {np.datetime64(d, "D"): _cs_zscore(p[k])
                    for k, d in enumerate(ds.dates[t0 : t1 + 1])}
    return zmaps


def scores_ens_causal(hour: int, zmaps: dict) -> np.ndarray:
    """Rolling-24h blend: every view closes at or before this anchor's decision.

    Binance stamps a kline with its OPEN time, so anchor h's bar dated t closes
    at ``t (h+1):00`` — anchors h' > h close later that calendar day than the
    target does. The calendar-day blend therefore hands anchor h a signal up to
    23h fresher than its own cutoff. Legitimate against the execution price
    (which is a further ~24h out) but it confounds the comparison against the
    single-anchor designs, which get no such freshness.

    Here each member contributes its most recent view at or before the target's
    own cutoff: same calendar day for h' <= h, previous day for h' > h. The
    result is a genuine trailing-24h rolling blend — the version you could
    actually trade at hour h with no informational edge over the base design.
    """
    ds = ds_for(hour)
    t0, t1 = decision_range(ds)
    one_day = np.timedelta64(1, "D")
    out = np.full((t1 - t0 + 1, ds.n_stocks), np.nan)
    for k, d in enumerate(ds.dates[t0 : t1 + 1]):
        d = np.datetime64(d, "D")
        acc = np.zeros(ds.n_stocks)
        cnt = np.zeros(ds.n_stocks)
        for hp in HOURS:
            row = zmaps[hp].get(d if hp <= hour else d - one_day)
            if row is None:
                continue
            have = np.isfinite(row)
            acc += np.where(have, row, 0.0)
            cnt += have
        out[k] = np.where(cnt > 0, acc / np.where(cnt > 0, cnt, 1.0), np.nan)
    return out


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def evaluate(design: str, hour: int, preds: np.ndarray,
             topk: int | None = None, cost_rate: float | None = None) -> dict:
    """Backtest ``preds`` on the anchor's own bars; returns the metrics dict.

    ``topk`` overrides the portfolio width. IC is computed from the same
    ``preds`` regardless, so sweeping topk separates signal quality (fixed)
    from portfolio construction (varying) — the check on whether a Sharpe gain
    is real alpha or the top of the ranking getting lucky.

    ``cost_rate`` overrides the per-side fee. A design that only wins at the
    modelled fee is not deployable: turnover differences between designs (and
    between topk settings) only show up in the P&L once the fee is realistic,
    so the edge has to be re-read at the fee actually paid.
    """
    ds = ds_for(hour)
    cfg = load_config(f"{BASE}-h{hour:02d}")
    if topk is not None:
        cfg["strategy"] = {**cfg["strategy"], "topk": int(topk)}
    if cost_rate is not None:
        cfg["backtest"] = {**cfg["backtest"], "cost_rate": float(cost_rate)}
    t0, t1 = decision_range(ds)

    rl, _ = run_backtest(cfg, ds, preds)
    bmark, ew = benchmark_nav(ds, cfg, t0, t1)
    t = min(len(rl), len(bmark), len(ew))
    bpy = bars_per_year(cfg)
    m = return_metrics(rl["nav"].to_numpy()[:t], benchmark_nav=bmark[:t],
                       turnover=rl["turnover"].to_numpy()[:t],
                       annualize=bpy, bars_per_year=bpy)
    m.update(compute_ic(preds, ds.label, t0, t1))
    m["design"] = design
    m["hour"] = hour
    m["topk"] = int(cfg["strategy"]["topk"])
    m["cost_rate"] = float(cfg["backtest"]["cost_rate"])
    m["policy_nav"] = float(rl["nav"].iloc[-1])
    m["benchmark_nav"] = float(bmark[-1])
    m["equal_weight_nav"] = float(ew[-1])
    # daily excess-over-BTC series, kept for the paired bootstrap
    nav = rl["nav"].to_numpy()[:t]
    r_pol = np.diff(np.concatenate([[1.0], nav])) / np.concatenate([[1.0], nav[:-1]])
    r_bm = np.diff(np.concatenate([[1.0], bmark[:t]])) / np.concatenate([[1.0], bmark[:t - 1]])
    m["_daily_ret"] = r_pol.tolist()
    m["_daily_excess"] = (r_pol - r_bm).tolist()
    return m


def main() -> None:
    global BASE, _LAG

    ap = argparse.ArgumentParser(description="Backtest the 4 hour designs × 24 anchors")
    ap.add_argument("--designs", nargs="*", default=DESIGNS, choices=DESIGNS)
    ap.add_argument("--hours", nargs="*", type=int, default=HOURS)
    ap.add_argument("--topk", nargs="*", type=int, default=None,
                    help="sweep portfolio widths; scores are computed once and "
                         "re-run through the portfolio for each width")
    ap.add_argument("--cost-rate", nargs="*", type=float, default=None,
                    help="sweep per-side fee (e.g. 0.001 0.002)")
    ap.add_argument("--out", default=None, help="output basename under outputs/hour-designs")
    ap.add_argument("--base", default=BASE,
                    help="config family, e.g. binance-lgb-momtopk-lag0")
    args = ap.parse_args()

    BASE, _LAG = args.base, None
    print(f"[designs] base={BASE} exec_lag={anchor_lag()}")

    OUT.mkdir(parents=True, exist_ok=True)
    blends = {d: build_ens_blend(base_scorer(d))
              for d in args.designs if d in BLENDED}
    zmaps = {d: build_zmaps(base_scorer(d))
             for d in args.designs if d in CAUSAL}

    rows = []
    for design in args.designs:
        for h in args.hours:
            if design == "single":
                preds = scores_single(h)
            elif design in BLENDED:
                preds = scores_ens(h, blends[design])
            elif design in CAUSAL:
                preds = scores_ens_causal(h, zmaps[design])
            else:
                preds = scores_pooled(h, design)
            for k in (args.topk or [None]):
                for c in (args.cost_rate or [None]):
                    m = evaluate(design, h, preds, topk=k, cost_rate=c)
                    rows.append(m)
                    print(f"[designs] {design:<18} h{h:02d} topk={m['topk']:<3} "
                          f"cost={m['cost_rate']:.4f} "
                          f"sharpe={m['sharpe']:+.3f}  IR={m['information_ratio']:+.3f}  "
                          f"turn={m['annual_turnover']:.1f}")

    base = args.out or "raw"
    (OUT / f"{base}.json").write_text(json.dumps(rows, ensure_ascii=False))
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in rows])
    csv = OUT / (f"{base}.csv" if args.out else "per_anchor.csv")
    df.to_csv(csv, index=False)
    print(f"\n[designs] wrote {csv} ({len(df)} rows) + {base}.json")


if __name__ == "__main__":
    main()
