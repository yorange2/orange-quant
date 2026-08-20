"""Backtest: TopkDropout portfolio over the test segment, net of taker fees.

Timing is set by ``label.exec_lag`` (see ``dataset.exec_lag``), so the fill and
the training label can never drift apart: a signal from features at day t fills
at close[t+lag] and earns close[t+lag] → close[t+1+lag], which IS label[t].
Decision days run t ∈ [test_start, test_end−1−lag].

  * ``lag=1`` — the legacy qlib pipeline. The live runner trades on the signal
    day instead, so the backtest measures a strategy live does not run (the
    deviation is documented in runner.py).
  * ``lag=0`` — fill at close[t], matching the live runner: it places a market
    order minutes after the bar it decided on.

TopkDropout port (qlib contrib/strategy/signal_strategy.py):
  * last  = held coins sorted by pred desc (NaN pred ranks last)
  * today = top (n_drop + topk − len(last)) not-held coins by pred desc
  * comb  = last ∪ today sorted desc; sell = last ∩ bottom-n_drop(comb) —
    never sells a higher-ranked coin to buy a lower one
  * buy   = today[: len(sell) + topk − len(last)]
  * a coin is droppable only when held ≥ hold_thresh decision days
  * budget: value = cash_after_sells × risk_degree / len(buy) — risk_degree
    acts on CASH, not total equity (qlib semantics)
  * fills at close[t+1], cost_rate per side; coins without a bar on the
    execution day are skipped for both sell and buy (suspended → stay held)

Outputs to paths.output_dir/: nav.csv (same schema as rl/backtest.py),
metrics.json (return_metrics + IC/RankIC + benchmark/equal-weight NAV),
nav.png (3-line chart vs BTC and equal-weight).
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from orange_quant.lgb.dataset import LGBDataset, exec_lag, load_or_build, load_config
from orange_quant.lgb.ensemble import EnsembleLGB
from orange_quant.rl.backtest import benchmark_closes, plot_navs, write_metrics
from orange_quant.rl.metrics import bars_per_year, per_date_corr, return_metrics


def load_model(cfg: dict, ckpt: str | None = None) -> EnsembleLGB:
    path = Path(ckpt) if ckpt else Path(cfg["paths"]["model_dir"]) / "model.pkl"
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"[lgb-backtest] loaded model: {path}")
    return model


def predict_block(model: EnsembleLGB, feats: np.ndarray, t0: int, t1: int) -> np.ndarray:
    """Predict feats[t0:t1+1] at once → (D, N) float."""
    block = feats[t0 : t1 + 1]
    D, N, F = block.shape
    pred = model.predict(block.reshape(-1, F)).reshape(D, N)
    return pred


def compute_ic(pred: np.ndarray, label: np.ndarray, t0: int, t1: int) -> Dict[str, float]:
    """Per-decision-day Pearson/Spearman IC of pred vs label, mean ± std."""
    block = label[t0 : t1 + 1]
    valid = ~np.isnan(block)
    rows = np.argwhere(valid)
    flat_pred = pred[valid]
    flat_label = block[valid]
    date_idx = rows[:, 0]
    ics = per_date_corr(flat_pred, flat_label, date_idx, "pearson")
    rics = per_date_corr(flat_pred, flat_label, date_idx, "spearman")
    return {
        "ic_mean": float(ics.mean()) if len(ics) else float("nan"),
        "ic_std": float(ics.std()) if len(ics) else float("nan"),
        "icir": float(ics.mean() / ics.std()) if len(ics) > 2 and ics.std() > 0 else float("nan"),
        "rank_ic_mean": float(rics.mean()) if len(rics) else float("nan"),
        "rank_ic_std": float(rics.std()) if len(rics) else float("nan"),
        "rank_icir": float(rics.mean() / rics.std()) if len(rics) > 2 and rics.std() > 0 else float("nan"),
        "n_ic_days": int(len(ics)),
    }


def benchmark_nav(ds: LGBDataset, cfg: dict, t0: int, t1: int) -> Tuple[np.ndarray, np.ndarray]:
    """BTC and equal-weight NAVs over the same P&L windows as the strategy.

    Row k (signal day t) covers ret[t+lag] = close[t+1+lag]/close[t+lag] − 1,
    aligned with the strategy's drift for the same row.
    """
    lag = exec_lag(cfg)
    c = benchmark_closes(cfg, ds.dates).to_numpy()
    btc_ret = np.zeros(len(c))
    ok = ~np.isnan(c[:-1])
    btc_ret[:-1][ok] = c[1:][ok] / c[:-1][ok] - 1.0

    rows = [t + lag for t in range(t0, t1 + 1)]     # P&L window starts at exec day
    b = np.cumprod(1.0 + btc_ret[rows])
    ew = np.cumprod(1.0 + ds.ret[rows].mean(axis=1))
    return b, ew


def run_backtest(cfg: dict, ds: LGBDataset, preds: np.ndarray
                 ) -> Tuple[pd.DataFrame, List[Tuple[int, float]]]:
    """TopkDropout rollout over the test segment. ``preds`` is (D, N) for the
    decision days [test_start, test_end−2] (computed once by the caller).

    Returns (nav_df, positions) where positions[k] = [(code_idx, weight)] of
    the holdings at close of decision day k's execution (weight = value / NAV)
    — used for the industry-exposure report (roadmap C6)."""

    positions: List[List[Tuple[int, float]]]
    strat = cfg["strategy"]
    bt = cfg["backtest"]
    topk, n_drop, hold_thresh = (int(strat["topk"]), int(strat["n_drop"]),
                                 int(strat["hold_thresh"]))
    risk_degree, cost = float(strat["risk_degree"]), float(bt["cost_rate"])
    account = float(bt["account"])
    N = ds.n_stocks

    lag = exec_lag(cfg)
    test_s, test_e = ds.split_idx["test"]
    t1 = test_e - 1 - lag                           # last signal day
    close = ds.close                                # (T, N) NaN where no bar
    # A-share suspensions: a held name can go days/weeks without a bar. Trade
    # execution still requires a real bar (sells/buys skip NaN-price days), but
    # NAV valuation marks suspended holdings at the last known close (ffill).
    val_close = pd.DataFrame(close).ffill(axis=0).to_numpy(np.float64)

    cash = account
    holdings: Dict[int, float] = {}                 # coin index → shares
    entry: Dict[int, int] = {}                      # coin index → signal day

    def _pred(c: int, k: int) -> float:
        v = preds[k][c]
        return float(v) if not np.isnan(v) else -np.inf

    rows = []
    positions = []
    prev_nav = account
    for k, t in enumerate(range(test_s, t1 + 1)):
        exec_day = t + lag
        price = close[exec_day].astype(np.float64)  # (N,) — avoid float32 leakage
        vprice = val_close[exec_day]                # (N,) ffill'd for valuation
        held = list(holdings.keys())

        last = sorted(held, key=lambda c: _pred(c, k), reverse=True)
        n_new = n_drop + topk - len(last)
        not_held = [c for c in range(N) if c not in holdings]
        not_held.sort(key=lambda c: _pred(c, k), reverse=True)
        today = not_held[: max(n_new, 0)]
        comb = sorted(set(last) | set(today), key=lambda c: _pred(c, k), reverse=True)
        bottom = set(comb[-n_drop:]) if n_drop > 0 else set()
        sell = [c for c in last if c in bottom]
        buy = today[: max(len(sell) + topk - len(last), 0)]

        # ---- execute sells at close[exec_day] (suspended coins stay held) ----
        sell_value = 0.0
        for c in sell:
            if t - entry[c] + 1 < hold_thresh:
                continue                            # held too briefly
            if np.isnan(price[c]):
                continue                            # no bar → cannot sell
            shares = holdings.pop(c)
            del entry[c]
            proceeds = shares * price[c]
            cash += proceeds * (1.0 - cost)
            sell_value += proceeds

        # ---- buys: value = cash_after_sells × risk_degree / len(buy) ----
        buy_value = 0.0
        value = cash * risk_degree / len(buy) if buy else 0.0
        for c in buy:
            if value <= 0 or np.isnan(price[c]):
                continue
            shares = value / price[c]
            cash -= value * (1.0 + cost)
            holdings[c] = shares
            entry[c] = t
            buy_value += value

        nav = cash + sum(sh * vprice[c] for c, sh in holdings.items())
        max_w = max((sh * vprice[c] for c, sh in holdings.items()), default=0.0) / nav
        positions.append([(c, sh * vprice[c] / nav) for c, sh in holdings.items()])
        rows.append({
            "date": np.datetime_as_string(ds.dates[exec_day], unit="D"),
            "nav": nav / account,
            "ret": nav / prev_nav - 1.0,
            "turnover": (sell_value + buy_value) / 2.0 / nav,
            "n_held": len(holdings),
            "max_w": float(max_w),
            "in_cash": float(len(holdings) == 0),
        })
        prev_nav = nav

    return pd.DataFrame(rows), positions


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest LGB momtopk")
    parser.add_argument("config", help="config name without .yaml")
    parser.add_argument("--ckpt", default=None, help="override model.pkl path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = load_or_build(cfg)
    model = load_model(cfg, ckpt=args.ckpt)

    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    test_s, test_e = ds.split_idx["test"]
    t1 = test_e - 1 - exec_lag(cfg)
    preds = predict_block(model, ds.feats, test_s, t1)
    rl, positions = run_backtest(cfg, ds, preds)
    rl.to_csv(out_dir / "nav.csv", index=False)
    pos_df = pd.DataFrame(
        [{"date": d, "code": ds.codes[c], "weight": w}
         for day_pos, d in zip(positions, rl["date"])
         for c, w in day_pos])
    pos_df.to_csv(out_dir / "positions.csv", index=False)
    print(f"[lgb-backtest] {len(rl)} decision days, final NAV {rl['nav'].iloc[-1]:.4f}")

    bmark, ew = benchmark_nav(ds, cfg, test_s, t1)
    bmark_sym = cfg["market"]["benchmark_symbol"]
    t = min(len(rl), len(bmark), len(ew))
    bpy = bars_per_year(cfg)
    metrics = return_metrics(
        rl["nav"].to_numpy()[:t],
        benchmark_nav=bmark[:t],
        turnover=rl["turnover"].to_numpy()[:t],
        annualize=bpy,
        bars_per_year=bpy,
    )
    metrics.update(compute_ic(preds, ds.label, test_s, t1))
    metrics[f"benchmark_{bmark_sym}_nav"] = float(bmark[-1])
    metrics["benchmark_equal_weight_nav"] = float(ew[-1])
    metrics["policy_nav"] = float(rl["nav"].iloc[-1])
    write_metrics(metrics, out_dir, f"回测指标 (vs {bmark_sym})",
                  signed_prefixes=("annual", "excess", "information", "ic_", "rank_"))

    dates = pd.to_datetime(rl["date"]).to_numpy()
    plot_navs({"LGB 策略": rl["nav"].to_numpy()[:t],
               f"等权 top{ds.n_stocks}": ew[:t],
               bmark_sym: bmark[:t]},
              dates,
              f"{cfg['market']['venue']} LGB momtopk backtest "
              f"({dates[0].astype('datetime64[D]')} ~ "
              f"{dates[len(rl) - 1].astype('datetime64[D]')})",
              out_dir / "nav.png")

    from orange_quant.lgb.report import generate_report

    generate_report(cfg, ds, preds, out_dir, positions=positions)
    print(f"\n[lgb-backtest] outputs → {out_dir}/")
    print("[lgb-backtest] done")


if __name__ == "__main__":
    main()
