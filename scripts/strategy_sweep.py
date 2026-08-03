#!/usr/bin/env python3
"""
Sweep portfolio-construction parameters against the phase-robustness objective.

The phase study (scripts/phase_study.py) showed the ranking signal is stable
across day boundaries — Rank IC 0.041-0.057 on Binance — while realized return
is not, spanning -18% to +177%. That gap says the variance lives in portfolio
construction, not the model, so the levers worth sweeping are topk and
risk_degree rather than anything about the fit.

Because those levers touch only the backtest, each phase is trained **once** and
its test-segment signal cached; every parameter set then re-backtests the same
signal. A grid of N settings costs 4 trainings and 4N backtests, not 4N
trainings.

Ranking criterion is the **worst phase**, not the mean. A setting that earns
+177% on one boundary and -18% on another is not better than one that earns +40%
on all four — the whole point is to stop optimising a number that moves 80pp
when the day boundary shifts.

Usage:
    python -m scripts.strategy_sweep
    python -m scripts.strategy_sweep --topk 10,20,30 --risk-degree 0.95,0.7
    python -m scripts.strategy_sweep --refit          # ignore cached signals
"""

import argparse
import pickle
from itertools import product
from pathlib import Path

import pandas as pd
import yaml

CACHE_DIR = Path("data/phase_study/signals")
RESULTS_DIR = Path("data/phase_study")

# Expanding-window walk-forward. Every window trains from the same 2020-01 start
# and slides valid/test forward, so each is a genuinely out-of-sample ~6 months
# and together they cover 2024-07..2026-06. "D" reproduces the config's own
# splits, which is what makes it comparable to the single-window phase study.
WINDOWS = [
    {"name": "A", "train": {"start": "2020-01-01", "end": "2023-12-31"},
     "valid": {"start": "2024-01-01", "end": "2024-06-30"},
     "test":  {"start": "2024-07-01", "end": "2024-12-31"}},
    {"name": "B", "train": {"start": "2020-01-01", "end": "2024-06-30"},
     "valid": {"start": "2024-07-01", "end": "2024-12-31"},
     "test":  {"start": "2025-01-01", "end": "2025-06-30"}},
    {"name": "C", "train": {"start": "2020-01-01", "end": "2024-12-31"},
     "valid": {"start": "2025-01-01", "end": "2025-06-30"},
     "test":  {"start": "2025-07-01", "end": "2025-12-31"}},
    {"name": "D", "train": {"start": "2020-01-01", "end": "2025-06-30"},
     "valid": {"start": "2025-07-01", "end": "2026-01-31"},
     "test":  {"start": "2026-02-01", "end": "2026-06-27"}},
]
WINDOWS_BY_NAME = {w["name"]: w for w in WINDOWS}
BASELINE_WINDOW = "D"  # the one whose splits match config/*.yaml


def quarterly_windows(first="2022-01-01", last="2026-06-30",
                      valid_months=6, train_start="2020-01-01"):
    """Rolling 3-month test windows, each with its own expanding train segment.

    A/B/C/D give only four out-of-sample observations, which is far too few to
    fit anything regime-related on without repeating the single-window mistake
    at a larger scale. Quarterly steps turn the same history into ~18 windows.
    Every window keeps ``valid`` immediately before ``test`` and trains on
    everything before that, so no window ever sees its own test period.
    """
    out, start = [], pd.Timestamp(first)
    last = pd.Timestamp(last)
    while start <= last:
        test_end = start + pd.DateOffset(months=3) - pd.Timedelta(days=1)
        if test_end > last:
            break
        valid_start = start - pd.DateOffset(months=valid_months)
        out.append({
            "name": f"{start.year}Q{(start.month - 1) // 3 + 1}",
            "train": {"start": train_start,
                      "end": str((valid_start - pd.Timedelta(days=1)).date())},
            "valid": {"start": str(valid_start.date()),
                      "end": str((start - pd.Timedelta(days=1)).date())},
            "test":  {"start": str(start.date()), "end": str(test_end.date())},
        })
        start += pd.DateOffset(months=3)
    return out


QUARTERLY = quarterly_windows()
WINDOWS_BY_NAME.update({w["name"]: w for w in QUARTERLY})


def load_config(name: str) -> dict:
    with open(f"config/{name}.yaml") as f:
        return yaml.safe_load(f)


def phase_provider_uri(venue: str, phase: int) -> str:
    return f"data/qlib_data/{venue}_h{phase:02d}"


def ensure_qlib(venue: str, phase: int) -> None:
    """Point qlib at this phase's store before backtesting against it.

    The backtest reads prices and the benchmark from whatever store qlib was
    last initialized with, which is *not* necessarily the phase the signal came
    from — a cached signal skips training, and training is the only thing that
    would otherwise have called qlib.init. Getting this wrong silently prices
    every phase's signal against one store's bars.
    """
    import qlib
    from orange_quant.experiment import _end_active_experiment

    _end_active_experiment()
    qlib.init(provider_uri=phase_provider_uri(venue, phase), region="cn")


def get_signal(base_config: str, phase: int, venue: str, refit: bool = False,
               window: dict = None):
    """Train one (phase, window) cell once and cache its test-segment signal.

    The cache key carries the window name because different windows train on
    different data and predict different dates — sharing a key across them would
    silently backtest one window's parameters against another's signal.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"h{phase:02d}" + (f"-w{window['name']}" if window else "")
    cache = CACHE_DIR / f"{base_config}-{tag}.pkl"
    if cache.exists() and not refit:
        print(f"[sweep] {tag}: using cached signal ({cache})")
        return pickle.load(open(cache, "rb"))

    from orange_quant.experiment import QuantExperiment
    from scripts.phase_study import make_phase_config

    config_path = make_phase_config(base_config, phase, venue, window=window)
    print(f"[sweep] {tag}: training from {config_path}")
    exp = QuantExperiment.from_yaml(str(config_path))
    pred = exp.fit_predict()["predictions"]
    pickle.dump(pred, open(cache, "wb"))
    return pred


def calibration_check(df: pd.DataFrame, config: dict, base_config: str) -> None:
    """Compare the config's own settings against the phase study's numbers.

    The sweep reaches the backtest by a different route than phase_study
    (hand-assembled qlib.backtest + risk_analysis vs PortAnaRecord). Where the
    grid happens to contain the settings the config already uses, the two must
    agree — if they don't, the grid is measuring something else and none of it
    can be trusted.
    """
    ref_path = RESULTS_DIR / f"{base_config}-phases.csv"
    if not ref_path.exists():
        print("\n[calibration] no phase-study results to compare against; skipped")
        return

    cur = config.get("strategy", {}).get("kwargs", {})
    cell = df[(df["topk"] == cur.get("topk")) & (df["n_drop"] == cur.get("n_drop"))
              & (df["hold_thresh"] == cur.get("hold_thresh"))
              & (df["risk_degree"] == cur.get("risk_degree"))]
    # phase_study runs the config's own splits, so only the matching window is
    # comparable; the other windows train on different data by construction.
    if "window" in cell:
        cell = cell[cell["window"] == BASELINE_WINDOW]
    if cell.empty:
        print("\n[calibration] grid does not contain the config's own settings; skipped")
        return

    ref = pd.read_csv(ref_path).set_index("phase")["ann_return"]
    print("\n[calibration] sweep vs phase_study at the config's own settings "
          f"(topk={cur.get('topk')} n_drop={cur.get('n_drop')} "
          f"hold_thresh={cur.get('hold_thresh')} "
          f"risk_degree={cur.get('risk_degree')}):")
    worst = 0.0
    for _, row in cell.iterrows():
        phase = int(row["phase"])
        if phase not in ref.index:
            continue
        diff = abs(row["ann_return"] - ref[phase])
        worst = max(worst, diff)
        flag = "✅" if diff < 5e-3 else "❌"
        print(f"  {flag} phase {phase:02d}: sweep={row['ann_return']:+.4f} "
              f"phase_study={ref[phase]:+.4f}  diff={diff:.4f}")
    if worst >= 5e-3:
        print(f"  ❌ max diff {worst:.4f} — the two paths disagree; "
              f"treat the grid below as invalid until this is resolved.")
    else:
        print(f"  ✅ agreement within {worst:.4f}")


def run_backtest(signal, config: dict, topk: int, n_drop: int,
                 risk_degree: float, hold_thresh: int = None) -> dict:
    """Backtest one parameter set on an already-computed signal."""
    from qlib.backtest import backtest
    from qlib.contrib.evaluate import risk_analysis
    from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy

    bt_cfg = config.get("backtest", {})
    strat_cfg = config.get("strategy", {}).get("kwargs", {})

    strategy = TopkDropoutStrategy(
        signal=signal,
        topk=topk,
        n_drop=n_drop,
        hold_thresh=(hold_thresh if hold_thresh is not None
                     else strat_cfg.get("hold_thresh", 1)),
        risk_degree=risk_degree,
    )
    executor = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
    }

    dates = signal.index.get_level_values("datetime")
    portfolio, _ = backtest(
        start_time=dates.min(), end_time=dates.max(),
        strategy=strategy, executor=executor,
        benchmark=bt_cfg.get("benchmark", "BTC"),
        account=bt_cfg.get("account", 1000000),
        exchange_kwargs=bt_cfg.get("exchange_kwargs", {}),
    )

    report = portfolio["1day"][0]
    # qlib's own definition of excess return with cost, matching PortAnaRecord
    excess = report["return"] - report["bench"] - report["cost"]
    stats = risk_analysis(excess, freq="day")["risk"]

    # Absolute (USDT) return alongside the excess one. These can point opposite
    # ways: in a window where BTC rallies, a basket that merely lags BTC looks
    # catastrophic in excess terms while still making money, and vice versa. The
    # account experiences the absolute number, so both belong in the grid.
    abs_stats = risk_analysis(report["return"] - report["cost"], freq="day")["risk"]
    bench_ann = float(risk_analysis(report["bench"], freq="day")["risk"]
                      ["annualized_return"])
    return {"ann_return": float(stats["annualized_return"]),
            "info_ratio": float(stats["information_ratio"]),
            "max_drawdown": float(stats["max_drawdown"]),
            "abs_ann_return": float(abs_stats["annualized_return"]),
            "abs_info_ratio": float(abs_stats["information_ratio"]),
            "abs_max_drawdown": float(abs_stats["max_drawdown"]),
            "bench_ann_return": bench_ann}


def main():
    parser = argparse.ArgumentParser(description="Portfolio-parameter sweep across phases")
    parser.add_argument("--venue", default="binance")
    parser.add_argument("--config", default="binance-lgb-momtopk")
    parser.add_argument("--phases", default="0,6,12,18")
    parser.add_argument("--topk", default="10,15,20,30")
    parser.add_argument("--n-drop", default="5")
    parser.add_argument("--risk-degree", default="0.95,0.70")
    parser.add_argument("--hold-thresh", default=None,
                        help="Comma-separated grid; default is the config's own value")
    parser.add_argument("--refit", action="store_true", help="Ignore cached signals")
    parser.add_argument("--windows", default=BASELINE_WINDOW,
                        help=f"Walk-forward windows to evaluate, e.g. A,B,C,D "
                             f"(default: {BASELINE_WINDOW}, the config's own splits)")
    args = parser.parse_args()

    if args.windows.strip().lower() == "quarterly":
        windows = QUARTERLY
    else:
        windows = [WINDOWS_BY_NAME[w.strip()]
                   for w in args.windows.split(",") if w.strip()]

    phases = [int(p) for p in args.phases.split(",") if p.strip()]
    topks = [int(t) for t in args.topk.split(",") if t.strip()]
    n_drops = [int(d) for d in args.n_drop.split(",") if d.strip()]
    risk_degrees = [float(r) for r in args.risk_degree.split(",") if r.strip()]

    config = load_config(args.config)
    cur = config.get("strategy", {}).get("kwargs", {})
    hold_threshs = ([int(h) for h in args.hold_thresh.split(",") if h.strip()]
                    if args.hold_thresh else [cur.get("hold_thresh", 1)])

    print("=" * 70)
    print(f"🔧 Strategy sweep — {args.config}, phases {phases}")
    print(f"   current: topk={cur.get('topk')} n_drop={cur.get('n_drop')} "
          f"hold_thresh={cur.get('hold_thresh')} risk_degree={cur.get('risk_degree')}")
    print(f"   grid: topk={topks} n_drop={n_drops} hold_thresh={hold_threshs} "
          f"risk_degree={risk_degrees} "
          f"({len(topks)*len(n_drops)*len(hold_threshs)*len(risk_degrees)} settings "
          f"x {len(phases)} phases x {len(windows)} windows)")
    print(f"   windows: {[w['name'] for w in windows]}")
    print("=" * 70)

    # Phase-major within each window: a cell's signal and all its backtests run
    # while qlib points at that phase's store. Looping parameter-major would
    # price every phase against whichever store was initialized last.
    rows = []
    for window in windows:
        for phase in phases:
            signal = get_signal(args.config, phase, args.venue,
                                refit=args.refit, window=window)
            ensure_qlib(args.venue, phase)
            print(f"[sweep] w{window['name']} h{phase:02d}: backtesting against "
                  f"{phase_provider_uri(args.venue, phase)}")
            for topk, n_drop, ht, rd in product(topks, n_drops, hold_threshs,
                                                risk_degrees):
                if n_drop > topk:
                    continue  # cannot drop more names than the book holds
                try:
                    m = run_backtest(signal, config, topk, n_drop, rd, hold_thresh=ht)
                except Exception as e:  # noqa: BLE001 — one cell must not lose the grid
                    print(f"  ❌ topk={topk} n_drop={n_drop} ht={ht} rd={rd} "
                          f"w{window['name']} phase={phase}: {e}")
                    continue
                rows.append({"topk": topk, "n_drop": n_drop, "hold_thresh": ht,
                             "risk_degree": rd, "window": window["name"],
                             "phase": phase, **m})
                print(f"    topk={topk:3d} n_drop={n_drop} ht={ht} rd={rd:.2f}  "
                      f"ann_return={m['ann_return']:+.2%}")

    if not rows:
        raise SystemExit("Every backtest failed; nothing to report")

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / f"{args.config}-sweep.csv", index=False)

    # Rank by the worst cell — robustness, not peak performance. Neither window
    # nor phase is a group key, so the aggregates run over every (window, phase)
    # cell at once: "worst" means worst day-boundary in the worst OOS period.
    summary = df.groupby(["topk", "n_drop", "hold_thresh", "risk_degree"]).agg(
        worst_ann=("ann_return", "min"),
        mean_ann=("ann_return", "mean"),
        best_ann=("ann_return", "max"),
        worst_ir=("info_ratio", "min"),
        mean_ir=("info_ratio", "mean"),
        worst_mdd=("max_drawdown", "min"),
        worst_abs=("abs_ann_return", "min"),
        mean_abs=("abs_ann_return", "mean"),
        worst_abs_mdd=("abs_max_drawdown", "min"),
    )
    summary["spread"] = summary["best_ann"] - summary["worst_ann"]
    summary = summary.sort_values("worst_ann", ascending=False)

    calibration_check(df, config, args.config)

    idx = ["topk", "n_drop", "hold_thresh", "risk_degree"]
    print("\n" + "=" * 70)
    print("📊 Per-cell detail (window x phase)")
    print("=" * 70)
    print(df.pivot_table(index=idx, columns=["window", "phase"], values="ann_return")
            .to_string(float_format=lambda v: f"{v: .3f}"))
    if df["window"].nunique() > 1:
        print("\nMean EXCESS return by window (is the edge present in every OOS period?):")
        print(df.pivot_table(index=idx, columns="window", values="ann_return")
                .to_string(float_format=lambda v: f"{v: .3f}"))
        print("\nMean ABSOLUTE (USDT) return by window — what the account actually earns:")
        print(df.pivot_table(index=idx, columns="window", values="abs_ann_return")
                .to_string(float_format=lambda v: f"{v: .3f}"))
        print("\nBenchmark (BTC) annualized by window, for reference:")
        print(df.groupby("window")["bench_ann_return"].mean()
                .to_string(float_format=lambda v: f"{v: .3f}"))
    print("\n" + "=" * 70)
    print("🏆 Ranked by WORST cell across all windows x phases")
    print("=" * 70)
    print(summary.to_string(float_format=lambda v: f"{v: .3f}"))
    print(f"\n💾 {RESULTS_DIR / f'{args.config}-sweep.csv'}")


if __name__ == "__main__":
    main()
