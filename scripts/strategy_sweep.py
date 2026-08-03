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


def get_signal(base_config: str, phase: int, venue: str, refit: bool = False):
    """Train phase ``phase`` once and cache its test-segment signal."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{base_config}-h{phase:02d}.pkl"
    if cache.exists() and not refit:
        print(f"[sweep] phase {phase:02d}: using cached signal ({cache})")
        return pickle.load(open(cache, "rb"))

    from orange_quant.experiment import QuantExperiment
    from scripts.phase_study import make_phase_config

    config_path = make_phase_config(base_config, phase, venue)
    print(f"[sweep] phase {phase:02d}: training from {config_path}")
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
              & (df["risk_degree"] == cur.get("risk_degree"))]
    if cell.empty:
        print("\n[calibration] grid does not contain the config's own settings; skipped")
        return

    ref = pd.read_csv(ref_path).set_index("phase")["ann_return"]
    print("\n[calibration] sweep vs phase_study at the config's own settings "
          f"(topk={cur.get('topk')} n_drop={cur.get('n_drop')} "
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
                 risk_degree: float) -> dict:
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
        hold_thresh=strat_cfg.get("hold_thresh", 1),
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
    return {"ann_return": float(stats["annualized_return"]),
            "info_ratio": float(stats["information_ratio"]),
            "max_drawdown": float(stats["max_drawdown"])}


def main():
    parser = argparse.ArgumentParser(description="Portfolio-parameter sweep across phases")
    parser.add_argument("--venue", default="binance")
    parser.add_argument("--config", default="binance-lgb-momtopk")
    parser.add_argument("--phases", default="0,6,12,18")
    parser.add_argument("--topk", default="10,15,20,30")
    parser.add_argument("--n-drop", default="5")
    parser.add_argument("--risk-degree", default="0.95,0.70")
    parser.add_argument("--refit", action="store_true", help="Ignore cached signals")
    args = parser.parse_args()

    phases = [int(p) for p in args.phases.split(",") if p.strip()]
    topks = [int(t) for t in args.topk.split(",") if t.strip()]
    n_drops = [int(d) for d in args.n_drop.split(",") if d.strip()]
    risk_degrees = [float(r) for r in args.risk_degree.split(",") if r.strip()]

    config = load_config(args.config)
    cur = config.get("strategy", {}).get("kwargs", {})
    print("=" * 70)
    print(f"🔧 Strategy sweep — {args.config}, phases {phases}")
    print(f"   current: topk={cur.get('topk')} n_drop={cur.get('n_drop')} "
          f"risk_degree={cur.get('risk_degree')}")
    print(f"   grid: topk={topks} n_drop={n_drops} risk_degree={risk_degrees} "
          f"({len(topks)*len(n_drops)*len(risk_degrees)} settings x {len(phases)} phases)")
    print("=" * 70)

    # Phase-major: a phase's signal and all its backtests run while qlib points
    # at that phase's store. Looping parameter-major would price every phase
    # against whichever store was initialized last.
    rows = []
    for phase in phases:
        signal = get_signal(args.config, phase, args.venue, refit=args.refit)
        ensure_qlib(args.venue, phase)
        print(f"[sweep] phase {phase:02d}: backtesting against "
              f"{phase_provider_uri(args.venue, phase)}")
        for topk, n_drop, rd in product(topks, n_drops, risk_degrees):
            if n_drop > topk:
                continue  # cannot drop more names than the book holds
            try:
                m = run_backtest(signal, config, topk, n_drop, rd)
            except Exception as e:  # noqa: BLE001 — one cell must not lose the grid
                print(f"  ❌ topk={topk} n_drop={n_drop} rd={rd} phase={phase}: {e}")
                continue
            rows.append({"topk": topk, "n_drop": n_drop, "risk_degree": rd,
                         "phase": phase, **m})
            print(f"    topk={topk:3d} n_drop={n_drop} rd={rd:.2f}  "
                  f"ann_return={m['ann_return']:+.2%}")

    if not rows:
        raise SystemExit("Every backtest failed; nothing to report")

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / f"{args.config}-sweep.csv", index=False)

    # Rank by the worst phase — robustness, not peak performance
    summary = df.groupby(["topk", "n_drop", "risk_degree"]).agg(
        worst_ann=("ann_return", "min"),
        mean_ann=("ann_return", "mean"),
        best_ann=("ann_return", "max"),
        worst_ir=("info_ratio", "min"),
        mean_ir=("info_ratio", "mean"),
        worst_mdd=("max_drawdown", "min"),
    )
    summary["spread"] = summary["best_ann"] - summary["worst_ann"]
    summary = summary.sort_values("worst_ann", ascending=False)

    calibration_check(df, config, args.config)

    print("\n" + "=" * 70)
    print("📊 Per-phase detail")
    print("=" * 70)
    print(df.pivot_table(index=["topk", "n_drop", "risk_degree"], columns="phase",
                         values="ann_return").to_string(float_format=lambda v: f"{v: .3f}"))
    print("\n" + "=" * 70)
    print("🏆 Ranked by WORST phase (robustness objective)")
    print("=" * 70)
    print(summary.to_string(float_format=lambda v: f"{v: .3f}"))
    print(f"\n💾 {RESULTS_DIR / f'{args.config}-sweep.csv'}")


if __name__ == "__main__":
    main()
