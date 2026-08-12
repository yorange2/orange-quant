"""Generic backtest stability analysis — sub-window split + block bootstrap.

Generalizes ``scripts/hour_of_day_stability.py`` to ANY trained lgb config
(needs outputs/<config>/nav.csv from a prior backtest). For each config:

  * loads the cached dataset + model, predicts the test decision days
    ([test_start, test_end-2], same window as the backtest);
  * splits the test segment into N non-overlapping sub-windows; per window:
    mean daily IC / ICIR / RankIC (vs the z-scored label) and the geometric
    annualized excess vs the config's benchmark (exec-day aligned, exactly
    the backtest's P&L window), annualized with the config's bars_per_year;
  * block bootstrap (10-day blocks) of the daily excess series → empirical
    SE + 95% CI of the full-window annualized excess.

With >= 2 configs, also reports cross-config ranking stability — the
multiple-comparisons guard:

  * pairwise sub-window Spearman correlations of the excess ranking (are the
    best configs the same in every window?);
  * per-window winners and whether the full-window winner wins each window;
  * how many configs are all-windows positive on IC and on excess.

Verdict convention (same as the hour-of-day analysis):
  * every window positive AND ranking stable     → structure, trustworthy;
  * every window positive but ranking ~0         → broad weak effect, no
    differentiation (the hour-of-day finding);
  * any window negative or ranking flips         → the full-window result is
    driven by one segment (regime), treat as noise.

Usage (from orange-quant/)::
    ../.venv/bin/python scripts/backtest_stability.py <config> [<config> ...] [--windows N]
    ../.venv/bin/python scripts/backtest_stability.py cn-lgb-momtopk-top50 cn-lgb-momtopk-top300
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from orange_quant.lgb.dataset import load_or_build
from orange_quant.rl.backtest import benchmark_closes
from orange_quant.rl.dataset import load_config
from orange_quant.rl.metrics import bars_per_year, per_date_corr

BLOCK = 10          # days per bootstrap block
N_REPS = 2000


def block_bootstrap(x: np.ndarray, seed: int = 0) -> tuple[float, tuple[float, float]]:
    """(SE, 95% CI) of the annualized mean of the daily series x (block bootstrap;
    the block length preserves the small autocorrelation of daily excess)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    nblocks = int(np.ceil(n / BLOCK))
    reps = np.empty(N_REPS)
    for r in range(N_REPS):
        idx = np.concatenate([
            np.arange(s, min(s + BLOCK, n))
            for _ in range(nblocks)
            for s in [rng.integers(0, max(n - BLOCK + 1, 1))]
        ])[:n]
        reps[r] = x[idx].mean() * 252
    return float(reps.std(ddof=1)), (float(np.percentile(reps, 2.5)),
                                     float(np.percentile(reps, 97.5)))


def analyze(cfg_name: str, n_windows: int) -> dict:
    cfg = load_config(cfg_name)
    ds = load_or_build(cfg)
    with open(Path(cfg["paths"]["model_dir"]) / "model.pkl", "rb") as f:
        model = pickle.load(f)
    test_s, test_e = ds.split_idx["test"]
    t1 = test_e - 2                                   # last decision day
    D = t1 - test_s + 1
    preds = model.predict(ds.feats[test_s : t1 + 1].reshape(-1, ds.n_feats)
                          ).reshape(D, ds.n_stocks)
    label = ds.label[test_s : t1 + 1]
    nav = pd.read_csv(Path(cfg["paths"]["output_dir"]) / "nav.csv")
    if len(nav) < D:
        raise SystemExit(f"[stability] {cfg_name}: nav.csv has {len(nav)} rows < {D} decision days")
    sret = nav["ret"].to_numpy()[:D]
    c = benchmark_closes(cfg, ds.dates).to_numpy()
    bret0 = np.zeros(len(c))
    ok = ~np.isnan(c[:-1])
    bret0[:-1][ok] = c[1:][ok] / c[:-1][ok] - 1.0
    bret = bret0[test_s + 1 : test_e][:D]             # exec-day benchmark
    if np.isnan(bret).any():
        raise SystemExit(f"[stability] {cfg_name}: benchmark has NaN in the test window")
    bpy = bars_per_year(cfg)
    ex = sret - bret

    bounds = np.linspace(0, D, n_windows + 1, dtype=int)
    wins = []
    for i in range(n_windows):
        a, b = bounds[i], bounds[i + 1]
        block = label[a:b]
        valid = ~np.isnan(block)
        rows_idx = np.argwhere(valid)
        ic = per_date_corr(preds[a:b][valid], block[valid], rows_idx[:, 0], "pearson")
        ric = per_date_corr(preds[a:b][valid], block[valid], rows_idx[:, 0], "spearman")
        e = float(np.prod((1.0 + sret[a:b]) / (1.0 + bret[a:b])) ** (bpy / max(b - a, 1)) - 1.0)
        wins.append({
            "days": int(b - a),
            "ic": float(ic.mean()) if len(ic) else float("nan"),
            "icir": float(ic.mean() / ic.std()) if len(ic) > 2 and ic.std() > 0 else float("nan"),
            "rank_ic": float(ric.mean()) if len(ric) else float("nan"),
            "excess": e,
        })
    e_full = float(np.prod((1.0 + sret) / (1.0 + bret)) ** (bpy / D) - 1.0)
    se, ci = block_bootstrap(ex)
    return {"name": cfg_name, "wins": wins, "excess_full": e_full,
            "ic_full": float(np.mean([w["ic"] for w in wins])),
            "se": se, "ci": ci, "bpy": bpy}


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest stability analysis")
    ap.add_argument("configs", nargs="+", help="config names (>=1; >=2 enables ranking checks)")
    ap.add_argument("--windows", type=int, default=3, help="number of sub-windows (default 3)")
    args = ap.parse_args()

    results = [analyze(c, args.windows) for c in args.configs]
    for r in results:
        print(f"=== {r['name']} ===")
        print(f"    {'窗口':<6} {'天数':>4} {'IC':>8} {'ICIR':>6} {'RankIC':>8} {'超额(年化)':>10}")
        for i, w in enumerate(r["wins"], 1):
            print(f"    sw{i:<5} {w['days']:>4} {w['ic']:>+8.4f} {w['icir']:>6.3f} "
                  f"{w['rank_ic']:>+8.4f} {w['excess']:>+9.1%}")
        print(f"    全窗  {r['wins'][0]['days'] * len(r['wins']):>4} "
              f"{np.nanmean([w['ic'] for w in r['wins']]):>+8.4f} "
              f"{np.nanmean([w['icir'] for w in r['wins']]):>6.3f} "
              f"{np.nanmean([w['rank_ic'] for w in r['wins']]):>+8.4f} "
              f"{r['excess_full']:>+9.1%}  bootstrap SE {r['se']:.1%}, "
              f"95% CI [{r['ci'][0]:+.1%}, {r['ci'][1]:+.1%}]")

    if len(results) < 2:
        return
    n = args.windows
    ex_mat = np.array([[r["wins"][i]["excess"] for i in range(n)] for r in results])
    ic_mat = np.array([[r["wins"][i]["ic"] for i in range(n)] for r in results])
    names = [r["name"] for r in results]
    print("\n[跨配置排名稳定性] (Spearman)")
    for i in range(n):
        for j in range(i + 1, n):
            print(f"    {names[0].split('-')[-1]} 超额 sw{i+1} vs sw{j+1}: "
                  f"{spearmanr(ex_mat[:, i], ex_mat[:, j]).statistic:+.3f}")
    print("    各窗口超额 winner:", [names[int(np.argmax(ex_mat[:, i]))] for i in range(n)])
    full_winner = names[int(np.argmax([r["excess_full"] for r in results]))]
    win_in = [names[int(np.argmax(ex_mat[:, i]))] == full_winner for i in range(n)]
    print(f"    全窗超额最佳 [{full_winner}] 在 {sum(win_in)}/{n} 个子窗口仍最佳: {win_in}")
    print("    IC 各窗口 winner:", [names[int(np.argmax(ic_mat[:, i]))] for i in range(n)])
    print("[普遍性] 超额三窗全正的配置:",
          f"{sum(all(x > 0 for x in ex_mat[i]) for i in range(len(results)))}/{len(results)} "
          f"({[names[i] for i in range(len(results)) if all(x > 0 for x in ex_mat[i])]})")
    print("          IC 三窗全正的配置:",
          f"{sum(all(x > 0 for x in ic_mat[i]) for i in range(len(results)))}/{len(results)} "
          f"({[names[i] for i in range(len(results)) if all(x > 0 for x in ic_mat[i])]})")


if __name__ == "__main__":
    main()
