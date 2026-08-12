"""Hour-of-day A/B error compression (24 runs): sub-window stability + bootstrap SE.

For each of the 24 ``binance-lgb-momtopk-h{HH}`` runs (fixed-clock daily
series, test 2026-02~07):

  * split the test decision days into 3 non-overlapping ~60-day sub-windows;
    per sub-window: mean daily IC (pearson, vs the z-scored label) and the
    geometric annualized excess vs BTC on the same exec-day window;
  * block bootstrap (10-day blocks) of the daily excess series → empirical
    SE + 95% CI of the annualized excess (block length preserves the small
    autocorrelation of daily excess).

Checks against the multiple-comparisons trap:
  1. hour ranking stability: Spearman correlation of excess/ICIR rankings
     between sub-windows (and of each sub-window vs the full window);
  2. does the full-window best hour survive its bootstrap CI?
  3. is the observed best excess beyond pure noise — max z-score across the
     24 hours vs the expected max of 24 N(0,1) draws (~2.05)?

Run from orange-quant/::
    ../.venv/bin/python scripts/hour_of_day_stability.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from orange_quant.rl.backtest import benchmark_closes
from orange_quant.rl.dataset import load_config
from orange_quant.rl.metrics import per_date_corr
from orange_quant.lgb.dataset import load_or_build

N_SUB = 3
BLOCK = 10          # days per bootstrap block
N_REPS = 2000
PREFIX = "binance-lgb-momtopk-h"


def block_bootstrap(x: np.ndarray, seed: int = 0) -> tuple[float, tuple[float, float]]:
    """(SE, 95% CI) of the annualized mean of the daily series x (block bootstrap)."""
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


def main() -> None:
    rows = []
    for h in range(24):
        name = f"{PREFIX}{h:02d}"
        cfg = load_config(name)
        ds = load_or_build(cfg)
        with open(f"models/{name}/model.pkl", "rb") as f:
            model = pickle.load(f)
        test_s, test_e = ds.split_idx["test"]
        t1 = test_e - 2                                  # last decision day
        D = t1 - test_s + 1
        preds = model.predict(ds.feats[test_s : t1 + 1].reshape(-1, ds.n_feats)
                              ).reshape(D, ds.n_stocks)
        label = ds.label[test_s : t1 + 1]
        sret = pd.read_csv(f"outputs/{name}/nav.csv").iloc[:D]["ret"].to_numpy()
        c = benchmark_closes(cfg, ds.dates).to_numpy()
        bret0 = np.zeros(len(c))
        ok = ~np.isnan(c[:-1])
        bret0[:-1][ok] = c[1:][ok] / c[:-1][ok] - 1.0
        bret = bret0[test_s + 1 : test_e][:D]            # exec-day benchmark
        ex = sret - bret

        # ---- per-sub-window IC / excess (geometric annualized) ----
        bounds = np.linspace(0, D, N_SUB + 1, dtype=int)
        ic_sw, ex_sw = [], []
        for i in range(N_SUB):
            a, b = bounds[i], bounds[i + 1]
            block = label[a:b]
            valid = ~np.isnan(block)
            rows_idx = np.argwhere(valid)
            ic = per_date_corr(preds[a:b][valid], block[valid], rows_idx[:, 0], "pearson")
            ic_sw.append(float(ic.mean()) if len(ic) else float("nan"))
            e = np.prod((1.0 + sret[a:b]) / (1.0 + bret[a:b])) - 1.0
            ex_sw.append(float((1.0 + e) ** (252 / max(b - a, 1)) - 1.0))

        # ---- full-window bootstrap ----
        se, ci = block_bootstrap(ex)
        e_full = float(np.prod((1.0 + sret) / (1.0 + bret)) ** (252 / D) - 1.0)
        rows.append({"h": h, "ic_sw": ic_sw, "ex_sw": ex_sw,
                     "ic_full": float(np.mean(ic_sw)), "ex_full": e_full,
                     "se": se, "ci_lo": ci[0], "ci_hi": ci[1],
                     "z": e_full / se if se > 0 else float("nan")})

    df = pd.DataFrame(rows)
    print(f"{'UTC时':>4} {'全窗IC':>7} {'子窗超额1':>9} {'子窗超额2':>9} {'子窗超额3':>9} "
          f"{'全窗超额':>9} {'SE':>7} {'95%CI':>16} {'z':>5}")
    for r in sorted(rows, key=lambda x: x["h"]):
        print(f"{r['h']:>3}:00 {r['ic_full']:>+7.4f} "
              f"{'/'.join(f'{v:+.0%}' for v in r['ex_sw']):>11} "
              f"{r['ex_full']:>+8.1%} {r['se']:>7.1%} "
              f"[{r['ci_lo']:+.0%},{r['ci_hi']:+.0%}] {r['z']:>5.2f}")

    # ---- stability checks ----
    ex_mat = np.array([r["ex_sw"] for r in rows])
    ic_mat = np.array([r["ic_sw"] for r in rows])
    print("\n[1] 子窗口间超额排名稳定性 (Spearman):")
    for i in range(N_SUB):
        for j in range(i + 1, N_SUB):
            rho = spearmanr(ex_mat[:, i], ex_mat[:, j]).statistic
            print(f"    sw{i+1} vs sw{j+1}: {rho:+.3f}")
    print("[1b] 各子窗口 vs 全窗 超额排名:",
          [f"{spearmanr(ex_mat[:, i], df['ex_full']).statistic:+.2f}" for i in range(N_SUB)])
    print("[1c] 各子窗口 vs 全窗 IC 排名:",
          [f"{spearmanr(ic_mat[:, i], df['ic_full']).statistic:+.2f}" for i in range(N_SUB)])

    best = df.loc[df["ex_full"].idxmax()]
    print(f"\n[2] 全窗最佳小时 h{int(best['h']):02d}: 超额 {best['ex_full']:+.1%}, "
          f"bootstrap SE {best['se']:.1%}, 95% CI [{best['ci_lo']:+.1%}, {best['ci_hi']:+.1%}]")
    print(f"    它的三个子窗口超额: {[f'{v:+.1%}' for v in best['ex_sw']]}")
    print(f"    它是否在多数子窗口仍排名前 1/3: "
          f"{[int(np.argsort(-ex_mat[:, i])[:8].tolist().index(int(best['h']))) + 1 for i in range(N_SUB)]}")

    zs = df["z"].dropna()
    print(f"\n[3] 24 小时 max z = {zs.max():.2f} (纯噪声期望 ~{np.sqrt(2 * np.log(24)):.2f})；"
          f"z>2 的小时数: {int((zs > 2).sum())}/24, z>1.5: {int((zs > 1.5).sum())}/24")
    print(f"    若 24 小时同质, 预期 z>1.5 的个数 ≈ {24 * 0.067:.1f}, z>2 ≈ {24 * 0.023:.1f}")


if __name__ == "__main__":
    main()
