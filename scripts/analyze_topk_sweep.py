"""Does the blended design's Sharpe edge survive changing the portfolio width?

The 24-view blend gained ~+1.0 to +1.4 Sharpe over the plain pooled model while
gaining only +0.003 to +0.007 IC. Those two numbers are wildly out of
proportion, which leaves two explanations:

  * **real** — the blend cleans up the TOP of the ranking specifically, which
    is exactly what a topk portfolio trades and what IC (an all-names
    correlation) barely measures. Then the edge should persist, and fade only
    as topk widens toward the full universe;
  * **luck** — the blend happened to put a few big winners inside the top 20 in
    this one 179-day window. Then the edge should collapse as soon as the
    portfolio width changes, with no orderly pattern.

topk=40 of 48 coins is the control: at that width almost nothing is selected
away, so every design must converge toward equal-weight and the gaps must
shrink to ~0. A design still showing a large edge at topk=40 would indicate a
bug, not alpha.

IC is identical across widths by construction (it is computed from the same
scores), so it appears once per design as the fixed signal-quality reference.

Run from orange-quant/ (after ``run_hour_designs.py --topk ... --out topk_sweep``)::
    ../.venv/bin/python scripts/analyze_topk_sweep.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("outputs/hour-designs")
LABEL = {
    "single": "每锚点一模型",
    "ens": "24 模型集成",
    "pooled": "池化单模型",
    "pooled-ens": "池化×24视角·日历日",
    "pooled-ens-causal": "池化×24视角·滚动24h",
}
BPY = 238.0
BLOCK = 10
N_BOOT = 4000
RNG = np.random.default_rng(20260819)


def sharpe(r: np.ndarray) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(BPY)) if sd > 0 else np.nan


def block_idx(n: int, block: int, rng) -> np.ndarray:
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    return np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]


def main() -> None:
    rows = json.loads((OUT / "topk_sweep.json").read_text())
    by = {(r["design"], r["hour"], r["topk"]): r for r in rows}
    designs = list(dict.fromkeys(r["design"] for r in rows))
    hours = sorted({r["hour"] for r in rows})
    topks = sorted({r["topk"] for r in rows})

    # ------------------------------------------------------------------ 1
    print("=" * 88)
    print("1. 24 锚点平均 Sharpe × 组合宽度")
    print("=" * 88)
    grid = {}
    for d in designs:
        for k in topks:
            grid[(d, k)] = float(np.mean([by[(d, h, k)]["sharpe"] for h in hours]))
    df = pd.DataFrame({f"topk={k}": [grid[(d, k)] for d in designs] for k in topks},
                      index=[LABEL[d] for d in designs])
    # IC is a property of the scores, identical at every width — report the
    # 24-anchor mean (not h00's, which is one draw from a spread of ~0.013)
    df.insert(0, "IC(24锚点均值)",
              [float(np.mean([by[(d, h, topks[0])]["ic_mean"] for h in hours]))
               for d in designs])
    print(df.to_string(float_format=lambda x: f"{x:+.3f}"))

    # ------------------------------------------------------------------ 2
    print("\n" + "=" * 88)
    print("2. 相对池化单模型的 ΔSharpe（24 锚点均值）")
    print("=" * 88)
    base = "pooled"
    tbl = {}
    for d in designs:
        if d == base:
            continue
        tbl[LABEL[d]] = {f"topk={k}": grid[(d, k)] - grid[(base, k)] for k in topks}
    print(pd.DataFrame(tbl).T.to_string(float_format=lambda x: f"{x:+.3f}"))
    print("\ntopk=40/48 是对照：几乎不做选择，各设计必须收敛，Δ 应趋近 0。")

    # ------------------------------------------------------------------ 3
    print("\n" + "=" * 88)
    print(f"3. 分块自助（block={BLOCK}, {N_BOOT}次）Δ vs 池化 — 锚点平均超额序列")
    print("=" * 88)
    n = min(len(r["_daily_excess"]) for r in rows)
    series = {}
    for d in designs:
        for k in topks:
            stack = np.array([by[(d, h, k)]["_daily_excess"][:n] for h in hours])
            series[(d, k)] = stack.mean(axis=0)

    boots = {}
    idxs = [block_idx(n, BLOCK, RNG) for _ in range(N_BOOT)]
    for key, s in series.items():
        boots[key] = np.array([sharpe(s[i]) for i in idxs])

    for d in designs:
        if d == base:
            continue
        line = []
        for k in topks:
            diff = boots[(d, k)] - boots[(base, k)]
            pt = sharpe(series[(d, k)]) - sharpe(series[(base, k)])
            lo, hi = np.percentile(diff, [2.5, 97.5])
            mark = "*" if lo > 0 else " "
            line.append(f"k={k:<3}{pt:+.2f} [{lo:+.2f},{hi:+.2f}]{mark}")
        print(f"{LABEL[d]:<20} " + "  ".join(line))
    print("\n* = 95%CI 不含 0")

    # ------------------------------------------------------------------ 4
    print("\n" + "=" * 88)
    print("4. 判读")
    print("=" * 88)
    for d in ("pooled-ens", "pooled-ens-causal"):
        if d not in designs:
            continue
        deltas = [grid[(d, k)] - grid[(base, k)] for k in topks]
        narrow, wide = deltas[0], deltas[-1]
        mono = all(deltas[i] >= deltas[i + 1] - 1e-9 for i in range(len(deltas) - 1))
        print(f"{LABEL[d]}: Δ 从 topk={topks[0]} 的 {narrow:+.3f} "
              f"到 topk={topks[-1]} 的 {wide:+.3f}"
              f"   单调衰减={'是' if mono else '否'}")

    pd.DataFrame(grid.items()).to_csv(OUT / "topk_summary.csv", index=False)
    print(f"\n[analyze] {OUT}/topk_summary.csv")


if __name__ == "__main__":
    main()
