"""Compare the four hour-of-day designs: distributions, not point estimates.

Consumes ``outputs/hour-designs/raw.json`` (written by ``run_hour_designs.py``)
and answers three questions in order of how much they should move a decision:

1. **How large is anchor luck?** Within each design, the spread of Sharpe over
   the 24 clock anchors. This is a free robustness check — 24 backtests of the
   same strategy on the same 179 days, differing only in what time of day the
   "daily" bar closes. If the within-design spread swamps the between-design
   gap, no design has been shown to be better.

2. **Do the designs differ, anchor by anchor?** A paired read across the 24
   anchors (each design scored on the same execution calendar), plus the
   IC/RankIC comparison — IC is the cleaner statistic here because it is not
   filtered through a 20-name TopK portfolio.

3. **Does the gap survive resampling?** A moving-block bootstrap (block = 10
   days, preserving the autocorrelation the 24h label induces) on the daily
   excess-over-BTC series of the anchor-averaged portfolio.

Anchors are deliberately NOT treated as 24 independent samples: they overlap
23/24 by construction, so the paired table is descriptive and every inferential
statement comes from the block bootstrap on the time dimension instead.

Run from orange-quant/::
    ../.venv/bin/python scripts/analyze_hour_designs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("outputs/hour-designs")
DESIGNS = ["single", "ens", "ens-causal", "pooled", "pooled-hf",
           "pooled-ens", "pooled-hf-ens", "pooled-ens-causal"]
LABEL = {
    "single": "方案2成员 (每锚点1模型)",
    "ens": "方案2 (24模型集成·日历日)",
    "ens-causal": "方案2 (24模型集成·滚动24h)",
    "pooled": "方案1 (池化单模型)",
    "pooled-hf": "方案3 (池化+小时特征)",
    "pooled-ens": "方案4 (池化×24视角·日历日)",
    "pooled-hf-ens": "方案4' (噪声地板参照)",
    "pooled-ens-causal": "方案4 (池化×24视角·滚动24h)",
}
SHARED_SIGNAL = {"ens", "pooled-ens", "pooled-hf-ens"}   # 见 σ 口径注记
# “日历日”= 24 个视角都取同一日历日；“滚动24h”= 每个视角都截止于目标锚点自己的
# 决策时刻之前（即相对决策时刻因果）。两者相对执行时刻都无前视，差别是新鲜度。
# 代码 ID 保留 -causal 后缀（产物文件名已固化），措辞以“滚动24h”为准。
BPY = 238.0          # crypto-daily annualization used across the repo
BLOCK = 10
N_BOOT = 5000
RNG = np.random.default_rng(20260819)


def sharpe(r: np.ndarray) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(BPY)) if sd > 0 else np.nan


def block_bootstrap_idx(n: int, block: int, rng) -> np.ndarray:
    """Moving-block bootstrap indices (circular), length n."""
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    idx = np.concatenate([(s + np.arange(block)) % n for s in starts])
    return idx[:n]


def main() -> None:
    rows = json.loads((OUT / "raw.json").read_text())
    by = {(r["design"], r["hour"]): r for r in rows}
    designs = [d for d in DESIGNS if any(r["design"] == d for r in rows)]
    hours = sorted({r["hour"] for r in rows})

    # ---------------------------------------------------------------- 1
    print("=" * 78)
    print("1. 锚点运气：每个设计在 24 个时钟锚点上的表现分布")
    print("=" * 78)
    summary = []
    for d in designs:
        s = np.array([by[(d, h)]["sharpe"] for h in hours])
        ir = np.array([by[(d, h)]["information_ratio"] for h in hours])
        ic = np.array([by[(d, h)]["ic_mean"] for h in hours])
        ric = np.array([by[(d, h)]["rank_ic_mean"] for h in hours])
        to = np.array([by[(d, h)]["annual_turnover"] for h in hours])
        summary.append({
            "design": LABEL[d],
            "Sharpe均值": s.mean(), "Sharpe标准差": s.std(ddof=1),
            "Sharpe最差": s.min(), "Sharpe最好": s.max(),
            "Sharpe>0占比": float((s > 0).mean()),
            "IR均值": ir.mean(), "IC均值": ic.mean(), "RankIC均值": ric.mean(),
            "年换手": to.mean(),
        })
    df = pd.DataFrame(summary).set_index("design")
    print(df.to_string(float_format=lambda x: f"{x:+.4f}"))

    spread = {d: np.ptp([by[(d, h)]["sharpe"] for h in hours]) for d in designs}
    gap = (max(r["Sharpe均值"] for r in summary)
           - min(r["Sharpe均值"] for r in summary))
    print(f"\n锚点内极差(最大者): {max(spread.values()):.3f}"
          f"   设计间均值最大差: {gap:.3f}"
          f"   → 比值 {max(spread.values()) / max(gap, 1e-9):.1f}×")

    shared = [LABEL[d] for d in designs if d in SHARED_SIGNAL]
    if shared:
        print("\n⚠ σ 口径不一致：按日历日对齐的混合分数与目标锚点无关（24 个锚点共用")
        print("  同一份分数），所以下列设计的 σ 只反映执行价差异，不含信号差异——")
        print("  比其余设计的 σ 系统性偏小，两者不可直接比较：")
        print("    " + "、".join(shared))

    # ---------------------------------------------------------------- 2
    print("\n" + "=" * 78)
    print("2. 逐锚点配对对比（同一执行日历，同一 179 天）")
    print("=" * 78)
    base = "single"
    if base in designs:
        tbl = []
        for d in designs:
            if d == base:
                continue
            ds_ = np.array([by[(d, h)]["sharpe"] - by[(base, h)]["sharpe"]
                            for h in hours])
            dic = np.array([by[(d, h)]["ic_mean"] - by[(base, h)]["ic_mean"]
                            for h in hours])
            dric = np.array([by[(d, h)]["rank_ic_mean"] - by[(base, h)]["rank_ic_mean"]
                             for h in hours])
            tbl.append({
                "vs 每锚点1模型": LABEL[d],
                "ΔSharpe均值": ds_.mean(), "ΔSharpe胜率": float((ds_ > 0).mean()),
                "ΔIC均值": dic.mean(), "ΔIC胜率": float((dic > 0).mean()),
                "ΔRankIC均值": dric.mean(),
            })
        print(pd.DataFrame(tbl).set_index("vs 每锚点1模型").to_string(
            float_format=lambda x: f"{x:+.4f}"))
        print("\n注：24 个锚点重叠 23/24，胜率不是 24 个独立观测，仅作描述。")

    # ---------------------------------------------------------------- 3
    print("\n" + "=" * 78)
    print(f"3. 分块自助法（block={BLOCK}天, {N_BOOT}次）— 锚点平均组合的超额收益")
    print("=" * 78)
    series = {}
    n = min(len(by[(d, h)]["_daily_excess"]) for d in designs for h in hours)
    for d in designs:
        stack = np.array([by[(d, h)]["_daily_excess"][:n] for h in hours])
        series[d] = stack.mean(axis=0)          # equal-weight over anchors
    print(f"对齐后每条序列 {n} 个交易日\n")

    boots = {d: np.empty(N_BOOT) for d in designs}
    for b in range(N_BOOT):
        idx = block_bootstrap_idx(n, BLOCK, RNG)   # shared → paired resample
        for d in designs:
            boots[d][b] = sharpe(series[d][idx])
    for d in designs:
        pt = sharpe(series[d])
        lo, hi = np.percentile(boots[d], [2.5, 97.5])
        print(f"{LABEL[d]:<24} 超额Sharpe {pt:+.3f}  95%CI [{lo:+.3f}, {hi:+.3f}]")

    if base in designs:
        print()
        for d in designs:
            if d == base:
                continue
            diff = boots[d] - boots[base]
            pt = sharpe(series[d]) - sharpe(series[base])
            lo, hi = np.percentile(diff, [2.5, 97.5])
            p = float((diff <= 0).mean())
            verdict = "CI 不含 0" if lo > 0 or hi < 0 else "CI 含 0 → 无法区分"
            print(f"Δ vs 每锚点1模型: {LABEL[d]:<24} {pt:+.3f}  "
                  f"95%CI [{lo:+.3f}, {hi:+.3f}]  P(Δ≤0)={p:.3f}  {verdict}")

    # ---------------------------------------------------------------- 4
    print("\n" + "=" * 78)
    print("4. 子窗口复检（repo 采纳门槛：每个窗口都为正才算结构）")
    print("=" * 78)
    n_win = 3
    edges = np.linspace(0, n, n_win + 1).astype(int)
    tbl = []
    for d in designs:
        row = {"design": LABEL[d]}
        for w in range(n_win):
            seg = series[d][edges[w]:edges[w + 1]]
            row[f"sw{w + 1}超额Sharpe"] = sharpe(seg)
        row["全窗全正"] = all(row[f"sw{w + 1}超额Sharpe"] > 0 for w in range(n_win))
        tbl.append(row)
    print(pd.DataFrame(tbl).set_index("design").to_string(
        float_format=lambda x: f"{x:+.3f}"))

    # ---------------------------------------------------------------- 5
    print("\n" + "=" * 78)
    print("5. 噪声地板：pooled-hf 对 pooled（小时特征增益 0.004% → 信息等价）")
    print("=" * 78)
    if "pooled" in designs and "pooled-hf" in designs:
        dd = np.array([by[("pooled-hf", h)]["sharpe"] - by[("pooled", h)]["sharpe"]
                       for h in hours])
        print(f"逐锚点 ΔSharpe: 均值 {dd.mean():+.3f}  标准差 {dd.std(ddof=1):.3f}  "
              f"最大绝对值 {np.abs(dd).max():.3f}")
        print("→ 单锚点上 |ΔSharpe| 小于约 1.7（2σ）的任何对比都读不出信息；")
        print("  24 锚点均值差才是可读的量级（此处噪声对只差 "
              f"{dd.mean():+.3f}，池化对单锚点差 "
              f"{np.mean([by[('pooled', h)]['sharpe'] - by[('single', h)]['sharpe'] for h in hours]):+.3f}）。")

    df.to_csv(OUT / "summary.csv")
    print(f"\n[analyze] {OUT}/summary.csv")


if __name__ == "__main__":
    main()
