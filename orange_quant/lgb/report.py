"""Cross-sectional evaluation report (roadmap C3).

Pure-alpha diagnostics for one backtest run, computed over the test-segment
decision days — no portfolio mechanics, so results are comparable across
universe widths (roadmap C2) without noise from TopkDropout execution:

  * per-decision-day Pearson / Rank IC series → mean, std, ICIR, and a
    cumulative-IC chart (per-day IC cumsum, the standard "is the signal
    decaying?" view);
  * per-year IC breakdown (mean / ICIR per calendar year);
  * decile analysis: each decision day, ranks the predictions into 10 equal
    buckets and records the realized next-day return (ret[t+1], the same
    P&L window as the strategy: signal at t, execute at close[t+1], earn
    close[t+2]/close[t+1] - 1). Mean daily return per decile → the
    monotonicity ladder; the long-short spread (decile 10 − decile 1) is a
    market-beta-free alpha measure — evaluation only, A-shares are not
    shortable in production.

Writes outputs/<config>/report.md + report_ic.png + report_deciles.png,
called from ``orange_quant.lgb.backtest`` after the portfolio metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from orange_quant.lgb.dataset import LGBDataset
from orange_quant.rl.metrics import per_date_corr


def _decision_rows(ds: LGBDataset) -> Tuple[int, int]:
    """Decision-day span [test_start, test_end-2] (same as the backtest)."""
    s, e = ds.split_idx["test"]
    return s, e - 2


def decile_analysis(preds: np.ndarray, ret: np.ndarray, t0: int, t1: int,
                    n_deciles: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-decision-day decile buckets of realized next-day return.

    Returns (decile_means (K,) daily mean return per decile, asc, day_means (D,)
    per-day spread top−bottom, spread_tstat) — rows where < 20 names have a
    prediction are skipped. ret[i] = close[i+1]/close[i] − 1, so the P&L row for
    decision day t is ret[t+1], matching the strategy's execution-day window.
    """
    sums = np.zeros(n_deciles)
    counts = np.zeros(n_deciles)
    spreads = []
    for k, t in enumerate(range(t0, t1 + 1)):
        p = preds[k]
        r = ret[t + 1]
        valid = ~np.isnan(p)
        n = int(valid.sum())
        if n < 20:
            continue
        pv, rv = p[valid], r[valid]
        # rankdata breaks LightGBM leaf ties so qcut never drops duplicates
        b = pd.qcut(rankdata(pv), n_deciles, labels=False)
        for d in range(n_deciles):
            m = b == d
            sums[d] += rv[m].sum()
            counts[d] += m.sum()
        top = rv[b == n_deciles - 1].mean()
        bot = rv[b == 0].mean()
        spreads.append(top - bot)
    means = np.divide(sums, counts, out=np.full(n_deciles, np.nan),
                      where=counts > 0)
    spread_days = np.asarray(spreads)
    tstat = np.nan
    if len(spread_days) > 2:
        sd = spread_days.std(ddof=1)
        if sd > 0:
            tstat = float(spread_days.mean() / sd * np.sqrt(len(spread_days)))
    return means, spread_days, tstat


def ic_series(preds: np.ndarray, label: np.ndarray, t0: int, t1: int,
              dates: np.ndarray) -> pd.DataFrame:
    """Per-decision-day Pearson/Rank IC vs the z-scored label → DataFrame."""
    block = label[t0 : t1 + 1]
    valid = ~np.isnan(block)
    rows = np.argwhere(valid)
    date_idx = rows[:, 0]
    ics = per_date_corr(preds[valid], block[valid], date_idx, "pearson")
    rics = per_date_corr(preds[valid], block[valid], date_idx, "spearman")
    # date_idx is block-relative; per_date_corr returns values in order of
    # np.unique(date_idx), so map back onto the absolute calendar
    uniq = np.unique(date_idx)
    return pd.DataFrame({
        "date": pd.to_datetime(dates[t0 + uniq]),
        "ic": ics,
        "rank_ic": rics,
    })


def _ic_table(ics: pd.DataFrame, label: str) -> str:
    mean, std = ics["ic"].mean(), ics["ic"].std()
    ric_mean, ric_std = ics["rank_ic"].mean(), ics["rank_ic"].std()
    return (f"| {label} | {mean:+.4f} | {std:.4f} | "
            f"{mean / std if std > 0 else float('nan'):+.4f} | "
            f"{ric_mean:+.4f} | {ric_std:.4f} | "
            f"{ric_mean / ric_std if ric_std > 0 else float('nan'):+.4f} | "
            f"{len(ics)} |")


def _industry_exposure(positions: list, ds: LGBDataset) -> str:
    """Markdown table of the TopK holdings' SW-industry exposure (roadmap C6).

    positions[k] = [(code_idx, weight)]; mean weight per industry over
    decision days, plus top-3 concentration and HHI of the mean weights.
    No industry map (crypto) → returns "".
    """
    from orange_quant.data.industry import load_industry_map

    ind = load_industry_map()
    if not ind:
        return ""
    per_day = []
    for day_pos in positions:
        w = {}
        for c, wt in day_pos:
            g = ind.get(ds.codes[c])
            if g:
                w[g] = w.get(g, 0.0) + wt
        per_day.append(w)
    names = sorted({g for w in per_day for g in w})
    mean_w = {g: sum(w.get(g, 0.0) for w in per_day) / len(per_day)
              for g in names}
    ranked = sorted(mean_w.items(), key=lambda kv: kv[1], reverse=True)
    hhi = sum(v * v for _, v in ranked)
    top3 = sum(v for _, v in ranked[:3])
    lines = [
        f"## TopK 持仓行业暴露（SW 一级，当前快照近似历史）",
        f"",
        f"| 行业 | 日均权重 |",
        f"|---|---|",
    ]
    lines += [f"| {g} | {v:.2%} |" for g, v in ranked if v > 0.01]
    lines += [
        f"",
        f"| 集中度指标 | 值 |",
        f"|---|---|",
        f"| top-3 行业合计 | {top3:.2%} |",
        f"| HHI（平均权重） | {hhi:.4f} |",
        f"| 行业数（>1% 权重） | {sum(1 for _, v in ranked if v > 0.01)} |",
        f"",
    ]
    return "\n".join(lines)


def generate_report(cfg: dict, ds: LGBDataset, preds: np.ndarray,
                    out_dir: Path, positions: list | None = None) -> None:
    """Write report.md + report_ic.png + report_deciles.png into out_dir.

    ``positions`` ([(code_idx, weight)] per decision day) adds the TopK
    holdings' industry-exposure section (roadmap C6) when the SW industry map
    is available.
    """
    t0, t1 = _decision_rows(ds)
    ics = ic_series(preds, ds.label, t0, t1, ds.dates)
    means, spread_days, spread_t = decile_analysis(preds, ds.ret, t0, t1)

    years = pd.to_datetime(ics["date"]).dt.year
    lines = [
        f"# {cfg['market']['venue']} LGB 横截面评估报告",
        f"",
        f"配置 `{cfg['market']['venue']}-lgb-momtopk`，test 段决策日 "
        f"{pd.Timestamp(ds.dates[t0]).date()} ~ {pd.Timestamp(ds.dates[t1]).date()}，"
        f"共 {len(ics)} 天，截面 {ds.n_stocks} 只。",
        f"",
        f"## IC 概览（预测 vs 次日收益 z-score）",
        f"",
        f"| 指标 | IC mean | IC std | ICIR | RankIC mean | RankIC std | RankICIR | 天数 |",
        f"|---|---|---|---|---|---|---|---|",
        _ic_table(ics, "全 test 段"),
        f"## 分年度 IC",
        f"",
        f"| 年份 | 天数 | IC mean | ICIR | RankIC mean | RankICIR |",
        f"|---|---|---|---|---|---|",
    ]
    for y, g in ics.groupby(years):
        m, s = g["ic"].mean(), g["ic"].std()
        rm, rs = g["rank_ic"].mean(), g["rank_ic"].std()
        lines.append(f"| {y} | {len(g)} | {m:+.4f} | "
                     f"{m / s if s > 0 else float('nan'):+.4f} | "
                     f"{rm:+.4f} | {rm / rs if rs > 0 else float('nan'):+.4f} |")

    lines += [
        f"",
        f"## Decile 单调性（预测分 10 档 → 次日实际收益，等权）",
        f"",
        f"| decile | 日均收益 | 年化（×252） |",
        f"|---|---|---|",
    ]
    for d in range(len(means)):
        if np.isnan(means[d]):
            continue
        lines.append(f"| D{d + 1} | {means[d]:+.5f} | {means[d] * 252:+.2%} |")
    lines += [
        f"",
        f"## 多空 spread（D10 − D1 日收益，纯 alpha 度量，实盘不做空）",
        f"",
        f"| 指标 | 值 |",
        f"|---|---|",
        f"| 日均 spread | {float(np.nanmean(spread_days)):+.5f} |",
        f"| 年化 spread | {float(np.nanmean(spread_days)) * 252:+.2%} |",
        f"| spread 标准差 | {float(np.nanstd(spread_days)):.5f} |",
        f"| t 统计量 | {spread_t:+.2f} |",
        f"| 有效天数 | {len(spread_days)} |",
        f"",
        f"## 说明",
        f"",
        f"- IC/ICIR 与 decile 均按**决策日**计算（信号日 t → 执行日 t+1 收盘 → "
        f"收益 close[t+2]/close[t+1]−1，与回测 P&L 窗口一致）。",
        f"- 停牌日收益按 0 计（数据集约定，与策略 NAV 一致）。",
        f"- 多空 spread 剥离市场 beta，仅作评估；A 股实盘无法做空，不进入交易。",
        f"",
    ]
    if positions:
        lines += [_industry_exposure(positions, ds), ""]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    dates = ics["date"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, ics["ic"].cumsum(), label="累计 Pearson IC", linewidth=1.5)
    ax.plot(dates, ics["rank_ic"].cumsum(), label="累计 Rank IC", linewidth=1.5)
    ax.set_title(f"{cfg['market']['venue']} LGB 累计 IC（test 段决策日）")
    ax.set_ylabel("累计 IC")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "report_ic.png", dpi=150)

    fig, ax = plt.subplots(figsize=(10, 5))
    dls = np.arange(1, len(means) + 1)
    ax.bar(dls, means * 252, color="#4472C4")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{cfg['market']['venue']} LGB decile 收益阶梯（年化 %）")
    ax.set_xlabel("decile（1 = 预测最弱）")
    ax.set_ylabel("年化收益")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "report_deciles.png", dpi=150)
    plt.close("all")

    print(f"[lgb-report] ICIR={ics['ic'].mean() / ics['ic'].std():+.4f} "
          f"RankICIR={ics['rank_ic'].mean() / ics['rank_ic'].std():+.4f}, "
          f"decile spread D10−D1 日均={np.nanmean(spread_days):+.5f} "
          f"(t={spread_t:+.2f})")
    print(f"[lgb-report] report.md + report_ic.png + report_deciles.png → {out_dir}/")
