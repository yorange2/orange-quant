"""Performance metrics for a daily NAV series (with optional benchmark)."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def return_metrics(nav: np.ndarray,
                   benchmark_nav: Optional[np.ndarray] = None,
                   turnover: Optional[np.ndarray] = None,
                   annualize: float = 252.0,
                   bars_per_year: float = 252.0) -> Dict[str, float]:
    """Compute standard metrics from a NAV series.

    nav: (T,) cumulative NAV (nav[0] = 1 + first-day return).
    turnover: (T,) one-sided turnover |Δw| per bar; annualized as
        sum / years.
    annualize: bars per year for vol/sharpe scaling (252 daily, 6048 hourly).
    years is derived from bars_per_year so annualized returns stay correct
    at any frequency (defaults to annualize for back-compat).
    """
    rets = np.diff(nav) / nav[:-1]
    T = len(rets)
    years = T / bars_per_year
    total = nav[-1] / nav[0] - 1.0
    ann_ret = (nav[-1] / nav[0]) ** (1 / years) - 1 if years > 0 and nav[0] > 0 else np.nan
    ann_vol = float(np.std(rets, ddof=1) * np.sqrt(annualize))
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else np.nan

    peak = np.maximum.accumulate(nav)
    drawdown = nav / peak - 1.0
    max_dd = float(drawdown.min())

    win_rate = float((rets > 0).mean())
    worst_day = float(rets.min())
    best_day = float(rets.max())

    out: Dict[str, float] = {
        "total_return": float(total),
        "annual_return": float(ann_ret),
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "worst_day": worst_day,
        "best_day": best_day,
        "n_days": float(T),
    }

    if benchmark_nav is not None:
        brets = np.diff(benchmark_nav) / benchmark_nav[:-1]
        if len(brets) == T:
            excess = rets - brets
            btotal = benchmark_nav[-1] / benchmark_nav[0] - 1.0
            bann = (benchmark_nav[-1] / benchmark_nav[0]) ** (1 / years) - 1 \
                if years > 0 else np.nan
            bvol = float(np.std(brets, ddof=1) * np.sqrt(annualize))
            out.update({
                "benchmark_total_return": float(btotal),
                "benchmark_annual_return": float(bann),
                "excess_annual_return": float(ann_ret - bann),
                "information_ratio": float(np.mean(excess) / (np.std(excess, ddof=1) + 1e-12)
                                           * np.sqrt(annualize)),
            })

    if turnover is not None:
        out["annual_turnover"] = float(turnover.sum() / years)

    return out
