"""Performance metrics for a daily NAV series (with optional benchmark)."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

_DEFAULT_BARS_PER_YEAR = 252.0


def bars_per_year(cfg: dict) -> float:
    """Annualization factor for one config, declared in ``backtest.bars_per_year``.

    Sourced from config rather than per-module constants or a runtime guess at
    the row spacing, so every strategy's Sharpe/IR on the same market is
    comparable.
    """
    return float((cfg.get("backtest") or {}).get(
        "bars_per_year", _DEFAULT_BARS_PER_YEAR))


def per_date_corr(pred: np.ndarray, y: np.ndarray, date_idx: np.ndarray,
                  method: str = "pearson") -> np.ndarray:
    """Per-date correlation (pearson/spearman) of pred vs y; NaN stats dropped.

    ``date_idx`` labels each row with its date; dates with < 2 valid rows are
    skipped. Shared so valid-time and test-time IC use one definition.
    """
    from scipy.stats import pearsonr, spearmanr

    fn = pearsonr if method == "pearson" else spearmanr
    vals = []
    for t in np.unique(date_idx):
        m = date_idx == t
        if m.sum() < 2:
            continue
        r = fn(pred[m], y[m])
        if not np.isnan(r.statistic):
            vals.append(r.statistic)
    return np.asarray(vals, dtype=np.float64)


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
