"""Alpha158 feature set ported from qlib to plain pandas.

Computes the full 158-feature Alpha158 set (qlib ``Alpha158DL`` with default
config) from a per-symbol OHLCV DataFrame. The formulas are ported verbatim
from the qlib checkout (``qlib/contrib/data/loader.py`` + ``qlib/data/ops.py``
+ ``qlib/data/_libs/rolling.pyx``) so results are comparable with the legacy
qlib pipeline's IC / excess-return numbers.

Porting notes (each was verified against the qlib source):
  * ``vwap`` is synthesized as ``close`` (the legacy Binance store did the
    same — a vwap column was required by Alpha158, so VWAP0 is the constant 1).
  * All rolling operators use ``min_periods=1`` (qlib's ``Rolling`` default);
    std/corr use pandas ddof=1 (matches pandas rolling, which qlib calls
    directly).
  * ``Greater``/``Less`` are elementwise ``maximum``/``minimum`` — NOT
    comparisons. Comparisons are ``Gt``/``Lt`` (used only in CNTP/CNTN).
  * ``Slope``/``Resi``/``Rsquare`` replicate qlib's Cython rolling regression:
    x = window position 1..N (offset-invariant for slope/r²; the residual is
    evaluated at the newest position, also offset-invariant), NaN values
    dropped before fitting.
  * ``Rsquare`` and ``Corr`` output NaN where the relevant rolling std is
    ``isclose(0, atol=2e-05)`` (qlib masks these).
  * Features keep raw NaN — LightGBM consumes missing values natively, and
    the legacy pipeline fed raw (un-normalized) Alpha158 features too.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

_WINDOWS = [5, 10, 20, 30, 60]
_ROLLING_OPS = [
    "ROC", "MA", "STD", "BETA", "RSQR", "RESI", "MAX", "MIN", "QTLU", "QTLD",
    "RANK", "RSV", "IMAX", "IMIN", "IMXD", "CORR", "CORD", "CNTP", "CNTN",
    "CNTD", "SUMP", "SUMN", "SUMD", "VMA", "VSTD", "WVMA", "VSUMP", "VSUMN",
    "VSUMD",
]

#: The 158 feature columns, in qlib ``get_feature_config`` output order.
FEATURE_COLS: List[str] = (
    ["KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2"]
    + ["OPEN0", "HIGH0", "LOW0", "VWAP0"]
    + [f"{op}{d}" for op in _ROLLING_OPS for d in _WINDOWS]
)

_EPS = 1e-12


def _rolling_slope(x: np.ndarray) -> float:
    """Cython Slope port: OLS slope of y over window positions 1..N (NaN dropped)."""
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return np.nan
    xs = np.arange(1.0, n + 1.0)
    s_x, s_y = xs.sum(), x.sum()
    s_xy = (xs * x).sum()
    s_x2 = (xs * xs).sum()
    denom = n * s_x2 - s_x * s_x
    return (n * s_xy - s_x * s_y) / denom if denom != 0 else np.nan


def _rolling_resi(x: np.ndarray) -> float:
    """Cython Resi port: y_newest − fitted(newest); newest always has x = N."""
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return np.nan
    xs = np.arange(1.0, n + 1.0)
    s_x, s_y = xs.sum(), x.sum()
    s_xy = (xs * x).sum()
    s_x2 = (xs * xs).sum()
    denom = n * s_x2 - s_x * s_x
    slope = (n * s_xy - s_x * s_y) / denom if denom != 0 else np.nan
    if np.isnan(slope):
        return np.nan
    intercept = s_y / n - slope * (s_x / n)
    return float(x[-1] - (slope * n + intercept))


def _rolling_rsquare(x: np.ndarray) -> float:
    """Cython Rsquare port: r² of the window regression (NaN dropped)."""
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return np.nan
    xs = np.arange(1.0, n + 1.0)
    s_x, s_y = xs.sum(), x.sum()
    s_xy = (xs * x).sum()
    s_x2 = (xs * xs).sum()
    s_y2 = (x * x).sum()
    denom_x = n * s_x2 - s_x * s_x
    denom_y = n * s_y2 - s_y * s_y
    if denom_x <= 0 or denom_y <= 0:
        return np.nan
    r = (n * s_xy - s_x * s_y) / np.sqrt(denom_x * denom_y)
    return float(r * r)


def _near_zero_std_mask(s: pd.Series, d: int) -> pd.Series:
    """True where the rolling std is within atol 2e-05 of zero (qlib masks)."""
    return np.isclose(s.rolling(d, min_periods=1).std(), 0, atol=2e-05)


def alpha158_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute the 158 Alpha158 features from one symbol's OHLCV bars.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Date-indexed with columns ``open, high, low, close, volume`` (missing
        calendar days as NaN rows are fine — all features are NaN-safe).

    Returns
    -------
    pd.DataFrame of shape (T, 158) with ``FEATURE_COLS`` columns; raw NaN
    preserved.
    """
    c, o, h, l = ohlcv["close"], ohlcv["open"], ohlcv["high"], ohlcv["low"]
    v = ohlcv["volume"]

    d1 = c.diff()                                   # close - Ref(close, 1)
    up = np.maximum(d1, 0.0)                        # Greater(d1, 0)
    dn = np.maximum(-d1, 0.0)                       # Greater(Ref(close,1)-close, 0)
    ab = d1.abs()                                   # Abs(d1)
    chg = c / c.shift(1)                            # close/Ref(close,1) ratio
    lr = np.log(v + 1)                              # Log(volume+1)
    vchg = v / v.shift(1)                           # volume/Ref(volume,1)
    logvr = np.log(vchg + 1)                        # Log(volume/Ref(volume,1)+1)
    wv = (chg - 1.0).abs() * v                      # Abs(close/Ref(close,1)-1)*volume
    vd = v.diff()                                   # volume - Ref(volume,1)
    vup = np.maximum(vd, 0.0)
    vdn = np.maximum(-vd, 0.0)
    vab = vd.abs()

    cols = {}

    # ---- kbar (9) ---------------------------------------------------------
    cols["KMID"] = (c - o) / o
    cols["KLEN"] = (h - l) / o
    cols["KMID2"] = (c - o) / (h - l + _EPS)
    cols["KUP"] = (h - np.maximum(o, c)) / o        # Greater(open, close)
    cols["KUP2"] = (h - np.maximum(o, c)) / (h - l + _EPS)
    cols["KLOW"] = (np.minimum(o, c) - l) / o       # Less(open, close)
    cols["KLOW2"] = (np.minimum(o, c) - l) / (h - l + _EPS)
    cols["KSFT"] = (2 * c - h - l) / o
    cols["KSFT2"] = (2 * c - h - l) / (h - l + _EPS)

    # ---- price (4) --------------------------------------------------------
    cols["OPEN0"] = o / c
    cols["HIGH0"] = h / c
    cols["LOW0"] = l / c
    cols["VWAP0"] = 1.0                             # vwap := close

    # ---- rolling (29 ops × 5 windows) -------------------------------------
    for d in _WINDOWS:
        r = lambda col, win=d: col.rolling(win, min_periods=1)  # noqa: E731

        cols[f"ROC{d}"] = c.shift(d) / c
        cols[f"MA{d}"] = r(c).mean() / c
        cols[f"STD{d}"] = r(c).std() / c
        cols[f"BETA{d}"] = r(c).apply(_rolling_slope, raw=True) / c
        rsqr = r(c).apply(_rolling_rsquare, raw=True)
        cols[f"RSQR{d}"] = rsqr.mask(_near_zero_std_mask(c, d))
        cols[f"RESI{d}"] = r(c).apply(_rolling_resi, raw=True) / c
        cols[f"MAX{d}"] = r(h).max() / c
        cols[f"MIN{d}"] = r(l).min() / c
        cols[f"QTLU{d}"] = r(c).quantile(0.8) / c
        cols[f"QTLD{d}"] = r(c).quantile(0.2) / c
        cols[f"RANK{d}"] = r(c).rank(pct=True)
        cols[f"RSV{d}"] = (c - r(l).min()) / (r(h).max() - r(l).min() + _EPS)

        idx_max = r(h).apply(lambda x: x.argmax() + 1, raw=True)
        idx_min = r(l).apply(lambda x: x.argmin() + 1, raw=True)
        cols[f"IMAX{d}"] = idx_max / d
        cols[f"IMIN{d}"] = idx_min / d
        cols[f"IMXD{d}"] = (idx_max - idx_min) / d

        corr = r(c).corr(lr)
        corr = corr.mask(_near_zero_std_mask(c, d) | _near_zero_std_mask(lr, d))
        cols[f"CORR{d}"] = corr
        cord = r(chg).corr(logvr)
        cord = cord.mask(_near_zero_std_mask(chg, d) | _near_zero_std_mask(logvr, d))
        cols[f"CORD{d}"] = cord

        g = c > c.shift(1)                          # Gt: NaN → False
        ls = c < c.shift(1)                         # Lt
        cntp = r(g).mean()
        cntn = r(ls).mean()
        cols[f"CNTP{d}"] = cntp
        cols[f"CNTN{d}"] = cntn
        cols[f"CNTD{d}"] = cntp - cntn

        abs_sum = r(ab).sum() + _EPS
        cols[f"SUMP{d}"] = r(up).sum() / abs_sum
        cols[f"SUMN{d}"] = r(dn).sum() / abs_sum
        cols[f"SUMD{d}"] = (r(up).sum() - r(dn).sum()) / abs_sum

        cols[f"VMA{d}"] = r(v).mean() / (v + _EPS)
        cols[f"VSTD{d}"] = r(v).std() / (v + _EPS)
        cols[f"WVMA{d}"] = r(wv).std() / (r(wv).mean() + _EPS)

        vabs_sum = r(vab).sum() + _EPS
        cols[f"VSUMP{d}"] = r(vup).sum() / vabs_sum
        cols[f"VSUMN{d}"] = r(vdn).sum() / vabs_sum
        cols[f"VSUMD{d}"] = (r(vup).sum() - r(vdn).sum()) / vabs_sum

    return pd.DataFrame(cols, index=ohlcv.index)[FEATURE_COLS]


if __name__ == "__main__":
    # smoke: 60 bars of fake OHLCV → correct shape, no exceptions, first rows NaN
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    rng = np.random.default_rng(0)
    px = 100 * np.cumprod(1 + rng.normal(0, 0.02, 60))
    fake = pd.DataFrame({
        "open": px * 0.999, "high": px * 1.01, "low": px * 0.99,
        "close": px, "volume": rng.integers(1e4, 1e6, 60).astype(float),
    }, index=idx)
    f = alpha158_features(fake)
    assert f.shape == (60, 158)
    assert list(f.columns) == FEATURE_COLS
    assert f.iloc[0]["VWAP0"] == 1.0
    assert not np.isnan(f.iloc[0]["MA5"])  # min_periods=1 → partial window
    print(f"OK: {f.shape}, VWAP0={f.iloc[0]['VWAP0']}, MA5[0]={f.iloc[0]['MA5']:.4f}")
