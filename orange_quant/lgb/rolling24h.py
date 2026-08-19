"""Live scorer for the rolling-24h design: one pooled model, 24 hourly views.

The research path (``scripts/run_hour_designs.py``) builds the blend from 24
cached anchor datasets — fine for a weekly A/B, useless in a container. This is
the deployable form: read each coin's hourly CSV once, slice it into the 24
clock-anchored daily series, score each with the same pooled model, then average
the cross-sectional z-scores.

Why the alignment falls out for free here: a Binance kline stamped ``T`` closes
at ``T+1h``, so the most recent *closed* bar at wall-clock ``now`` opened at
``(now − 1h).floor('h')``. Walking back 24 hours from there yields exactly one
bar per clock hour, spanning a contiguous trailing 24-hour window whose newest
edge is the latest closed bar. That IS the causal rule from the backtest
(``scores_ens_causal``: views at or before the target anchor's own cutoff) — no
explicit "same day vs previous day" branch is needed, because "the last 24
closed hourly bars" expresses it directly. ``scripts/test_causal_alignment.py``
pins the backtest side of that equivalence.

Cost note: the naive shape of this is 24 anchors × N coins CSV reads. Each
coin's hourly file is read **once** and sliced 24 ways instead.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from orange_quant.lgb.features import FEATURE_COLS, alpha158_features
from orange_quant.rl.dataset import bar_reader

N_VIEWS = 24
MIN_BARS = 60          # feature warmup floor, same as the single-view runner


def latest_closed_bar(now: pd.Timestamp) -> pd.Timestamp:
    """Open time of the most recent kline that has finished.

    A bar stamped ``T`` covers ``[T, T+1h)``, so it is closed iff ``T + 1h <=
    now``. At 09:00:00 sharp the 08:00 bar is the newest closed one.
    """
    return (now - pd.Timedelta(1, "h")).floor("h")


def view_timestamps(now: pd.Timestamp) -> List[pd.Timestamp]:
    """The 24 view cutoffs: last closed bar and the 23 hours before it.

    Returned oldest-first. Every clock hour appears exactly once, which is what
    makes these 24 distinct anchor views rather than 24 samples of one.
    """
    last = latest_closed_bar(now)
    return [last - pd.Timedelta(k, "h") for k in range(N_VIEWS - 1, -1, -1)]


class Rolling24hScorer:
    """Blend of 24 clock-anchored views scored by a single pooled model."""

    def __init__(self, cfg: dict, codes: List[str], model, lookback: int = 160) -> None:
        self.codes = codes
        self.model = model
        self.lookback = int(lookback)
        # hour_of_day must be OFF here: we want the whole hourly frame per coin
        # and do the slicing ourselves, one read instead of 24.
        self.raw_cfg = copy.deepcopy(cfg)
        self.raw_cfg.setdefault("data", {}).pop("hour_of_day", None)

    # ------------------------------------------------------------------ bars
    def _load_bars(self) -> Dict[str, pd.DataFrame]:
        read = bar_reader(self.raw_cfg)
        bars: Dict[str, pd.DataFrame] = {}
        for code in self.codes:
            b = read(code)
            if b is not None and not b.empty:
                bars[code] = b
        return bars

    # ---------------------------------------------------------------- views
    def _view_features(self, bars: Dict[str, pd.DataFrame], ts: pd.Timestamp,
                       max_lag_hours: int) -> tuple:
        """(X (N, 158), newest bar used) for the anchor series of ``ts.hour``.

        Two staleness traps, both silent, both closed here:

        * ``anchor.index <= ts`` falls back to whatever the newest bar is, so a
          stale feed yields a full, confident-looking matrix built from days-old
          data. The returned ``newest`` exposes that to the caller.
        * that fallback is **per coin**. A universe frozen in the past contains
          names since delisted from the venue; their CSVs still hold plenty of
          history, so they would score off week-old features and compete for
          top-k slots while the fleet-wide check — which looks at the freshest
          coin — stays green. Any coin whose own bar lags beyond tolerance is
          left NaN, which ranks it out rather than trading it blind.
        """
        X = np.full((len(self.codes), len(FEATURE_COLS)), np.nan, np.float32)
        tol = pd.Timedelta(max_lag_hours, "h")
        newest = None
        for j, code in enumerate(self.codes):
            b = bars.get(code)
            if b is None:
                continue
            anchor = b[b.index.hour == ts.hour]      # the clock-anchored daily series
            anchor = anchor[anchor.index <= ts]
            if len(anchor) < MIN_BARS:
                continue
            last = anchor.index[-1]
            if ts - last > tol:                      # this coin is dark for this view
                continue
            f = alpha158_features(anchor.iloc[-self.lookback:])
            X[j] = f[FEATURE_COLS].iloc[-1].to_numpy(np.float32)
            newest = last if newest is None else max(newest, last)
        return X, newest

    @staticmethod
    def _cs_zscore(p: np.ndarray) -> np.ndarray:
        """Cross-sectional z-score over the coins that scored (NaN-safe)."""
        out = np.full_like(p, np.nan, dtype=np.float64)
        m = np.isfinite(p)
        if m.sum() > 2:
            sd = p[m].std(ddof=1)
            if sd > 0:
                out[m] = (p[m] - p[m].mean()) / sd
        return out

    # ----------------------------------------------------------------- score
    def scores(self, now: Optional[pd.Timestamp] = None,
               max_lag_hours: int = 0) -> tuple:
        """Blended score per coin, plus a report describing what was **actually** used.

        Returns ``(scores (N,), report)``. Coins no view could score come back
        NaN — the caller ranks those last, as in the single-view path.

        ``report["stale"]`` is the one field a live caller must check. Each view
        asks for the bar at its own clock hour; if the feed has not caught up,
        pandas hands back an older bar instead of raising, so the blend would
        otherwise be built from days-old data while reporting a fresh window.
        ``max_lag_hours=0`` demands every view sit exactly on its requested bar.
        """
        now = pd.Timestamp.utcnow().tz_localize(None) if now is None else now
        stamps = view_timestamps(now)
        bars = self._load_bars()

        acc = np.zeros(len(self.codes))
        cnt = np.zeros(len(self.codes))
        used, actuals, lags = [], [], []
        for ts in stamps:
            X, newest = self._view_features(bars, ts, max_lag_hours)
            if newest is None or np.isfinite(X).sum() == 0:
                continue
            p = self.model.predict(X)
            p = np.where(np.isnan(X).all(axis=1), np.nan, p)
            z = self._cs_zscore(p)
            have = np.isfinite(z)
            if not have.any():
                continue
            acc += np.where(have, z, 0.0)
            cnt += have
            used.append(ts)
            actuals.append(newest)
            lags.append((ts - newest) / pd.Timedelta(1, "h"))

        out = np.where(cnt > 0, acc / np.where(cnt > 0, cnt, 1.0), np.nan)
        max_lag = max(lags) if lags else float("inf")
        stale = (len(used) != N_VIEWS) or (max_lag > max_lag_hours)
        report = {
            "views_used": len(used),
            "requested_window": [str(stamps[0]), str(stamps[-1])],
            "data_window": [str(min(actuals)), str(max(actuals))] if actuals else None,
            "max_lag_hours": None if not lags else round(max_lag, 1),
            "coins_scored": int((cnt > 0).sum()),
            "min_views_per_coin": int(cnt[cnt > 0].min()) if (cnt > 0).any() else 0,
            "bars_loaded": len(bars),
            "stale": bool(stale),
        }
        if stale:
            report["stale_reason"] = (
                f"only {len(used)}/{N_VIEWS} views scored" if len(used) != N_VIEWS
                else f"newest bar lags its view by {max_lag:.0f}h "
                     f"(allowed {max_lag_hours}h) — hourly feed not caught up")
        return out, report
