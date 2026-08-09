"""Universe selection: freeze the top-N most liquid names before training.

Market-agnostic: both A-shares (cn) and crypto compute liquidity from the local
raw CSV library over [liquidity_start, freeze_date], so no look-ahead and no
extra API dependency. A-shares proxy liquidity by mean daily amount (元) when
available, else volume×close; crypto by mean quote-volume.

The optional ``membership: csi300`` flag intersects with the *current* CSI300
constituent snapshot via akshare (approximation — akshare has no historical
constituents API; documented survivorship caveat). Default is pure liquidity.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd


def _is_stock(symbol: str) -> bool:
    """A-share code segments: exclude indices (SH000/SZ399) and ETFs
    (SH5xx/SZ1xx). Crypto symbols (e.g. BTC) pass through."""
    if symbol.startswith(("SH000", "SZ399", "SH5", "SZ1")):
        return False
    return True


def _load_csv(symbol: str, raw_dir: Path) -> Optional[pd.DataFrame]:
    p = raw_dir / f"{symbol}.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, parse_dates=["date"])
    except Exception:  # noqa: BLE001 - unparseable file = missing
        return None
    if df.empty:
        return None
    return df


def freeze_universe(
    raw_dir: str,
    top_n: int,
    freeze_date: str,
    liquidity_start: str,
    membership: Optional[str] = None,
    min_history_days: int = 250,
) -> List[str]:
    """Top-N by mean daily liquidity in [liquidity_start, freeze_date].

    Liquidity proxy: mean(amount) if the CSV has an amount column, else
    mean(volume × close). Stocks/coins must have ≥ min_history_days of data in
    the window (survivorship of actively traded names only).
    """
    raw = Path(raw_dir)
    freeze = pd.Timestamp(freeze_date)
    lo = pd.Timestamp(liquidity_start)

    liq: dict[str, float] = {}
    for csv in sorted(raw.glob("*.csv")):
        sym = csv.stem
        if not _is_stock(sym):
            continue
        df = _load_csv(sym, raw)
        if df is None:
            continue
        if not {"open", "high", "low", "close", "volume"}.issubset(df.columns):
            continue
        w = df[(df["date"] >= lo) & (df["date"] <= freeze)]
        # min_history_days means calendar days — count unique dates so the
        # check works for both daily and hourly bars
        if w["date"].dt.date.nunique() < min_history_days:
            continue
        if "amount" in w.columns and w["amount"].notna().mean() > 0.5:
            lq = w["amount"].mean()
        else:
            lq = float((w["volume"] * w["close"]).mean())
        if lq and lq > 0:
            liq[sym] = float(lq)

    if membership:
        cons = _csi300_snapshot()
        if cons:
            liq = {s: v for s, v in liq.items() if s in cons}
            print(f"[universe] intersected with CSI300 snapshot: {len(liq)} names")

    ranked = sorted(liq, key=liq.get, reverse=True)[:top_n]
    print(f"[universe] frozen {len(ranked)} names at {freeze_date} "
          f"(liq window {lo.date()}~{freeze.date()})")
    return ranked


def _csi300_snapshot() -> Optional[set]:
    """Current CSI300 constituents via akshare (24h cached, best-effort)."""
    import time

    cache = Path("data/universe/csi300_cons.csv")
    if cache.exists() and (time.time() - cache.stat().st_mtime) < 86400:
        return set(pd.read_csv(cache, header=None)[0].astype(str).str.zfill(6))
    try:
        import akshare as ak

        df = ak.index_stock_cons_csindex("000300")
        codes = set(df["成分券代码"].astype(str).str.zfill(6))
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.Series(sorted(codes)).to_csv(cache, index=False, header=False)
        return {f"SH{c}" if c.startswith(("6", "9")) else f"SZ{c}" for c in codes}
    except Exception as e:  # noqa: BLE001 - optional feature
        print(f"[universe] akshare CSI300 snapshot failed, skip membership: {e}")
        return None
