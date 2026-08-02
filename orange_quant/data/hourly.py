#!/usr/bin/env python3
"""
Hourly-bar download + phase resampling (exchange-agnostic).

Every experiment here trains on daily bars cut at UTC 00:00, because that is the
only daily bar the exchanges serve. That cut is arbitrary: a "day" could just as
well run 06:00 -> 06:00. Re-cutting the *same* price series at each of the 24
possible offsets ("phases") yields 24 daily datasets that differ only in where
the day boundary falls. That is what makes them useful — if a strategy only
works on phase 0, its backtest is partly sampling luck rather than edge.

One 1h download per venue is enough; every phase is derived from it:

    download_hourly(source, coins)      -> data/{venue}_hourly/{COIN}.csv
    build_phase(source, phase)          -> data/qlib_data/{venue}_h{HH}/

``fetch_hourly`` is the only exchange-specific piece. It returns uniform rows
``[timestamp_ms, open, high, low, close, volume]`` — the same contract as
``pipeline.fetch_daily``.

History availability differs sharply by venue:
  * Binance paginates 1h bars back to listing (full 2020- history).
  * Hyperliquid only retains ~5000 hourly candles (~208 days) and returns
    nothing for earlier ``startTime``, so HL phases cannot cover the configured
    2022-2025 train split. Run the phase study on Binance.

Usage:
    python -m orange_quant.data.hourly download --venue binance --coins BTC,ETH
    python -m orange_quant.data.hourly verify   --venue binance
    python -m orange_quant.data.hourly build    --venue binance --phases 0,6,12,18
"""

import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd

from orange_quant.data import pipeline

_HOUR_MS = 3_600_000
_REQUEST_DELAY = 0.3

# Same field order as the daily CSVs written by pipeline.candles_to_csv
_HOURLY_COLUMNS = ["datetime", "open", "close", "high", "low", "volume"]


@dataclass
class HourlySource:
    """Per-venue hooks + paths for the hourly/phase pipeline."""

    label: str                                    # e.g. "Binance"
    hourly_dir: Path                              # raw 1h CSV dir
    daily_qlib_dir: Path                          # native store, defines the universe
    phase_raw_tmpl: str                           # "data/binance_phase{phase:02d}_raw"
    phase_qlib_tmpl: str                          # "data/qlib_data/binance_h{phase:02d}"
    fetch_hourly: Callable[[str, int, int], list]  # (symbol, start_ms, end_ms) -> rows
    resolve_symbols: Callable[[Sequence[str]], Dict[str, str]]  # coins -> {coin: symbol}
    daily_raw_dir: Optional[Path] = None          # native daily CSVs, for verify

    def phase_raw_dir(self, phase: int) -> Path:
        return Path(self.phase_raw_tmpl.format(phase=phase))

    def phase_qlib_dir(self, phase: int) -> Path:
        return Path(self.phase_qlib_tmpl.format(phase=phase))


# --------------------------------------------------------------------------
# 1h download
# --------------------------------------------------------------------------

def _date_to_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _rows_to_frame(rows: list) -> pd.DataFrame:
    """Uniform OHLCV rows [ts, o, h, l, c, v] -> hourly frame (UTC datetimes)."""
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[_HOURLY_COLUMNS]


def load_hourly(source: HourlySource, coin: str) -> Optional[pd.DataFrame]:
    """Read a coin's stored 1h bars, deduped and sorted. None if not downloaded."""
    path = source.hourly_dir / f"{coin}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["datetime"])
    if df.empty:
        return None
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")
    return (df.drop_duplicates(subset="datetime", keep="last")
              .sort_values("datetime")
              .reset_index(drop=True))


def universe_from_qlib(qlib_dir: Path) -> List[str]:
    """Coins in a qlib store's instruments/all.txt (the tradable universe)."""
    inst = qlib_dir / "instruments" / "all.txt"
    if not inst.exists():
        return []
    return [line.split("\t")[0]
            for line in inst.read_text().strip().splitlines() if "\t" in line]


def download_hourly(source: HourlySource, coins: Sequence[str],
                    start: str = "2020-01-01", force: bool = False) -> dict:
    """Incrementally download 1h bars for ``coins`` into ``source.hourly_dir``.

    Resumes from each CSV's last stored hour unless ``force``. Returns
    {coin: bar_count}; coins with no data are omitted.
    """
    source.hourly_dir.mkdir(parents=True, exist_ok=True)
    symbols = source.resolve_symbols(coins)
    end_ms = int(time.time() * 1000)
    start_ms = _date_to_ms(start)

    missing = [c for c in coins if c not in symbols]
    if missing:
        print(f"  ⚠ No exchange symbol for {len(missing)} coins, skipping: {missing}")

    counts = {}
    for i, coin in enumerate(coins):
        symbol = symbols.get(coin)
        if symbol is None:
            continue

        existing = None if force else load_hourly(source, coin)
        if existing is not None:
            last_ms = int(existing["datetime"].iloc[-1].timestamp() * 1000)
            if last_ms + 2 * _HOUR_MS >= end_ms:
                print(f"  [{i+1}/{len(coins)}] {coin:10s} up to date ({len(existing)} bars)")
                counts[coin] = len(existing)
                continue
            fetch_from = last_ms + _HOUR_MS
        else:
            fetch_from = start_ms

        print(f"  [{i+1}/{len(coins)}] {coin:10s} ({symbol}) fetching...", end=" ", flush=True)
        rows = source.fetch_hourly(symbol, fetch_from, end_ms)
        if not rows:
            print("⚠ no data")
            if existing is not None:
                counts[coin] = len(existing)
            continue

        new_df = _rows_to_frame(rows)
        if existing is not None:
            new_df = pd.concat([existing, new_df])
        new_df = (new_df.drop_duplicates(subset="datetime", keep="last")
                        .sort_values("datetime")
                        .reset_index(drop=True))
        new_df.to_csv(source.hourly_dir / f"{coin}.csv", index=False)

        added = len(new_df) - (len(existing) if existing is not None else 0)
        span = (f"{new_df['datetime'].iloc[0]:%Y-%m-%d} ~ "
                f"{new_df['datetime'].iloc[-1]:%Y-%m-%d}")
        print(f"✅ +{added} bars ({len(new_df)} total, {span})")
        counts[coin] = len(new_df)
        time.sleep(_REQUEST_DELAY)

    return counts


# --------------------------------------------------------------------------
# phase resampling
# --------------------------------------------------------------------------

def resample_phase(hourly: pd.DataFrame, phase: int, min_hours: int = 24) -> pd.DataFrame:
    """Re-cut 1h bars into daily bars whose day starts at ``phase``:00 UTC.

    The bar labelled date ``D`` spans ``[D phase:00, D+1 phase:00)``, so phase 0
    reproduces the exchange's native UTC daily bar (see ``verify_phase0``).

    Buckets with fewer than ``min_hours`` bars are dropped — an incomplete day
    would otherwise fabricate an OHLC that no phase actually traded. Since a
    bucket holds at most 24 hourly slots, the default of 24 keeps only fully
    covered days.

    Returns a frame in the daily-CSV schema: date, open, close, high, low,
    volume, factor.
    """
    if not 0 <= phase <= 23:
        raise ValueError(f"phase must be in 0..23, got {phase}")

    df = hourly.sort_values("datetime")
    # Shift back by the phase so ordinary day-flooring lands on the bucket start.
    bucket = (df["datetime"] - pd.Timedelta(hours=phase)).dt.floor("D")

    out = df.groupby(bucket).agg(
        open=("open", "first"),
        close=("close", "last"),
        high=("high", "max"),
        low=("low", "min"),
        volume=("volume", "sum"),
        hours=("close", "size"),
    )
    out = out[out["hours"] >= min_hours].drop(columns="hours")
    out.insert(0, "date", out.index.strftime("%Y-%m-%d"))
    out["factor"] = 1.0
    return out.reset_index(drop=True)


def _resample_to_csv(source: HourlySource, phase: int, coins: Sequence[str],
                     min_hours: int) -> set:
    """Write phase-``phase`` daily CSVs; returns the coins actually written."""
    raw_dir = source.phase_raw_dir(phase)
    raw_dir.mkdir(parents=True, exist_ok=True)

    written, skipped = set(), []
    for coin in coins:
        hourly = load_hourly(source, coin)
        if hourly is None or hourly.empty:
            skipped.append(coin)
            continue
        daily = resample_phase(hourly, phase, min_hours=min_hours)
        if daily.empty:
            skipped.append(coin)
            continue
        daily.to_csv(raw_dir / f"{coin}.csv", index=False)
        written.add(coin)

    print(f"[phase {phase:02d}] resampled {len(written)}/{len(coins)} coins -> {raw_dir}")
    if skipped:
        print(f"  ⚠ no usable hourly data for {len(skipped)}: {sorted(skipped)}")
    return written


def _rebuild_phase(source: HourlySource, phase: int, keep: set) -> dict:
    """Rebuild the qlib store for one phase from its CSVs, restricted to ``keep``."""
    qlib_dir = source.phase_qlib_dir(phase)
    shim = pipeline.DataSource(
        label=f"{source.label} h{phase:02d}",
        raw_dir=source.phase_raw_dir(phase),
        qlib_dir=qlib_dir,
        get_top_symbols=lambda n: [],
        fetch_daily=lambda *a: [],
    )
    built, dates = pipeline.rebuild_qlib(shim, keep=keep)
    print(f"  ✅ {qlib_dir} — {len(built)} coins, {dates[0]} ~ {dates[-1]}")
    return {"phase": phase, "coins": len(built), "start": dates[0], "end": dates[-1],
            "qlib_dir": str(qlib_dir)}


def build_phases(source: HourlySource, phases: Sequence[int],
                 coins: Optional[Sequence[str]] = None, min_hours: int = 24) -> List[dict]:
    """Build qlib stores for several phases sharing one identical universe.

    Resamples every phase first, then rebuilds them all against the *intersection*
    of coins that produced usable bars in every phase (further restricted to the
    native store's universe). A coin present in some phases but not others would
    otherwise make the phases trade different assets, and any performance gap
    between them could no longer be attributed to the day boundary.
    """
    native = set(universe_from_qlib(source.daily_qlib_dir))
    if coins is None:
        coins = sorted(native)

    per_phase = {p: _resample_to_csv(source, p, coins, min_hours) for p in phases}

    keep = set.intersection(*per_phase.values()) if per_phase else set()
    if native:
        keep &= native
    if not keep:
        print("⚠ No coin has usable bars in every phase; nothing to build")
        return []

    dropped = sorted(set(coins) - keep)
    print(f"\nShared universe across phases {list(phases)}: {len(keep)} coins")
    if dropped:
        print(f"  dropped (not usable in every phase / not in native store): {dropped}")

    return [_rebuild_phase(source, p, keep) for p in phases]


def build_phase(source: HourlySource, phase: int, coins: Optional[Sequence[str]] = None,
                min_hours: int = 24) -> dict:
    """Build a single phase's qlib store (universe = the native store's)."""
    results = build_phases(source, [phase], coins=coins, min_hours=min_hours)
    return results[0] if results else {"phase": phase, "coins": 0}


# --------------------------------------------------------------------------
# correctness check
# --------------------------------------------------------------------------

def verify_phase0(source: HourlySource, coins: Optional[Sequence[str]] = None,
                  tol: float = 1e-3) -> pd.DataFrame:
    """Check phase 0 against the exchange's native daily bars.

    Resampling 1h -> phase 0 must reproduce the UTC daily bar the venue serves.
    Any material mismatch means the bucketing is wrong, so this is the test that
    validates every other phase. Returns per-coin max relative error by field.

    Volume is compared too, but a small drift there is expected: summed hourly
    volume and the venue's daily aggregate are computed independently.
    """
    if source.daily_raw_dir is None:
        raise ValueError(f"{source.label} has no daily_raw_dir to verify against")
    if coins is None:
        coins = universe_from_qlib(source.daily_qlib_dir)

    rows = []
    for coin in coins:
        hourly = load_hourly(source, coin)
        daily_path = source.daily_raw_dir / f"{coin}.csv"
        if hourly is None or not daily_path.exists():
            continue

        got = resample_phase(hourly, 0).set_index("date")
        want = pd.read_csv(daily_path).set_index("date")
        shared = got.index.intersection(want.index)
        if len(shared) == 0:
            continue

        rec = {"coin": coin, "days": len(shared)}
        for field in ("open", "close", "high", "low", "volume"):
            a, b = got.loc[shared, field], want.loc[shared, field]
            denom = b.abs().replace(0, pd.NA)
            rec[field] = float(((a - b).abs() / denom).max(skipna=True))
        rows.append(rec)

    report = pd.DataFrame(rows)
    if report.empty:
        print("⚠ Nothing to verify — download 1h data first")
        return report

    ohlc = report[["open", "close", "high", "low"]].to_numpy().max()
    print(f"\nphase-0 vs native daily bars ({len(report)} coins, "
          f"{int(report['days'].sum())} coin-days)")
    print(f"  max OHLC relative error  : {ohlc:.3e}")
    print(f"  max volume relative error: {report['volume'].max():.3e}")
    bad = report[report[["open", "close", "high", "low"]].max(axis=1) > tol]
    if bad.empty:
        print(f"  ✅ every coin within tol={tol}")
    else:
        print(f"  ❌ {len(bad)} coins exceed tol={tol}:")
        print(bad.to_string(index=False))
    return report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def get_source(venue: str) -> HourlySource:
    if venue == "binance":
        from biance_lgb_momtopk.data import build
    elif venue == "hyperliquid":
        from hyperliquid_lgb_momtopk.data import build
    else:
        raise SystemExit(f"unknown venue: {venue}")
    return build.HOURLY_SOURCE


def main():
    parser = argparse.ArgumentParser(description="1h download + phase resampling")
    parser.add_argument("action", choices=["download", "verify", "build"])
    parser.add_argument("--venue", default="binance", choices=["binance", "hyperliquid"])
    parser.add_argument("--coins", type=str, default=None,
                        help="Comma-separated subset (default: the qlib universe)")
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--force", action="store_true", help="Ignore stored bars, refetch")
    parser.add_argument("--phases", type=str, default="0,6,12,18",
                        help="Comma-separated hours 0..23 (build only)")
    args = parser.parse_args()

    source = get_source(args.venue)
    coins = ([c.strip().upper() for c in args.coins.split(",") if c.strip()]
             if args.coins else universe_from_qlib(source.daily_qlib_dir))
    if not coins:
        raise SystemExit(f"No coins — build the daily dataset for {args.venue} first")

    print("=" * 60)
    print(f"⏱  {source.label} 1h / phase pipeline — {args.action} ({len(coins)} coins)")
    print("=" * 60)

    if args.action == "download":
        counts = download_hourly(source, coins, start=args.start, force=args.force)
        total = sum(counts.values())
        print(f"\n✅ {len(counts)} coins, {total:,} hourly bars in {source.hourly_dir}")
    elif args.action == "verify":
        verify_phase0(source, coins)
    else:
        phases = [int(p) for p in args.phases.split(",") if p.strip()]
        build_phases(source, phases, coins=coins)


if __name__ == "__main__":
    main()
