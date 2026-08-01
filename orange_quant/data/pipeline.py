#!/usr/bin/env python3
"""
Shared qlib dataset build pipeline (exchange-agnostic).

Given a :class:`DataSource` (fetch hooks + paths supplied by an exchange adapter),
this incrementally downloads daily bars, writes qlib CSVs, and rebuilds the qlib
binary store. Only ``get_top_symbols`` and ``fetch_daily`` are exchange-specific;
everything below is identical across venues.

``fetch_daily`` must return closed daily bars as uniform rows
``[timestamp_ms, open, high, low, close, volume]`` so ``candles_to_csv`` can format
them the same way for every exchange.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone
from typing import Callable, List

import pandas as pd

_REQUEST_DELAY = 0.3


@dataclass
class DataSource:
    label: str                                   # e.g. "Binance" / "Hyperliquid"
    raw_dir: Path                                # raw CSV dir
    qlib_dir: Path                               # qlib binary dir
    get_top_symbols: Callable[[int], list]       # (n) -> [(symbol, coin)]
    fetch_daily: Callable[[str, int, int], list]  # (symbol, start_ms, end_ms) -> rows
    fallback_coins: List[str] = field(default_factory=list)


def _date_to_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _ms_to_date(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def candles_to_csv(candles: list, coin: str) -> str:
    """Uniform OHLCV rows [ts, o, h, l, c, v] -> qlib CSV."""
    lines = ["date,open,close,high,low,volume,factor"]
    for ts, o, h, l, c, v in candles:
        date = _ms_to_date(ts)
        lines.append(f"{date},{o},{c},{h},{l},{v},1.0")
    return "\n".join(lines)


def load_coins(source: DataSource) -> list:
    """Active coin list: qlib instruments, falling back to raw CSVs,
    then live top-volume pairs, then the static fallback."""
    inst_file = source.qlib_dir / "instruments" / "all.txt"
    if inst_file.exists():
        coins = [line.split("\t")[0]
                 for line in inst_file.read_text().strip().splitlines()
                 if "\t" in line]
        if coins:
            return coins
    if source.raw_dir.exists():
        coins = sorted(f.stem for f in source.raw_dir.glob("*.csv"))
        if coins:
            return coins
    try:
        return [coin for _, coin in source.get_top_symbols(20)]
    except Exception as e:
        print(f"[data] ⚠ Failed to fetch top pairs ({e}), using static fallback")
        return list(source.fallback_coins)


def _stable_liquid_coins(source: DataSource, min_avg_quote_vol: float,
                         min_history_days: int, lookback: int = 30) -> set:
    """Coins with enough history AND enough *sustained* liquidity, computed from
    the already-downloaded CSVs (no extra API calls).

    Uses the trailing ``lookback``-day average quote volume (volume * close)
    instead of a live 24h snapshot, so freshly-listed meme coins whose live
    volume spikes into the ranking for a day — but whose sustained liquidity is
    negligible — are excluded. ``min_history_days`` additionally drops coins too
    new to have reliable features.
    """
    keep = set()
    for f in source.raw_dir.glob("*.csv"):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if len(df) < min_history_days or "volume" not in df or "close" not in df:
            continue
        recent = df.tail(lookback)
        avg_qv = float((recent["volume"] * recent["close"]).mean())
        if avg_qv >= min_avg_quote_vol:
            keep.add(f.stem)
    return keep


def _rebuild_qlib(source: DataSource, keep=None):
    """Rebuild qlib binaries from the raw CSVs, returns (coins, sorted_dates).

    ``keep`` (optional): restrict the qlib universe to this set of coins. Raw
    CSVs for coins outside it stay on disk but are left out of instruments/
    features — used to drop coins that have fallen below the liquidity floor so
    they no longer pollute training / backtest / live selection. When None
    (default), every downloaded CSV is included (unchanged behaviour).
    """
    import numpy as np
    qlib_dir = source.qlib_dir
    qlib_dir.mkdir(parents=True, exist_ok=True)

    coins = sorted([f.stem for f in source.raw_dir.glob("*.csv")])
    if keep is not None:
        coins = [c for c in coins if c in keep]
    all_dates = set()
    inst_lines = []

    for coin in coins:
        df = pd.read_csv(source.raw_dir / f"{coin}.csv")
        all_dates.update(df["date"].tolist())
        inst_lines.append(f"{coin}\t{df['date'].min()}\t{df['date'].max()}")

    sorted_dates = sorted(all_dates)

    (qlib_dir / "calendars").mkdir(parents=True, exist_ok=True)
    (qlib_dir / "calendars" / "day.txt").write_text("\n".join(sorted_dates))

    (qlib_dir / "instruments").mkdir(parents=True, exist_ok=True)
    (qlib_dir / "instruments" / "all.txt").write_text("\n".join(inst_lines))

    features_dir = qlib_dir / "features"
    date_to_idx = {d: i for i, d in enumerate(sorted_dates)}
    print(f"  Building features ({len(sorted_dates)} days in the calendar)...")
    for coin in coins:
        df = pd.read_csv(source.raw_dir / f"{coin}.csv").set_index("date").sort_index()
        # qlib reads features/{instrument.lower()}/, so the dir must be lowercase
        coin_dir = features_dir / coin.lower()
        coin_dir.mkdir(parents=True, exist_ok=True)
        start_idx = date_to_idx.get(df.index[0], 0)
        for field_name in ["open", "close", "high", "low", "volume", "factor"]:
            values = df[field_name].values.astype(np.float32)
            data = np.hstack([start_idx, values]).astype("<f")
            data.tofile(str(coin_dir / f"{field_name}.day.bin"))

    # VWAP proxy (Alpha158 requires a vwap field; use close)
    print("  Generating VWAP proxy field (vwap=close)...")
    if features_dir.exists():
        for coin in coins:
            close_bin = features_dir / coin.lower() / "close.day.bin"
            vwap_bin = features_dir / coin.lower() / "vwap.day.bin"
            if close_bin.exists() and not vwap_bin.exists():
                data = np.fromfile(close_bin, dtype="<f")
                data.tofile(str(vwap_bin))
        print(f"  VWAP proxy field generated for {len(coins)} coins")

    return coins, sorted_dates


def rebuild_data(source: DataSource, top: int = 50, start: str = "2020-01-01",
                 force_download: bool = False, restrict_to_top: bool = False,
                 min_history_days: int = 0, min_avg_quote_vol: float = 0.0):
    """Incrementally download data and rebuild qlib binaries.

    ``restrict_to_top``: when True, the rebuilt qlib universe is limited to
    coins with ``>= min_history_days`` of history AND a trailing-30-day average
    quote volume ``>= min_avg_quote_vol`` (computed from the CSVs). This uses
    *sustained* liquidity, not a live 24h snapshot, so a meme coin whose volume
    spikes for a single day cannot enter the tradable universe. Coins that fell
    off keep their CSVs on disk but are excluded from instruments/features.
    When False (default), all downloaded CSVs are included — unchanged.
    """
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    end_ms = int(time.time() * 1000)
    start_ms = _date_to_ms(start)

    pairs = source.get_top_symbols(top)
    print(f"\n[Step 0] {source.label} spot Top {top}:")
    for i, (sym, coin) in enumerate(pairs):
        print(f"  {i+1:3d}. {sym:18s} → {coin}")

    print(f"\n[Step 1/3] Downloading daily bars ({start} ~ {today_str})...")
    source.raw_dir.mkdir(parents=True, exist_ok=True)
    total, new_total = 0, 0

    for sym, coin in pairs:
        csv_file = source.raw_dir / f"{coin}.csv"

        if csv_file.exists() and not force_download:
            existing = pd.read_csv(csv_file)
            last_date = existing["date"].iloc[-1]
            last_ms = _date_to_ms(last_date) + 86400000

            if last_ms >= end_ms - 86400000:
                print(f"  {coin:10s} already up to date ({len(existing)} days, through {last_date}), skipping")
                total += len(existing)
                continue

            print(f"  {coin:10s} updating {last_date} → {today_str} ...", end=" ", flush=True)
            candles = source.fetch_daily(sym, last_ms, end_ms)
            if not candles:
                print(f"⚠ no new data")
                total += len(existing)
                continue

            new_csv = candles_to_csv(candles, coin)
            new_df = pd.read_csv(pd.io.common.StringIO(new_csv))
            combined = pd.concat([existing, new_df]).drop_duplicates(
                subset="date", keep="last"
            ).sort_values("date")
            combined.to_csv(csv_file, index=False)
            added = len(combined) - len(existing)
            print(f"✅ +{added} days ({len(combined)} days total)")
            total += len(combined)
            new_total += added
            time.sleep(_REQUEST_DELAY)
        else:
            if force_download and csv_file.exists():
                print(f"  {coin:10s} force re-downloading...", end=" ", flush=True)
            else:
                print(f"  {coin:10s} ({sym}) downloading for the first time...", end=" ", flush=True)
            candles = source.fetch_daily(sym, start_ms, end_ms)
            if not candles:
                print("⚠ no data")
                continue

            csv_file.write_text(candles_to_csv(candles, coin))
            print(f"✅ {len(candles)} days")
            total += len(candles)
            new_total += len(candles)
            time.sleep(_REQUEST_DELAY)

    print(f"\n  Total {total} daily bars ({new_total} newly added this run)")

    print("\n[Step 2/3] Rebuilding qlib binaries...")
    if restrict_to_top:
        keep = _stable_liquid_coins(source, min_avg_quote_vol, min_history_days)
        print(f"  Restricting universe to {len(keep)} coins "
              f"(>= {min_history_days}d history AND >= ${min_avg_quote_vol:,.0f}/day "
              f"30d-avg quote volume): {sorted(keep)}")
    else:
        keep = None
    coins, dates = _rebuild_qlib(source, keep=keep)
    if not coins:
        print("\n⚠ No data files, skipping rebuild")
        return
    print(f"\n✅ Done! {source.qlib_dir}")
    print(f"   Coins: {len(coins)}, time range: {dates[0]} ~ {dates[-1]}")
