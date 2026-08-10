#!/usr/bin/env python3
"""
Shared daily-data build pipeline (exchange-agnostic).

Given a :class:`DataSource` (fetch hooks + paths supplied by a venue adapter),
this incrementally downloads daily bars into per-symbol CSVs. Only
``get_top_symbols`` and ``fetch_daily`` are exchange-specific; everything below
is identical across venues. No qlib involvement — the CSVs feed the RL dataset
(``orange_quant.rl.dataset``) directly.

``fetch_daily`` must return closed daily bars as uniform rows
``[timestamp_ms, open, high, low, close, volume]`` so ``candles_to_csv`` can
format them the same way for every exchange.
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
    raw_dir: Path                                # raw CSV dir (daily)
    get_top_symbols: Callable[[int], list]       # (n) -> [(symbol, coin)]
    fetch_daily: Callable[[str, int, int], list]  # (symbol, start_ms, end_ms) -> rows
    fetch_hourly: Callable[[str, int, int], list] = None  # optional 1h hook
    h1_raw_dir: Path = None                      # raw CSV dir (hourly, optional)
    fallback_coins: List[str] = field(default_factory=list)


def _date_to_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _ms_to_date(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def candles_to_csv(candles: list, freq: str = "1d") -> str:
    """Uniform OHLCV rows [ts, o, h, l, c, v] -> CSV (date,open,close,high,low,volume).

    Hourly bars must keep the hour in the timestamp — a date-only format would
    collapse 24 bars into one day (a real past bug)."""
    fmt = "%Y-%m-%d %H:%M:%S" if freq == "1h" else "%Y-%m-%d"
    lines = ["date,open,close,high,low,volume"]
    for ts, o, h, l, c, v in candles:
        lines.append(f"{datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime(fmt)},{o},{c},{h},{l},{v}")
    return "\n".join(lines)


def rebuild_data(source: DataSource, top: int = 50, start: str = "2020-01-01",
                 force_download: bool = False, freq: str = "1d"):
    """Incrementally download bars into raw CSVs.

    ``freq``: "1d" (daily, into ``raw_dir``) or "1h" (hourly, into
    ``h1_raw_dir``, via the venue's ``fetch_hourly`` hook).
    """
    hourly = freq == "1h"
    fetch = source.fetch_hourly if hourly else source.fetch_daily
    raw_dir = source.h1_raw_dir if hourly else source.raw_dir
    label = "hourly" if hourly else "daily"
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    end_ms = int(time.time() * 1000)
    start_ms = _date_to_ms(start)

    pairs = source.get_top_symbols(top)
    print(f"\n[Step 0] {source.label} spot Top {top}:")
    for i, (sym, coin) in enumerate(pairs):
        print(f"  {i+1:3d}. {sym:18s} → {coin}")

    print(f"\n[Step 1/3] Downloading {label} bars ({start} ~ {today_str})...")
    raw_dir.mkdir(parents=True, exist_ok=True)
    total, new_total = 0, 0

    for sym, coin in pairs:
        csv_file = raw_dir / f"{coin}.csv"

        if csv_file.exists() and not force_download:
            existing = pd.read_csv(csv_file)
            last_date = existing["date"].iloc[-1]
            step_ms = 3600000 if hourly else 86400000
            last_ms = _date_to_ms(last_date) + step_ms

            if last_ms >= end_ms - step_ms:
                print(f"  {coin:10s} already up to date ({len(existing)} days, through {last_date}), skipping")
                total += len(existing)
                continue

            print(f"  {coin:10s} updating {last_date} → {today_str} ...", end=" ", flush=True)
            candles = fetch(sym, last_ms, end_ms)
            if not candles:
                print(f"⚠ no new data")
                total += len(existing)
                continue

            new_csv = candles_to_csv(candles, freq)
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
            candles = fetch(sym, start_ms, end_ms)
            if not candles:
                print("⚠ no data")
                continue

            csv_file.write_text(candles_to_csv(candles, freq))
            print(f"✅ {len(candles)} days")
            total += len(candles)
            new_total += len(candles)
            time.sleep(_REQUEST_DELAY)

    print(f"\n  Total {total} daily bars ({new_total} newly added this run)")

    coins = sorted(f.stem for f in raw_dir.glob("*.csv"))
    dates = []
    for coin in coins:
        try:
            d = pd.read_csv(raw_dir / f"{coin}.csv", usecols=["date"])["date"]
            dates.append((d.min(), d.max()))
        except Exception:
            continue
    if not coins:
        print("\n⚠ No data files")
        return
    print(f"\n✅ Done! {len(coins)} symbols in {raw_dir}")
    if dates:
        print(f"   Range: {min(d[0] for d in dates)} ~ {max(d[1] for d in dates)}")
