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
    raw_dir: Path                                # raw CSV dir
    get_top_symbols: Callable[[int], list]       # (n) -> [(symbol, coin)]
    fetch_daily: Callable[[str, int, int], list]  # (symbol, start_ms, end_ms) -> rows
    fallback_coins: List[str] = field(default_factory=list)


def _date_to_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _ms_to_date(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def candles_to_csv(candles: list, coin: str) -> str:
    """Uniform OHLCV rows [ts, o, h, l, c, v] -> CSV (date,open,close,high,low,volume)."""
    lines = ["date,open,close,high,low,volume"]
    for ts, o, h, l, c, v in candles:
        date = _ms_to_date(ts)
        lines.append(f"{date},{o},{c},{h},{l},{v}")
    return "\n".join(lines)


def load_coins(source: DataSource) -> list:
    """Active coin list: raw CSVs first, then live top-volume pairs,
    then the static fallback."""
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


def rebuild_data(source: DataSource, top: int = 50, start: str = "2020-01-01",
                 force_download: bool = False, restrict_to_top: bool = False,
                 min_history_days: int = 0, min_avg_quote_vol: float = 0.0):
    """Incrementally download daily bars into raw CSVs.

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

    coins = sorted(f.stem for f in source.raw_dir.glob("*.csv"))
    dates = []
    for coin in coins:
        try:
            df = pd.read_csv(source.raw_dir / f"{coin}.csv")
            dates.append((df["date"].min(), df["date"].max()))
        except Exception:
            continue
    if not coins:
        print("\n⚠ No data files")
        return
    print(f"\n✅ Done! {len(coins)} symbols in {source.raw_dir}")
    if dates:
        print(f"   Range: {min(d[0] for d in dates)} ~ {max(d[1] for d in dates)}")
