#!/usr/bin/env python3
"""
Build the Hyperliquid spot daily-bar qlib dataset

1. Fetch the top-N tokens by spot volume on Hyperliquid
2. Incrementally download daily bars (existing data only gets the latest portion appended)
3. Convert to qlib binary format

Usage:
    python -m hyperliquid_lgb_momtopk.data.build          # top 50 by default
    python -m hyperliquid_lgb_momtopk.data.build --top 100
    python -m hyperliquid_lgb_momtopk.data.build --force   # force a full re-download
"""

import time
import argparse
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

RAW_DIR = Path("data/hyperliquid_raw")
QLIB_DIR = Path("data/qlib_data/hyperliquid")

_REQUEST_DELAY = 0.3

# Stablecoins / fiat-pegged bases to exclude from the universe
_SKIP_BASES = {"USDT", "USDE", "USDH", "USDHL", "FEUSD", "USR", "DAI", "BUIDL", "USDXL"}

_exchange = None


def _get_exchange():
    """Lazily create a shared public ccxt.hyperliquid client"""
    global _exchange
    if _exchange is None:
        import ccxt
        _exchange = ccxt.hyperliquid({"enableRateLimit": True, "timeout": 30000})
        _exchange.load_markets()
    return _exchange


def load_coins() -> list:
    """Active coin list: qlib instruments, falling back to raw CSVs,
    then live top-volume pairs, then a static default."""
    inst_file = QLIB_DIR / "instruments" / "all.txt"
    if inst_file.exists():
        coins = [line.split("\t")[0]
                 for line in inst_file.read_text().strip().splitlines()
                 if "\t" in line]
        if coins:
            return coins
    if RAW_DIR.exists():
        coins = sorted(f.stem for f in RAW_DIR.glob("*.csv"))
        if coins:
            return coins
    try:
        return [coin for _, coin in get_top_symbols(20)]
    except Exception as e:
        print(f"[data] ⚠ Failed to fetch top spot pairs ({e}), using static fallback")
        return ["HYPE", "PURR", "BTC", "ETH", "SOL"]


def _date_to_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _ms_to_date(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def get_top_symbols(n: int = 50) -> list:
    """Get the top-N USDC spot pairs by volume on Hyperliquid.

    Returns [(symbol, coin)] where symbol is a ccxt symbol like "HYPE/USDC"
    and coin is the ccxt base code (UBTC -> BTC, UETH -> ETH, USOL -> SOL).
    """
    ex = _get_exchange()
    tickers = ex.fetch_tickers()

    ranked = []
    for sym, t in tickers.items():
        market = ex.markets.get(sym)
        if not market or not market.get("spot") or market.get("quote") != "USDC":
            continue
        base = market["base"]
        if base in _SKIP_BASES:
            continue
        vol = t.get("quoteVolume")
        if vol is None:
            vol = float(t.get("info", {}).get("dayNtlVlm", 0) or 0)
        ranked.append((sym, base, float(vol)))

    ranked.sort(key=lambda x: x[2], reverse=True)
    return [(sym, base) for sym, base, _ in ranked[:n]]


def fetch_daily(symbol: str, start_ms: int, end_ms: int) -> list:
    """Fetch daily spot candles via ccxt (auto-paginated).

    Returns rows of [timestamp_ms, open, high, low, close, volume].
    """
    ex = _get_exchange()
    all_rows = []
    batch_start = start_ms

    while batch_start < end_ms:
        try:
            rows = ex.fetch_ohlcv(symbol, "1d", since=batch_start, limit=1000)
        except Exception as e:
            print(f"  API err: {e}")
            break

        if not rows:
            break

        all_rows.extend(rows)
        last_time = rows[-1][0]
        if last_time <= batch_start:
            break
        batch_start = last_time + 86400000
        time.sleep(_REQUEST_DELAY)

    # Filter out candles for the current day that haven't closed yet
    now_ms = int(time.time() * 1000)
    return [r for r in all_rows if r[0] + 86400000 <= now_ms]


def candles_to_csv(candles: list, coin: str) -> str:
    """ccxt OHLCV rows -> qlib CSV"""
    lines = ["date,open,close,high,low,volume,factor"]
    for ts, o, h, l, c, v in candles:
        date = _ms_to_date(ts)
        lines.append(f"{date},{o},{c},{h},{l},{v},1.0")
    return "\n".join(lines)


def _rebuild_qlib():
    """Rebuild qlib binaries from the raw CSVs, returns (coins, sorted_dates)"""
    import numpy as np
    QLIB_DIR.mkdir(parents=True, exist_ok=True)

    coins = sorted([f.stem for f in RAW_DIR.glob("*.csv")])
    all_dates = set()
    inst_lines = []

    for coin in coins:
        df = pd.read_csv(RAW_DIR / f"{coin}.csv")
        all_dates.update(df["date"].tolist())
        inst_lines.append(f"{coin}\t{df['date'].min()}\t{df['date'].max()}")

    sorted_dates = sorted(all_dates)

    (QLIB_DIR / "calendars").mkdir(parents=True, exist_ok=True)
    (QLIB_DIR / "calendars" / "day.txt").write_text("\n".join(sorted_dates))

    (QLIB_DIR / "instruments").mkdir(parents=True, exist_ok=True)
    (QLIB_DIR / "instruments" / "all.txt").write_text("\n".join(inst_lines))

    # Build features
    features_dir = QLIB_DIR / "features"
    date_to_idx = {d: i for i, d in enumerate(sorted_dates)}
    print(f"  Building features ({len(sorted_dates)} days in the calendar)...")
    for coin in coins:
        df = pd.read_csv(RAW_DIR / f"{coin}.csv").set_index("date").sort_index()
        # qlib reads features/{instrument.lower()}/, so the dir must be lowercase
        coin_dir = features_dir / coin.lower()
        coin_dir.mkdir(parents=True, exist_ok=True)
        start_idx = date_to_idx.get(df.index[0], 0)
        for field in ["open", "close", "high", "low", "volume", "factor"]:
            values = df[field].values.astype(np.float32)
            data = np.hstack([start_idx, values]).astype("<f")
            data.tofile(str(coin_dir / f"{field}.day.bin"))

    # VWAP proxy
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


def main():
    parser = argparse.ArgumentParser(description="Build the Hyperliquid spot daily-bar dataset")
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--force", action="store_true", help="Force a full re-download")
    args = parser.parse_args()

    print("=" * 60)
    print(f"📥 Building the Hyperliquid spot daily-bar dataset (Top {args.top})")
    print("=" * 60)

    rebuild_data(top=args.top, start=args.start, force_download=args.force)


def rebuild_data(top: int = 50, start: str = "2020-01-01", force_download: bool = False):
    """Incrementally download data and rebuild qlib binaries"""
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    end_ms = int(time.time() * 1000)
    start_ms = _date_to_ms(start)

    # Step 0: get the coin list
    pairs = get_top_symbols(top)
    print(f"\n[Step 0] Hyperliquid spot Top {top}:")
    for i, (sym, coin) in enumerate(pairs):
        print(f"  {i+1:3d}. {sym:18s} → {coin}")

    # Step 1: incremental download
    print(f"\n[Step 1/3] Downloading daily bars ({start} ~ {today_str})...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    total, new_total = 0, 0

    for sym, coin in pairs:
        csv_file = RAW_DIR / f"{coin}.csv"

        if csv_file.exists() and not force_download:
            existing = pd.read_csv(csv_file)
            last_date = existing["date"].iloc[-1]
            last_ms = _date_to_ms(last_date) + 86400000

            if last_ms >= end_ms - 86400000:
                print(f"  {coin:10s} already up to date ({len(existing)} days, through {last_date}), skipping")
                total += len(existing)
                continue

            print(f"  {coin:10s} updating {last_date} → {today_str} ...",
                  end=" ", flush=True)
            candles = fetch_daily(sym, last_ms, end_ms)
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
            candles = fetch_daily(sym, start_ms, end_ms)
            if not candles:
                print("⚠ no data")
                continue

            csv_file.write_text(candles_to_csv(candles, coin))
            print(f"✅ {len(candles)} days")
            total += len(candles)
            new_total += len(candles)
            time.sleep(_REQUEST_DELAY)

    print(f"\n  Total {total} daily bars ({new_total} newly added this run)")

    # Step 2+3: always rebuild qlib
    print("\n[Step 2/3] Rebuilding qlib binaries...")
    coins, dates = _rebuild_qlib()
    if not coins:
        print("\n⚠ No data files, skipping rebuild")
        return
    print(f"\n✅ Done! {QLIB_DIR}")
    print(f"   Coins: {len(coins)}, time range: {dates[0]} ~ {dates[-1]}")


if __name__ == "__main__":
    main()
