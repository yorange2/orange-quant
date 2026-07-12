#!/usr/bin/env python3
"""
Build the Binance spot daily-bar qlib dataset

1. Fetch the top-N USDT trading pairs by volume on Binance
2. Incrementally download daily bars (existing data only gets the latest portion appended)
3. Convert to qlib binary format

Usage:
    python -m biance_lgb_momtopk.data.build          # top 50 by default
    python -m biance_lgb_momtopk.data.build --top 100
    python -m biance_lgb_momtopk.data.build --force   # force a full re-download
"""

import time
import argparse
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests


_BINANCE_API = "https://api.binance.com/api/v3"
RAW_DIR = Path("data/binance_raw")
QLIB_DIR = Path("data/qlib_data/binance")


def load_coins() -> list:
    """Read the active coin list from the qlib instruments file"""
    inst_file = QLIB_DIR / "instruments" / "all.txt"
    if not inst_file.exists():
        # Fallback: read from the raw CSV directory
        if RAW_DIR.exists():
            return sorted([f.stem for f in RAW_DIR.glob("*.csv")])
        return []
    coins = []
    for line in inst_file.read_text().strip().splitlines():
        if "\t" in line:
            coins.append(line.split("\t")[0])
    return coins



_SKIP = {
    "USDCUSDT", "USDTUSDT", "TUSDUSDT", "BUSDUSDT", "DAIUSDT",
    "PAXUSDT", "USD1USDT", "FDUSDUSDT", "RLUSDUSDT", "EURUSDT",
    "XAUTUSDT", "PAXGUSDT",
    "UUSDT",  # trade-restricted on Binance (reduce-only), orders get rejected
}

_REQUEST_DELAY = 0.3


def _date_to_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _ms_to_date(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def get_top_symbols(n: int = 50) -> list:
    """Get the top-N USDT spot trading pairs by volume on Binance"""
    tickers = requests.get(f"{_BINANCE_API}/ticker/24hr", timeout=10).json()
    usdt = [(t["symbol"], float(t["quoteVolume"]))
            for t in tickers if t["symbol"].endswith("USDT")]
    usdt.sort(key=lambda x: x[1], reverse=True)

    result = []
    for symbol, vol in usdt:
        base = symbol.replace("USDT", "")
        if symbol in _SKIP:
            continue
        if any(x in base for x in ("UP", "DOWN", "BULL", "BEAR")):
            continue
        result.append((symbol, base))
        if len(result) >= n:
            break
    return result


def fetch_daily(symbol: str, start_ms: int, end_ms: int) -> list:
    """Fetch daily bars from the Binance API (auto-paginated)"""
    all_candles = []
    batch_start = start_ms
    while batch_start < end_ms:
        params = {"symbol": symbol, "interval": "1d",
                  "startTime": batch_start, "endTime": end_ms, "limit": 1000}
        try:
            resp = requests.get(f"{_BINANCE_API}/klines", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  API err: {e}")
            break
        if not data or not isinstance(data, list):
            break
        all_candles.extend(data)
        last_time = data[-1][0]
        if last_time <= batch_start:
            break
        batch_start = last_time + 86400000
        time.sleep(_REQUEST_DELAY)

    # Filter out candles for the current day that haven't closed yet (close_time > now)
    now_ms = int(time.time() * 1000)
    all_candles = [c for c in all_candles if c[6] <= now_ms]

    return all_candles


def candles_to_csv(candles: list, base: str) -> str:
    """Binance candles -> qlib CSV"""
    lines = ["date,open,close,high,low,volume,factor"]
    for c in candles:
        date = _ms_to_date(c[0])
        lines.append(f"{date},{c[1]},{c[4]},{c[2]},{c[3]},{c[5]},1.0")
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

    # Manually build features
    # qlib binary format: {field}.{freq}.bin, first 4 bytes are start_index, followed by float32 values
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

    # VWAP proxy: Binance has no VWAP data, use close instead (Alpha158 requires this field)
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
    parser = argparse.ArgumentParser(description="Build the Binance spot daily-bar dataset")
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--force", action="store_true", help="Force a full re-download")
    args = parser.parse_args()

    print("=" * 60)
    print(f"📥 Building the Binance spot daily-bar dataset (Top {args.top})")
    print("=" * 60)

    rebuild_data(top=args.top, start=args.start, force_download=args.force)


def rebuild_data(top: int = 50, start: str = "2020-01-01", force_download: bool = False):
    """Incrementally download data and rebuild qlib binaries (can be called directly by execute.py)

    Parameters
    ----------
    top : int
        Top-N USDT trading pairs by volume on Binance.
    start : str
        Data start date.
    force_download : bool
        Whether to force a full re-download of the CSVs.
    """
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    end_ms = int(time.time() * 1000)
    start_ms = _date_to_ms(start)

    # Step 0: get the coin list
    pairs = get_top_symbols(top)
    print(f"\n[Step 0] Top {top} USDT spot pairs by volume on Binance:")
    for i, (sym, base) in enumerate(pairs):
        print(f"  {i+1:3d}. {sym:15s} → {base}")

    # Step 1: incremental download
    print(f"\n[Step 1/3] Downloading daily bars ({start} ~ {today_str})...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    total, new_total = 0, 0

    for sym, base in pairs:
        csv_file = RAW_DIR / f"{base}.csv"

        if csv_file.exists() and not force_download:
            existing = pd.read_csv(csv_file)
            last_date = existing["date"].iloc[-1]
            last_ms = _date_to_ms(last_date) + 86400000

            if last_ms >= end_ms - 86400000:
                print(f"  {base:10s} already up to date ({len(existing)} days, through {last_date}), skipping")
                total += len(existing)
                continue

            print(f"  {base:10s} ({sym}) updating {last_date} → {today_str} ...",
                  end=" ", flush=True)
            candles = fetch_daily(sym, last_ms, end_ms)
            if not candles:
                print(f"⚠ no new data")
                total += len(existing)
                continue

            new_csv = candles_to_csv(candles, base)
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
                print(f"  {base:10s} force re-downloading...", end=" ", flush=True)
            else:
                print(f"  {base:10s} ({sym}) downloading for the first time...", end=" ", flush=True)
            candles = fetch_daily(sym, start_ms, end_ms)
            if not candles:
                print("⚠ no data")
                continue

            csv_file.write_text(candles_to_csv(candles, base))
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
