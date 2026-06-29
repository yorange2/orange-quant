#!/usr/bin/env python3
"""
构建加密货币 1小时 qlib 数据集（Binance 数据源）

1. 从 Binance 公开 API 下载 1h K线
2. 转换为 qlib 二进制格式
3. 创建日历和股票列表文件

用法：
    python scripts/build_crypto_data_1h.py
"""

import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_BINANCE_API = "https://api.binance.com/api/v3/klines"
RAW_DIR = Path("data/crypto_raw_1h")
QLIB_DIR = Path("data/qlib_data/crypto_1h")
QLIB_REPO = Path("/Users/yuanchengcheng/Documents/GitHub/qlib")

# 9 个 U现货币种 → Binance USDT 交易对
COIN_PAIRS = {
    "AVAX": "AVAXUSDT",
    "BTC": "BTCUSDT",
    "DOGE": "DOGEUSDT",
    "ENA": "ENAUSDT",
    "ETH": "ETHUSDT",
    "S": "SUSDT",
    "SOL": "SOLUSDT",
    "WLD": "WLDUSDT",
    "ZEC": "ZECUSDT",
}

START_DATE = "2023-06-01"
_REQUEST_DELAY = 0.25  # Binance  rate limit: ~1200/min, 0.25s is safe


def _date_to_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _fetch_binance(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    """
    从 Binance 获取 K线（自动分页，每批最多 1000 条）。

    Binance API 返回格式:
      [openTime, open, high, low, close, volume,
       closeTime, quoteVolume, trades, takerBuyBaseVol, takerBuyQuoteVol, ignore]
    """
    all_candles = []
    batch_start = start_ms

    while batch_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": batch_start,
            "endTime": end_ms,
            "limit": 1000,
        }
        try:
            resp = requests.get(_BINANCE_API, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  API err: {e}")
            break

        if not data or not isinstance(data, list):
            break

        all_candles.extend(data)

        # 下次从这批最后一条之后开始
        last_time = data[-1][0]
        if last_time <= batch_start:
            break
        batch_start = last_time + 1
        time.sleep(_REQUEST_DELAY)

    return all_candles


def _to_csv(candles: list, coin: str) -> str:
    """Binance K线 → qlib CSV"""
    lines = ["date,open,close,high,low,volume,factor"]
    for c in candles:
        dt = datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc)
        date = dt.strftime("%Y-%m-%d %H:%M:%S")
        o, h, l, cl, v = c[1], c[2], c[3], c[4], c[5]
        lines.append(f"{date},{o},{cl},{h},{l},{v},1.0")
    return "\n".join(lines)


def main():
    print("=" * 60)
    print("📥 构建加密货币 1小时 qlib 数据集 (Binance)")
    print("=" * 60)

    raw_dir = Path(RAW_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)

    end_ms = int(time.time() * 1000)
    start_ms = _date_to_ms(START_DATE)

    total = 0
    coin_list = list(COIN_PAIRS.keys())

    print(f"\n[Step 1/3] Binance 下载 {len(coin_list)} 个币种 1h 数据...")
    for coin in coin_list:
        csv_file = raw_dir / f"{coin}.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            print(f"[1h] {coin}.csv 已存在 ({len(df)} 条)，跳过")
            total += len(df)
            continue

        symbol = COIN_PAIRS[coin]
        print(f"[1h] {coin} ({symbol}) ...", end=" ", flush=True)
        candles = _fetch_binance(symbol, "1h", start_ms, end_ms)

        if not candles:
            print("⚠ 无数据")
            continue

        csv_content = _to_csv(candles, coin)
        csv_file.write_text(csv_content)
        print(f"✅ {len(candles)} 条")
        total += len(candles)
        time.sleep(_REQUEST_DELAY)

    print(f"\n[1h] 总计 {total} 条时线数据")

    # Step 2: dump_bin
    print("\n[Step 2/3] 转换为 qlib 二进制格式...")
    QLIB_DIR.mkdir(parents=True, exist_ok=True)

    dump_script = QLIB_REPO / "scripts" / "dump_bin.py"
    result = subprocess.run(
        [
            sys.executable, str(dump_script), "dump_all",
            "--data_path", str(RAW_DIR),
            "--qlib_dir", str(QLIB_DIR),
            "--include_fields", "open,close,high,low,volume,factor",
            "--date_field_name", "date",
            "--freq", "day",
        ],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        print(f"dump_bin 失败:\n{result.stderr[-500:]}")
        _build_manual()
        return
    print(result.stdout[-300:])

    # Step 3: calendar + instruments
    print("[Step 3/3] 创建日历和股票列表...")
    all_dates = set()
    for csv_file in raw_dir.glob("*.csv"):
        df = pd.read_csv(csv_file)
        all_dates.update(df["date"].tolist())

    sorted_dates = sorted(all_dates)
    (QLIB_DIR / "calendars").mkdir(parents=True, exist_ok=True)
    (QLIB_DIR / "calendars" / "day.txt").write_text("\n".join(sorted_dates))

    (QLIB_DIR / "instruments").mkdir(parents=True, exist_ok=True)
    coins = sorted([f.stem for f in raw_dir.glob("*.csv")])
    inst_lines = []
    for coin in coins:
        df = pd.read_csv(raw_dir / f"{coin}.csv")
        inst_lines.append(f"{coin}\t{df['date'].min()}\t{df['date'].max()}")
    (QLIB_DIR / "instruments" / "all.txt").write_text("\n".join(inst_lines))

    print(f"\n✅ 完成！{QLIB_DIR}")
    print(f"   币种: {len(coins)}, 总样本: {total}")
    print(f"   时间: {sorted_dates[0]} ~ {sorted_dates[-1]}")


def _build_manual():
    """手动构建（dump_bin 不可用时）"""
    import numpy as np
    features_dir = QLIB_DIR / "features"
    all_dates = set()
    for csv_file in RAW_DIR.glob("*.csv"):
        coin = csv_file.stem
        df = pd.read_csv(csv_file)
        df = df.set_index("date").sort_index()
        all_dates.update(df.index.tolist())
        coin_dir = features_dir / coin
        coin_dir.mkdir(parents=True, exist_ok=True)
        for field in ["open", "close", "high", "low", "volume", "factor"]:
            df[field].values.astype(np.float32).tofile(str(coin_dir / f"day.{field}.bin"))
    sorted_dates = sorted(all_dates)
    (QLIB_DIR / "calendars").mkdir(parents=True, exist_ok=True)
    (QLIB_DIR / "calendars" / "day.txt").write_text("\n".join(sorted_dates))
    (QLIB_DIR / "instruments").mkdir(parents=True, exist_ok=True)
    coins = sorted([f.stem for f in RAW_DIR.glob("*.csv")])
    inst_lines = []
    for coin in coins:
        df = pd.read_csv(RAW_DIR / f"{coin}.csv")
        inst_lines.append(f"{coin}\t{df['date'].min()}\t{df['date'].max()}")
    (QLIB_DIR / "instruments" / "all.txt").write_text("\n".join(inst_lines))
    print(f"   手动构建完成，{len(coins)} 币种，{len(sorted_dates)} 个时点")


if __name__ == "__main__":
    main()
