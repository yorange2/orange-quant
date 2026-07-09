#!/usr/bin/env python3
"""
构建 Hyperliquid 永续合约日线 qlib 数据集

1. 获取 Hyperliquid 成交量前 N 的永续合约
2. 增量下载日线（已有数据只补最新部分）
3. 转换为 qlib 二进制格式

Hyperliquid API:
    POST https://api.hyperliquid.xyz/info
    - {"type": "meta"}          → 所有永续合约元数据
    - {"type": "candleSnapshot", "req": {"coin": "BTC", "interval": "1d", ...}}

用法：
    python -m hyperliquid_lgb_momtopk.data.build          # 默认前50
    python -m hyperliquid_lgb_momtopk.data.build --top 100
    python -m hyperliquid_lgb_momtopk.data.build --force   # 强制全部重新下载
"""

import time
import argparse
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests

_HL_API = "https://api.hyperliquid.xyz/info"
RAW_DIR = Path("data/hyperliquid_raw")
QLIB_DIR = Path("data/qlib_data/hyperliquid")

_REQUEST_DELAY = 0.3


def load_coins() -> list:
    """从 qlib instruments 文件读取活跃币种列表"""
    inst_file = QLIB_DIR / "instruments" / "all.txt"
    if not inst_file.exists():
        if RAW_DIR.exists():
            return sorted([f.stem for f in RAW_DIR.glob("*.csv")])
        return []
    coins = []
    for line in inst_file.read_text().strip().splitlines():
        if "\t" in line:
            coins.append(line.split("\t")[0])
    return coins


def _date_to_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _ms_to_date(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def get_top_symbols(n: int = 50) -> list:
    """获取 Hyperliquid 成交量前 N 的永续合约"""
    resp = requests.post(_HL_API, json={"type": "meta"}, timeout=10)
    resp.raise_for_status()
    meta = resp.json()

    perps = []
    for p in meta["universe"]:
        name = p["name"]
        vol = float(p.get("dayNtlVlm", 0))
        perps.append((name, vol))

    perps.sort(key=lambda x: x[1], reverse=True)

    # 排除测试网代币
    skip_prefixes = ("TEST", "TOKEN")
    result = []
    for name, vol in perps:
        if any(name.startswith(p) for p in skip_prefixes):
            continue
        result.append((name, name))  # (symbol, base) — Hyperliquid 币名即 symbol
        if len(result) >= n:
            break
    return result


def fetch_daily(coin: str, start_ms: int, end_ms: int) -> list:
    """从 Hyperliquid API 获取日线蜡烛"""
    all_candles = []
    batch_start = start_ms

    while batch_start < end_ms:
        req = {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": "1d",
                "startTime": batch_start,
                "endTime": end_ms,
            },
        }
        try:
            resp = requests.post(_HL_API, json=req, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  API err: {e}")
            break

        if not data or not isinstance(data, list):
            break

        all_candles.extend(data)
        last_time = data[-1]["t"]
        if last_time <= batch_start:
            break
        batch_start = last_time + 86400000
        time.sleep(_REQUEST_DELAY)

    # 过滤掉当天未收盘的蜡烛（closeTime > 当前时间）
    now_ms = int(time.time() * 1000)
    all_candles = [c for c in all_candles if c["T"] <= now_ms]

    return all_candles


def candles_to_csv(candles: list, coin: str) -> str:
    """Hyperliquid 蜡烛 → qlib CSV"""
    lines = ["date,open,close,high,low,volume,factor"]
    for c in candles:
        date = _ms_to_date(c["t"])
        lines.append(f"{date},{c['o']},{c['c']},{c['h']},{c['l']},{c['v']},1.0")
    return "\n".join(lines)


def _rebuild_qlib():
    """从 raw CSV 重建 qlib 二进制，返回 (coins, sorted_dates)"""
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

    # 构建 features
    features_dir = QLIB_DIR / "features"
    date_to_idx = {d: i for i, d in enumerate(sorted_dates)}
    print(f"  构建 features (日历共 {len(sorted_dates)} 天)...")
    for coin in coins:
        df = pd.read_csv(RAW_DIR / f"{coin}.csv").set_index("date").sort_index()
        coin_dir = features_dir / coin
        coin_dir.mkdir(parents=True, exist_ok=True)
        start_idx = date_to_idx.get(df.index[0], 0)
        for field in ["open", "close", "high", "low", "volume", "factor"]:
            values = df[field].values.astype(np.float32)
            data = np.hstack([start_idx, values]).astype("<f")
            data.tofile(str(coin_dir / f"{field}.day.bin"))

    # VWAP 代理
    print("  生成 VWAP 代理字段 (vwap=close)...")
    if features_dir.exists():
        for coin in coins:
            close_bin = features_dir / coin / "close.day.bin"
            vwap_bin = features_dir / coin / "vwap.day.bin"
            if close_bin.exists() and not vwap_bin.exists():
                data = np.fromfile(close_bin, dtype="<f")
                data.tofile(str(vwap_bin))
        print(f"  VWAP 代理字段已为 {len(coins)} 个币种生成")

    return coins, sorted_dates


def main():
    parser = argparse.ArgumentParser(description="构建 Hyperliquid 永续合约日线数据集")
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--force", action="store_true", help="强制全量重新下载")
    args = parser.parse_args()

    print("=" * 60)
    print(f"📥 构建 Hyperliquid 永续合约日线数据集 (Top {args.top})")
    print("=" * 60)

    rebuild_data(top=args.top, start=args.start, force_download=args.force)


def rebuild_data(top: int = 50, start: str = "2020-01-01", force_download: bool = False):
    """增量下载数据并重建 qlib 二进制"""
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    end_ms = int(time.time() * 1000)
    start_ms = _date_to_ms(start)

    # Step 0: 获取币种列表
    pairs = get_top_symbols(top)
    print(f"\n[Step 0] Hyperliquid 永续合约 Top {top}:")
    for i, (sym, coin) in enumerate(pairs):
        print(f"  {i+1:3d}. {coin:15s}")

    # Step 1: 增量下载
    print(f"\n[Step 1/3] 下载日线 ({start} ~ {today_str})...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    total, new_total = 0, 0

    for sym, coin in pairs:
        csv_file = RAW_DIR / f"{coin}.csv"

        if csv_file.exists() and not force_download:
            existing = pd.read_csv(csv_file)
            last_date = existing["date"].iloc[-1]
            last_ms = _date_to_ms(last_date) + 86400000

            if last_ms >= end_ms - 86400000:
                print(f"  {coin:10s} 已是最新 ({len(existing)} 天, 截止 {last_date})，跳过")
                total += len(existing)
                continue

            print(f"  {coin:10s} 更新 {last_date} → {today_str} ...",
                  end=" ", flush=True)
            candles = fetch_daily(coin, last_ms, end_ms)
            if not candles:
                print(f"⚠ 无新数据")
                total += len(existing)
                continue

            new_csv = candles_to_csv(candles, coin)
            new_df = pd.read_csv(pd.io.common.StringIO(new_csv))
            combined = pd.concat([existing, new_df]).drop_duplicates(
                subset="date", keep="last"
            ).sort_values("date")
            combined.to_csv(csv_file, index=False)
            added = len(combined) - len(existing)
            print(f"✅ +{added} 天 (共 {len(combined)} 天)")
            total += len(combined)
            new_total += added
            time.sleep(_REQUEST_DELAY)
        else:
            if force_download and csv_file.exists():
                print(f"  {coin:10s} 强制重新下载...", end=" ", flush=True)
            else:
                print(f"  {coin:10s} 首次下载...", end=" ", flush=True)
            candles = fetch_daily(coin, start_ms, end_ms)
            if not candles:
                print("⚠ 无数据")
                continue

            csv_file.write_text(candles_to_csv(candles, coin))
            print(f"✅ {len(candles)} 天")
            total += len(candles)
            new_total += len(candles)
            time.sleep(_REQUEST_DELAY)

    print(f"\n  总计 {total} 条日线（本次新增 {new_total} 条）")

    # Step 2+3: 总是重建 qlib
    print("\n[Step 2/3] 重建 qlib 二进制...")
    coins, dates = _rebuild_qlib()
    if not coins:
        print("\n⚠ 无数据文件，跳过重建")
        return
    print(f"\n✅ 完成！{QLIB_DIR}")
    print(f"   币种: {len(coins)}, 时间: {dates[0]} ~ {dates[-1]}")


if __name__ == "__main__":
    main()
