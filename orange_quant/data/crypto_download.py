"""
Hyperliquid 加密货币数据下载

从 Hyperliquid 公开 API 获取 BTC, ETH, XRP, BNB, SOL 的历史日线数据，
输出为 qlib 兼容的 CSV 格式。
"""

import os
import time
import requests
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional

# Hyperliquid Info API
_HL_API = "https://api.hyperliquid.xyz/info"

# 默认支持的币种
DEFAULT_COINS = ["BTC", "ETH", "XRP", "BNB", "SOL"]

# Hyperliquid 上线时间（各币种的最早可用数据）
# BTC perpetual started ~2023-03, others followed
COIN_START_TIMES = {
    "BTC": "2023-06-01",
    "ETH": "2023-06-01",
    "XRP": "2024-01-01",
    "BNB": "2024-01-01",
    "SOL": "2024-01-01",
}


def _ms_to_date(ms: int) -> str:
    """毫秒时间戳 → YYYY-MM-DD 字符串"""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _date_to_ms(date_str: str) -> int:
    """YYYY-MM-DD → 毫秒时间戳"""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_candles(
    coin: str,
    interval: str = "1d",
    start_time_ms: Optional[int] = None,
    end_time_ms: Optional[int] = None,
) -> List[dict]:
    """
    从 Hyperliquid 获取 K 线数据。

    Parameters
    ----------
    coin : str
        币种代码，如 "BTC", "ETH"。
    interval : str
        K 线周期: "1m", "5m", "15m", "1h", "4h", "1d", "1w"。
    start_time_ms : int or None
        起始时间（毫秒）。None 表示最早可用。
    end_time_ms : int or None
        结束时间（毫秒）。None 表示最新。

    Returns
    -------
    list[dict]
        蜡烛数据列表，每个元素含 t, T, o, h, l, c, v, s 字段。
    """
    if end_time_ms is None:
        end_time_ms = int(time.time() * 1000)
    if start_time_ms is None:
        # 默认从 2023 年开始
        start_time_ms = _date_to_ms("2023-01-01")

    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": coin,
            "interval": interval,
            "startTime": start_time_ms,
            "endTime": end_time_ms,
        },
    }

    resp = requests.post(_HL_API, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_candles_paginated(
    coin: str,
    interval: str = "1d",
    start_time_ms: Optional[int] = None,
    end_time_ms: Optional[int] = None,
) -> List[dict]:
    """
    分页获取 K 线（避免 5000 条限制）。

    日线数据通常不会触发 5000 条上限（5000天 ≈ 13年），
    但分钟数据可能超过，此函数自动处理分页。
    """
    if end_time_ms is None:
        end_time_ms = int(time.time() * 1000)
    if start_time_ms is None:
        start_time_ms = _date_to_ms("2023-01-01")

    all_candles = []
    cursor = start_time_ms

    while cursor < end_time_ms:
        candles = fetch_candles(coin, interval, cursor, end_time_ms)
        if not candles:
            break
        all_candles.extend(candles)
        # 推进游标到最后一个蜡烛之后
        cursor = candles[-1]["t"] + 1
        time.sleep(0.2)  # 速率限制

        # 防无限循环
        if len(candles) < 2:
            break

    return all_candles


def candles_to_csv(candles: List[dict], coin: str) -> str:
    """
    将 Hyperliquid K 线转为 qlib 兼容的 CSV 字符串。

    列: date, open, close, high, low, volume, factor

    factor 在加密货币中恒为 1.0（无拆股/分红调整）。
    """
    lines = ["date,open,close,high,low,volume,factor"]
    for c in candles:
        date = _ms_to_date(c["t"])
        o = float(c["o"])
        h = float(c["h"])
        l = float(c["l"])
        c_price = float(c["c"])
        v = float(c["v"])
        factor = 1.0
        lines.append(f"{date},{o},{c_price},{h},{l},{v},{factor}")
    return "\n".join(lines)


def download_crypto_data(
    output_dir: str = "data/crypto_raw",
    coins: List[str] = None,
    start_date: Optional[str] = None,
) -> str:
    """
    下载加密货币日线数据并保存为 CSV。

    Parameters
    ----------
    output_dir : str
        CSV 文件保存目录。
    coins : list[str] or None
        要下载的币种列表。None 使用默认: BTC, ETH, XRP, BNB, SOL。
    start_date : str or None
        起始日期 YYYY-MM-DD。None 使用各币种默认启动时间。

    Returns
    -------
    str
        CSV 文件所在目录路径。
    """
    if coins is None:
        coins = DEFAULT_COINS

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    end_ms = int(time.time() * 1000)

    for coin in coins:
        csv_file = out_path / f"{coin}.csv"
        if csv_file.exists():
            print(f"[crypto] {coin}.csv 已存在，跳过")
            continue

        start = start_date or COIN_START_TIMES.get(coin, "2023-06-01")
        start_ms = _date_to_ms(start)
        print(f"[crypto] 下载 {coin} 日线数据 ({start} ~ {_ms_to_date(end_ms)})...")

        candles = fetch_candles_paginated(coin, "1d", start_ms, end_ms)
        if not candles:
            print(f"[crypto] ⚠ {coin} 无数据返回，跳过")
            continue

        csv_content = candles_to_csv(candles, coin)
        csv_file.write_text(csv_content)
        print(f"[crypto] ✅ {coin}: {len(candles)} 条日线 → {csv_file}")

    return str(out_path)


if __name__ == "__main__":
    download_crypto_data()
