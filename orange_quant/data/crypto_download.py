"""
Hyperliquid 加密货币数据下载

从 Hyperliquid 公开 API 获取所有永续合约日线数据，
自动过滤 k 前缀微盘币、数据不足 365 天的币种，
输出为 qlib 兼容的 CSV 格式。
"""

import time
import requests
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional, Set

# Hyperliquid Info API
_HL_API = "https://api.hyperliquid.xyz/info"

# 过滤规则
_EXCLUDE_PREFIXES = ("k",)           # k 前缀 = 1000x 计价微盘币
_MIN_DAYS = 365                      # 最少需要 365 天数据
_REQUEST_DELAY = 0.3                 # 请求间隔（秒）


def _ms_to_date(ms: int) -> str:
    """毫秒时间戳 → YYYY-MM-DD"""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _date_to_ms(date_str: str) -> int:
    """YYYY-MM-DD → 毫秒时间戳"""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_all_markets() -> List[str]:
    """
    从 Hyperliquid meta API 获取所有永续合约币种列表。

    Returns
    -------
    list[str]
        币种代码列表（如 "BTC", "ETH"）。
    """
    resp = requests.post(_HL_API, json={"type": "meta"}, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    universe = data.get("universe", [])
    return [item["name"] for item in universe]


def filter_coins(coins: List[str]) -> List[str]:
    """
    过滤币种列表：
    - 排除 k 前缀微盘币
    - 排除 USDC/USDT 等稳定币

    Parameters
    ----------
    coins : list[str]
        原始币种列表。

    Returns
    -------
    list[str]
        过滤后的币种列表。
    """
    skip = {"USDC", "USDT", "DAI", "USTC", "TUSD", "BUSD", "USD"}
    filtered = []
    for coin in coins:
        if coin in skip:
            continue
        if coin.startswith(_EXCLUDE_PREFIXES):
            continue
        filtered.append(coin)
    return filtered


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
        币种代码。
    interval : str
        K 线周期: "1m", "5m", "15m", "1h", "4h", "1d", "1w"。
    start_time_ms : int or None
        起始时间（毫秒）。
    end_time_ms : int or None
        结束时间（毫秒）。

    Returns
    -------
    list[dict]
        蜡烛数据列表。
    """
    if end_time_ms is None:
        end_time_ms = int(time.time() * 1000)
    if start_time_ms is None:
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


def candles_to_csv(candles: List[dict], coin: str) -> str:
    """将 K 线数据转为 qlib 兼容的 CSV 字符串。"""
    lines = ["date,open,close,high,low,volume,factor"]
    for c in candles:
        date = _ms_to_date(c["t"])
        o = float(c["o"])
        h = float(c["h"])
        l = float(c["l"])
        c_price = float(c["c"])
        v = float(c["v"])
        lines.append(f"{date},{o},{c_price},{h},{l},{v},1.0")
    return "\n".join(lines)


def download_crypto_data(
    output_dir: str = "data/crypto_raw",
    coins: Optional[List[str]] = None,
    start_date: str = "2023-06-01",
    min_days: int = _MIN_DAYS,
) -> dict:
    """
    下载加密货币日线数据并保存为 CSV。

    如果未指定 coins，自动从 Hyperliquid 获取所有永续合约币种，
    自动过滤 k 前缀微盘币，仅保留数据 >= min_days 天的币种。

    Parameters
    ----------
    output_dir : str
        CSV 文件保存目录。
    coins : list[str] or None
        要下载的币种列表。None = 自动获取全部。
    start_date : str
        起始日期 YYYY-MM-DD。
    min_days : int
        最少需要的天数，不足的币种会被跳过。

    Returns
    -------
    dict
        {"downloaded": [成功的币种], "skipped": [跳过原因], "total_candles": int}
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Step 1: 获取并过滤币种列表
    if coins is None:
        all_coins = fetch_all_markets()
        coins = filter_coins(all_coins)
        print(f"[crypto] Hyperliquid 共 {len(all_coins)} 个永续合约")
        print(f"[crypto] 过滤后 {len(coins)} 个候选币种")
    else:
        print(f"[crypto] 指定 {len(coins)} 个币种")

    start_ms = _date_to_ms(start_date)
    end_ms = int(time.time() * 1000)

    downloaded = []
    skipped = {}
    total_candles = 0

    for i, coin in enumerate(coins):
        csv_file = out_path / f"{coin}.csv"

        if csv_file.exists():
            # 检查已有文件的行数
            existing_lines = csv_file.read_text().strip().split("\n")
            if len(existing_lines) - 1 >= min_days:  # 减表头
                print(f"[crypto] [{i+1}/{len(coins)}] {coin}.csv 已存在 ({len(existing_lines)-1} 天)，跳过")
                downloaded.append(coin)
                total_candles += len(existing_lines) - 1
                continue

        print(f"[crypto] [{i+1}/{len(coins)}] 下载 {coin} ({start_date} ~ {_ms_to_date(end_ms)})...", end=" ", flush=True)

        try:
            candles = fetch_candles(coin, "1d", start_ms, end_ms)
        except Exception as e:
            print(f"❌ API 错误: {e}")
            skipped[coin] = str(e)
            time.sleep(_REQUEST_DELAY)
            continue

        if not candles or len(candles) < min_days:
            print(f"⏭ 仅 {len(candles)} 天 (<{min_days})，跳过")
            skipped[coin] = f"only {len(candles)} days"
            time.sleep(_REQUEST_DELAY)
            continue

        csv_content = candles_to_csv(candles, coin)
        csv_file.write_text(csv_content)
        print(f"✅ {len(candles)} 天")
        downloaded.append(coin)
        total_candles += len(candles)
        time.sleep(_REQUEST_DELAY)

    print(f"\n[crypto] 完成！成功 {len(downloaded)} 个, 跳过 {len(skipped)} 个, 共 {total_candles} 条日线")
    if skipped:
        print(f"[crypto] 跳过的币种: {list(skipped.keys())[:20]}...")

    return {
        "downloaded": downloaded,
        "skipped": skipped,
        "total_candles": total_candles,
    }


if __name__ == "__main__":
    download_crypto_data()
