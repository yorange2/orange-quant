#!/usr/bin/env python3
"""
构建 Binance 现货日线 qlib 数据集

1. 获取 Binance 成交量前 N 的 USDT 交易对
2. 增量下载日线（已有数据只补最新部分）
3. 转换为 qlib 二进制格式

用法：
    python -m biance_lgb_momtopk.data.build          # 默认前50
    python -m biance_lgb_momtopk.data.build --top 100
    python -m biance_lgb_momtopk.data.build --force   # 强制全部重新下载
"""

import sys
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests


_BINANCE_API = "https://api.binance.com/api/v3"
RAW_DIR = Path("data/binance_raw")
QLIB_DIR = Path("data/qlib_data/binance")


def _find_dump_bin() -> Path | None:
    """动态查找 qlib 的 dump_bin.py 脚本路径"""
    # 方式一：从已安装的 qlib 包中查找
    try:
        import qlib
        qlib_root = Path(qlib.__file__).resolve().parent.parent
        candidate = qlib_root / "scripts" / "dump_bin.py"
        print(f"  [find_dump_bin] qlib 包路径: {qlib_root}")
        if candidate.exists():
            print(f"  [find_dump_bin] ✅ 找到 (via qlib.__file__): {candidate}")
            return candidate
        else:
            print(f"  [find_dump_bin] 未找到 scripts/dump_bin.py 在 qlib 包中")
    except (ImportError, AttributeError) as e:
        print(f"  [find_dump_bin] qlib 导入失败: {e}")

    # 方式二：遍历 sys.path
    print(f"  [find_dump_bin] 遍历 sys.path ({len(sys.path)} 个路径)...")
    for site in sys.path:
        p = Path(site) / "qlib" / "scripts" / "dump_bin.py"
        if p.exists():
            print(f"  [find_dump_bin] ✅ 找到 (via sys.path): {p}")
            return p

    # 方式三：pip show 查找
    print(f"  [find_dump_bin] 尝试 pip show qlib...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "qlib"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            print(f"  [find_dump_bin] pip show 失败: {result.stderr.strip()[:200]}")
        else:
            for line in result.stdout.splitlines():
                if line.startswith("Location:"):
                    loc = line.split(":", 1)[1].strip()
                    p = Path(loc) / "qlib" / "scripts" / "dump_bin.py"
                    print(f"  [find_dump_bin] pip location: {loc}")
                    if p.exists():
                        print(f"  [find_dump_bin] ✅ 找到 (via pip show): {p}")
                        return p
                    else:
                        print(f"  [find_dump_bin] 路径存在但无 scripts/dump_bin.py (可能是 PyPI pyqlib 包)")
    except Exception as e:
        print(f"  [find_dump_bin] pip show 异常: {e}")

    print(f"  [find_dump_bin] ❌ 未找到 dump_bin.py，将使用手动构建")
    return None

# 20 个蓝筹币种（市值大、历史长、流动性好）
BLUECHIPS = [
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX",
    "LINK", "DOT", "LTC", "UNI", "AAVE", "NEAR", "TRX", "FET",
    "INJ", "XLM", "BCH", "HBAR",
]

_SKIP = {
    "USDCUSDT", "USDTUSDT", "TUSDUSDT", "BUSDUSDT", "DAIUSDT",
    "PAXUSDT", "USD1USDT", "FDUSDUSDT", "RLUSDUSDT", "EURUSDT",
    "XAUTUSDT", "PAXGUSDT",
}
_REQUEST_DELAY = 0.3


def _date_to_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _ms_to_date(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def get_top_symbols(n: int = 50) -> list:
    """获取 Binance 成交量前 N 的 USDT 现货交易对"""
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
    """从 Binance API 获取日线（自动分页）"""
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
    return all_candles


def candles_to_csv(candles: list, base: str) -> str:
    """Binance K线 → qlib CSV"""
    lines = ["date,open,close,high,low,volume,factor"]
    for c in candles:
        date = _ms_to_date(c[0])
        lines.append(f"{date},{c[1]},{c[4]},{c[2]},{c[3]},{c[5]},1.0")
    return "\n".join(lines)


def _rebuild_qlib():
    """从 raw CSV 重建 qlib 二进制"""
    import numpy as np
    QLIB_DIR.mkdir(parents=True, exist_ok=True)

    # 日历和 instruments（无论用哪种方式构建都先生成）
    coins = sorted([f.stem for f in RAW_DIR.glob("*.csv")])
    all_dates = set()
    for coin in coins:
        df = pd.read_csv(RAW_DIR / f"{coin}.csv")
        all_dates.update(df["date"].tolist())
    sorted_dates = sorted(all_dates)

    (QLIB_DIR / "calendars").mkdir(parents=True, exist_ok=True)
    (QLIB_DIR / "calendars" / "day.txt").write_text("\n".join(sorted_dates))

    (QLIB_DIR / "instruments").mkdir(parents=True, exist_ok=True)
    inst_lines = []
    for coin in coins:
        df = pd.read_csv(RAW_DIR / f"{coin}.csv")
        inst_lines.append(f"{coin}\t{df['date'].min()}\t{df['date'].max()}")
    (QLIB_DIR / "instruments" / "all.txt").write_text("\n".join(inst_lines))

    blue_lines = []
    for coin in BLUECHIPS:
        csv_file = RAW_DIR / f"{coin}.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            blue_lines.append(f"{coin}\t{df['date'].min()}\t{df['date'].max()}")
    (QLIB_DIR / "instruments" / "bluechips.txt").write_text("\n".join(blue_lines))

    # dump_bin（qlib 脚本，生成 features）
    dump_script = _find_dump_bin()
    dump_bin_ok = False
    if dump_script:
        print(f"  使用 dump_bin: {dump_script}")
        result = subprocess.run(
            [sys.executable, str(dump_script), "dump_all",
             "--data_path", str(RAW_DIR), "--qlib_dir", str(QLIB_DIR),
             "--include_fields", "open,close,high,low,volume,factor",
             "--date_field_name", "date", "--freq", "day"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            print("  dump_bin 完成")
            dump_bin_ok = True
        else:
            print(f"  dump_bin 失败 (exit={result.returncode}), 使用手动构建")
            if result.stderr:
                print(f"  stderr: {result.stderr[:500]}")
    else:
        print("  未找到 dump_bin.py, 使用手动构建")

    if not dump_bin_ok:
        # 手动构建 features（dump_bin 不可用时的兜底）
        # qlib 二进制格式: {field}.{freq}.bin, 文件头4字节为 start_index, 后续为 float32 值
        features_dir = QLIB_DIR / "features"
        # 构建日期 → 日历索引映射
        date_to_idx = {d: i for i, d in enumerate(sorted_dates)}
        print(f"  手动构建 features (日历共 {len(sorted_dates)} 天)...")
        for csv_file in sorted(RAW_DIR.glob("*.csv")):
            coin = csv_file.stem
            df = pd.read_csv(csv_file).set_index("date").sort_index()
            coin_dir = features_dir / coin
            coin_dir.mkdir(parents=True, exist_ok=True)
            # 计算该币种第一条日期在日历中的索引（start_index）
            first_date = df.index[0]
            start_idx = date_to_idx.get(first_date, 0)
            for field in ["open", "close", "high", "low", "volume", "factor"]:
                values = df[field].values.astype(np.float32)
                data = np.hstack([start_idx, values]).astype("<f")
                data.tofile(str(coin_dir / f"{field}.day.bin"))

    # VWAP 代理：Binance 无 VWAP 数据，用 close 替代（Alpha158 需要此字段）
    # 无论 dump_bin 是否成功，都需要补充此字段
    print("  生成 VWAP 代理字段 (vwap=close)...")
    features_dir = QLIB_DIR / "features"
    if features_dir.exists():
        existing_coins = [d.name for d in features_dir.iterdir() if d.is_dir()]
        for coin in existing_coins:
            close_bin = features_dir / coin / "close.day.bin"
            vwap_bin = features_dir / coin / "vwap.day.bin"
            if close_bin.exists() and not vwap_bin.exists():
                data = np.fromfile(close_bin, dtype="<f")
                data.tofile(str(vwap_bin))
        print(f"  VWAP 代理字段已为 {len(existing_coins)} 个币种生成")
    else:
        print("  ⚠ features 目录不存在，跳过 VWAP 生成")


def main():
    parser = argparse.ArgumentParser(description="构建 Binance 现货日线数据集")
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--force", action="store_true", help="强制全量重新下载")
    parser.add_argument("--rebuild-qlib", action="store_true", help="强制重建 qlib 二进制（即使无新数据）")
    args = parser.parse_args()

    print("=" * 60)
    print(f"📥 构建 Binance 现货日线数据集 (Top {args.top})")
    print("=" * 60)

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    end_ms = int(time.time() * 1000)
    start_ms = _date_to_ms(args.start)

    # Step 0: 获取币种列表
    pairs = get_top_symbols(args.top)
    print(f"\n[Step 0] Binance 成交量前 {args.top} USDT 现货:")
    for i, (sym, base) in enumerate(pairs):
        print(f"  {i+1:3d}. {sym:15s} → {base}")

    # Step 1: 增量下载
    print(f"\n[Step 1/3] 下载日线 ({args.start} ~ {today_str})...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    total, new_total = 0, 0

    for sym, base in pairs:
        csv_file = RAW_DIR / f"{base}.csv"

        if csv_file.exists() and not args.force:
            existing = pd.read_csv(csv_file)
            last_date = existing["date"].iloc[-1]
            last_ms = _date_to_ms(last_date) + 86400000  # 从次日开始

            if last_ms >= end_ms - 86400000:
                # 数据已是最新（距现在不到1天），跳过
                print(f"  {base:10s} 已是最新 ({len(existing)} 天, 截止 {last_date})，跳过")
                total += len(existing)
                continue

            # 增量获取
            print(f"  {base:10s} ({sym}) 更新 {last_date} → {today_str} ...",
                  end=" ", flush=True)
            candles = fetch_daily(sym, last_ms, end_ms)
            if not candles:
                print(f"⚠ 无新数据")
                total += len(existing)
                continue

            # 合并去重
            new_csv = candles_to_csv(candles, base)
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
            # 全新下载
            if args.force and csv_file.exists():
                print(f"  {base:10s} 强制重新下载...", end=" ", flush=True)
            else:
                print(f"  {base:10s} ({sym}) 首次下载...", end=" ", flush=True)
            candles = fetch_daily(sym, start_ms, end_ms)
            if not candles:
                print("⚠ 无数据")
                continue

            csv_file.write_text(candles_to_csv(candles, base))
            print(f"✅ {len(candles)} 天")
            total += len(candles)
            new_total += len(candles)
            time.sleep(_REQUEST_DELAY)

    print(f"\n  总计 {total} 条日线（本次新增 {new_total} 条）")

    # Step 2+3: 重建 qlib（数据有更新 / 强制重建 / 强制下载）
    if new_total > 0 or args.force or args.rebuild_qlib:
        if new_total > 0:
            print("\n[Step 2/3] 重建 qlib 二进制...")
        elif args.rebuild_qlib:
            print("\n[Step 2/3] 强制重建 qlib 二进制 (--rebuild-qlib)...")
        else:
            print("\n[Step 2/3] 强制重建 qlib 二进制 (--force)...")
        _rebuild_qlib()
        coins = sorted([f.stem for f in RAW_DIR.glob("*.csv")])
        dates = sorted(pd.read_csv(RAW_DIR / f"{coins[0]}.csv")["date"].tolist())
        print(f"\n✅ 完成！{QLIB_DIR}")
        print(f"   币种: {len(coins)}, 时间: {dates[0]} ~ {dates[-1]}")
    else:
        print("\n✅ 数据已是最新，无需重建")


def rebuild_data(top: int = 50, start: str = "2020-01-01", force_download: bool = False):
    """增量下载数据并重建 qlib 二进制（供 execute.py 直接调用）

    Parameters
    ----------
    top : int
        Binance 成交量前 N 的 USDT 交易对。
    start : str
        数据起始日期。
    force_download : bool
        是否强制全量重新下载 CSV。
    """
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    end_ms = int(time.time() * 1000)
    start_ms = _date_to_ms(start)

    # Step 0: 获取币种列表
    pairs = get_top_symbols(top)
    print(f"\n[Step 0] Binance 成交量前 {top} USDT 现货:")
    for i, (sym, base) in enumerate(pairs):
        print(f"  {i+1:3d}. {sym:15s} → {base}")

    # Step 1: 增量下载
    print(f"\n[Step 1/3] 下载日线 ({start} ~ {today_str})...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    total, new_total = 0, 0

    for sym, base in pairs:
        csv_file = RAW_DIR / f"{base}.csv"

        if csv_file.exists() and not force_download:
            existing = pd.read_csv(csv_file)
            last_date = existing["date"].iloc[-1]
            last_ms = _date_to_ms(last_date) + 86400000

            if last_ms >= end_ms - 86400000:
                print(f"  {base:10s} 已是最新 ({len(existing)} 天, 截止 {last_date})，跳过")
                total += len(existing)
                continue

            print(f"  {base:10s} ({sym}) 更新 {last_date} → {today_str} ...",
                  end=" ", flush=True)
            candles = fetch_daily(sym, last_ms, end_ms)
            if not candles:
                print(f"⚠ 无新数据")
                total += len(existing)
                continue

            new_csv = candles_to_csv(candles, base)
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
                print(f"  {base:10s} 强制重新下载...", end=" ", flush=True)
            else:
                print(f"  {base:10s} ({sym}) 首次下载...", end=" ", flush=True)
            candles = fetch_daily(sym, start_ms, end_ms)
            if not candles:
                print("⚠ 无数据")
                continue

            csv_file.write_text(candles_to_csv(candles, base))
            print(f"✅ {len(candles)} 天")
            total += len(candles)
            new_total += len(candles)
            time.sleep(_REQUEST_DELAY)

    print(f"\n  总计 {total} 条日线（本次新增 {new_total} 条）")

    # Step 2+3: 总是重建 qlib
    print("\n[Step 2/3] 重建 qlib 二进制...")
    _rebuild_qlib()
    coins = sorted([f.stem for f in RAW_DIR.glob("*.csv")])
    dates = sorted(pd.read_csv(RAW_DIR / f"{coins[0]}.csv")["date"].tolist())
    print(f"\n✅ 完成！{QLIB_DIR}")
    print(f"   币种: {len(coins)}, 时间: {dates[0]} ~ {dates[-1]}")


if __name__ == "__main__":
    main()
