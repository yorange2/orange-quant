#!/usr/bin/env python3
"""
构建加密货币 qlib 数据集

1. 从 Hyperliquid 下载 BTC, ETH, XRP, BNB, SOL 日线数据
2. 转换为 qlib 二进制格式
3. 创建日历和股票列表文件

用法：
    python scripts/build_crypto_data.py
"""

import sys
import subprocess
from pathlib import Path

# 将项目根目录加入 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orange_quant.data.crypto_download import download_crypto_data

CRYPTO_DATA_DIR = Path.home() / ".qlib" / "qlib_data" / "crypto"
RAW_DATA_DIR = Path("data/crypto_raw")
QLIB_REPO = Path("/Users/yuanchengcheng/Documents/GitHub/qlib")


def main():
    print("=" * 60)
    print("📥 构建加密货币 qlib 数据集")
    print("=" * 60)

    # Step 1: 下载原始数据
    print("\n[Step 1/3] 从 Hyperliquid 下载日线数据...")
    download_crypto_data(output_dir=str(RAW_DATA_DIR))

    # 检查下载结果
    csv_files = list(RAW_DATA_DIR.glob("*.csv"))
    if not csv_files:
        print("❌ 没有下载到任何数据，请检查网络连接。")
        sys.exit(1)
    print(f"   已下载 {len(csv_files)} 个币种: {[f.stem for f in csv_files]}")

    # Step 2: 转换为 qlib 二进制格式
    print("\n[Step 2/3] 转换为 qlib 二进制格式...")
    CRYPTO_DATA_DIR.mkdir(parents=True, exist_ok=True)

    dump_script = QLIB_REPO / "scripts" / "dump_bin.py"
    result = subprocess.run(
        [
            sys.executable,
            str(dump_script),
            "dump_all",
            "--data_path", str(RAW_DATA_DIR),
            "--qlib_dir", str(CRYPTO_DATA_DIR),
            "--include_fields", "open,close,high,low,volume,factor",
            "--date_field_name", "date",
            "--freq", "day",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        print(f"❌ dump_bin 失败:\n{result.stderr}")
        # Fallback: try without freq flag (older qlib versions)
        print("  尝试备选方式...")
        result = subprocess.run(
            [
                sys.executable,
                str(dump_script),
                "dump_all",
                "--data_path", str(RAW_DATA_DIR),
                "--qlib_dir", str(CRYPTO_DATA_DIR),
                "--include_fields", "open,close,high,low,volume,factor",
                "--date_field_name", "date",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            print(f"❌ 备选方式也失败:\n{result.stderr}")
            print("\n手动构建数据集...")
            _build_manual()
            return
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

    # Step 3: 创建 calendar 和 instruments
    print("\n[Step 3/3] 创建日历和股票列表...")
    import pandas as pd

    all_dates = set()
    for csv_file in RAW_DATA_DIR.glob("*.csv"):
        df = pd.read_csv(csv_file, parse_dates=["date"])
        all_dates.update(df["date"].dt.strftime("%Y-%m-%d").tolist())

    sorted_dates = sorted(all_dates)
    cal_dir = CRYPTO_DATA_DIR / "calendars"
    cal_dir.mkdir(parents=True, exist_ok=True)
    (cal_dir / "day.txt").write_text("\n".join(sorted_dates))

    # instruments 文件（格式: symbol\tstart_date\tend_date）
    inst_dir = CRYPTO_DATA_DIR / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)
    coins = sorted([f.stem for f in RAW_DATA_DIR.glob("*.csv")])
    # 读取每个币种的实际日期范围
    inst_lines = []
    for coin in coins:
        df = pd.read_csv(RAW_DATA_DIR / f"{coin}.csv", parse_dates=["date"])
        coin_start = df["date"].min().strftime("%Y-%m-%d")
        coin_end = df["date"].max().strftime("%Y-%m-%d")
        inst_lines.append(f"{coin}\t{coin_start}\t{coin_end}")
    (inst_dir / "all.txt").write_text("\n".join(inst_lines))

    print(f"\n✅ 完成！数据集位于: {CRYPTO_DATA_DIR}")
    print(f"   币种: {coins}")
    print(f"   交易日: {len(sorted_dates)} 天 ({sorted_dates[0]} ~ {sorted_dates[-1]})")
    print()
    print("使用方式:")
    print("  from orange_quant.workflow.experiment import run_from_yaml")
    print('  run_from_yaml("config/workflow_config_crypto.yaml")')


def _build_manual():
    """手动构建 qlib 数据集（当 dump_bin 不可用时）"""
    import pandas as pd
    import numpy as np

    features_dir = CRYPTO_DATA_DIR / "features"
    all_dates = set()

    for csv_file in RAW_DATA_DIR.glob("*.csv"):
        coin = csv_file.stem
        df = pd.read_csv(csv_file, parse_dates=["date"])
        df = df.set_index("date").sort_index()
        all_dates.update(df.index.strftime("%Y-%m-%d").tolist())

        coin_dir = features_dir / coin
        coin_dir.mkdir(parents=True, exist_ok=True)

        for field in ["open", "close", "high", "low", "volume", "factor"]:
            arr = df[field].values.astype(np.float32)
            arr.tofile(str(coin_dir / f"day.{field}.bin"))

    # calendars
    sorted_dates = sorted(all_dates)
    cal_dir = CRYPTO_DATA_DIR / "calendars"
    cal_dir.mkdir(parents=True, exist_ok=True)
    (cal_dir / "day.txt").write_text("\n".join(sorted_dates))

    # instruments（格式: symbol\tstart_date\tend_date）
    inst_dir = CRYPTO_DATA_DIR / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)
    coins = sorted([f.stem for f in RAW_DATA_DIR.glob("*.csv")])
    inst_lines = []
    for coin in coins:
        df = pd.read_csv(RAW_DATA_DIR / f"{coin}.csv", parse_dates=["date"])
        coin_start = df["date"].min().strftime("%Y-%m-%d")
        coin_end = df["date"].max().strftime("%Y-%m-%d")
        inst_lines.append(f"{coin}\t{coin_start}\t{coin_end}")
    (inst_dir / "all.txt").write_text("\n".join(inst_lines))

    print(f"   手动构建完成，{len(coins)} 个币种，{len(sorted_dates)} 个交易日")


if __name__ == "__main__":
    main()
