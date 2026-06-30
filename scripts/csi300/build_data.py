#!/usr/bin/env python3
"""
数据下载脚本

一键下载 qlib 中国A股日线数据。
首次运行会下载约 1-2 GB 数据，耗时取决于网络速度。
"""

import sys
from pathlib import Path

# 将项目根目录加入 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orange_quant.data.download import download_cn_data

if __name__ == "__main__":
    print("=" * 50)
    print("📥 Orange Quant 数据下载")
    print("=" * 50)
    download_cn_data()
