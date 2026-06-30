#!/usr/bin/env python3
"""
下载 qlib 中国 A 股日线数据

首次运行会下载约 1-2 GB 数据，耗时取决于网络速度。
数据存放: data/qlib_data/cn_data/
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PROVIDER_URI = str(PROJECT_ROOT / "data" / "qlib_data" / "cn_data")


def main():
    provider_uri = DEFAULT_PROVIDER_URI
    data_dir = Path(provider_uri)

    if data_dir.exists() and any(data_dir.iterdir()):
        print(f"[csi300] 数据目录已存在: {provider_uri}")
        print("[csi300] 如需重新下载，请删除该目录后重试。")
        return

    print(f"[csi300] 开始下载 A 股日线数据...")
    print(f"[csi300] 数据将保存到: {provider_uri}")

    try:
        import qlib
        qlib.init(provider_uri=provider_uri, region="cn")

        from qlib.tests.data import GetData
        GetData().qlib_data(
            target_dir=provider_uri,
            region="cn",
            interval="1d",
            delete_old=False,
        )
        print("[csi300] 数据下载完成！")
    except ImportError:
        raise RuntimeError(
            "无法导入 qlib。请确保已安装: pip install -e /path/to/qlib"
        )


if __name__ == "__main__":
    main()
