#!/usr/bin/env python3
"""
Download qlib China A-share daily-bar data

The first run downloads roughly 1-2 GB of data; time depends on network speed.
Data location: data/qlib_data/cn_data/
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PROVIDER_URI = str(PROJECT_ROOT / "data" / "qlib_data" / "cn_data")


def main():
    provider_uri = DEFAULT_PROVIDER_URI
    data_dir = Path(provider_uri)

    if data_dir.exists() and any(data_dir.iterdir()):
        print(f"[csi300] Data directory already exists: {provider_uri}")
        print("[csi300] To re-download, delete this directory and try again.")
        return

    print(f"[csi300] Starting download of A-share daily-bar data...")
    print(f"[csi300] Data will be saved to: {provider_uri}")

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
        print("[csi300] Data download complete!")
    except ImportError:
        raise RuntimeError(
            "Failed to import qlib. Make sure it's installed: pip install -e /path/to/qlib"
        )


if __name__ == "__main__":
    main()
