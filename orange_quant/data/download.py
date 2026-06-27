"""
qlib 数据下载封装

使用 qlib 官方脚本下载中国A股日线数据，转换为 qlib 二进制格式。
"""

import os
import subprocess
from pathlib import Path

# qlib 本地仓库路径（用于调用 dump_bin 脚本）
_QLIB_REPO = Path("/Users/yuanchengcheng/Documents/GitHub/qlib")

# 默认数据存放位置
DEFAULT_PROVIDER_URI = "~/.qlib/qlib_data/cn_data"


def get_default_provider_uri() -> str:
    """获取默认的 qlib 数据路径。"""
    return os.path.expanduser(DEFAULT_PROVIDER_URI)


def download_cn_data(
    provider_uri: str = DEFAULT_PROVIDER_URI,
    interval: str = "1d",
    region: str = "cn",
) -> None:
    """
    下载中国A股日线数据并转换为 qlib 二进制格式。

    首次运行会从 qlib 官方数据源下载所有历史日线数据（沪深300/中证500成分股），
    数据量约 1-2 GB，耗时取决于网络速度。

    Parameters
    ----------
    provider_uri : str
        qlib 数据目录路径。
    interval : str
        K线周期，默认 "1d"（日线）。也可用 "1min" 下载分钟数据。
    region : str
        市场区域，"cn" 表示中国A股。
    """
    provider_uri = os.path.expanduser(provider_uri)
    data_dir = Path(provider_uri)

    if data_dir.exists() and any(data_dir.iterdir()):
        print(f"[orange_quant] 数据目录已存在: {provider_uri}")
        print("[orange_quant] 如需重新下载，请删除该目录后重试。")
        return

    print(f"[orange_quant] 开始下载 {region} 市场 {interval} 数据...")
    print(f"[orange_quant] 数据将保存到: {provider_uri}")

    # 使用 qlib 的 get_data.py 脚本下载
    script = _QLIB_REPO / "scripts" / "data_collector" / "contrib" / "download_csv.sh"

    if not script.exists():
        # 改用 Python 方式下载
        _download_via_python(provider_uri, interval, region)
    else:
        subprocess.run(
            ["bash", str(script)],
            check=True,
            cwd=_QLIB_REPO,
        )

    print("[orange_quant] 数据下载完成！")


def _download_via_python(provider_uri: str, interval: str, region: str) -> None:
    """
    通过 qlib Python API 下载数据。
    使用 qlib 内置的数据下载功能。
    """
    try:
        import qlib
        qlib.init(provider_uri=provider_uri, region=region)

        from qlib.tests.data import GetData
        GetData().qlib_data(
            target_dir=provider_uri,
            region=region,
            interval=interval,
            delete_old=False,
        )
    except ImportError:
        raise RuntimeError(
            "无法导入 qlib。请确保已安装: pip install -e /path/to/qlib"
        )


if __name__ == "__main__":
    download_cn_data()
