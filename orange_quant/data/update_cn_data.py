"""抓取 2020-09-26 之后的 A 股日线数据（腾讯行情源），输出 dump_update 可用的 CSV。

旧 qlib cn_data（Yahoo 源）止于 2020-09-25。本脚本直连腾讯 K 线接口抓取后复权+不复权数据
（东财 push2 接口在本网络被限流、新浪接口依赖 mini-racer 在 macOS 上会段错误、BaoStock 登录无响应，
故选腾讯），以各股票在旧数据最后一日的收盘价为基准做缩放，保证新旧价格在衔接点连续；
$factor = scale * hfq / raw，保持 raw = $close / $factor 语义。
成交量统一为「股」（与旧数据一致），成交额统一为「元」。

用法:
    python -m orange_quant.data.update_cn_data                # 全量抓取
    python -m orange_quant.data.update_cn_data --limit 5      # 只抓前 5 只，用于冒烟测试
    python -m orange_quant.data.update_cn_data --out <dir> --qlib-dir <dir> --workers 8
    # 兼容入口（旧路径，转发到本模块）：python scripts/update_cn_data.py ...

之后合并:
    python qlib/scripts/dump_bin.py dump_update \
        --data_path <out> --qlib_dir ~/.qlib/qlib_data/cn_data --backup_dir ~/.qlib/backup
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# 旧数据的最后一天（Yahoo 源 v1 数据集截止日）
JUNCTION = "2020-09-25"
START_DATE = "20200920"  # 早于 junction 几个交易日，用于计算衔接缩放


def load_old_universe(qlib_dir: Path):
    """读取旧数据股票池及其起止日期。返回 {SYMBOL: (start_date, end_date)}。"""
    inst_file = Path(qlib_dir) / "instruments" / "all.txt"
    uni = {}
    for line in inst_file.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            uni[parts[0]] = (parts[1], parts[2])
    return uni


def old_junction_close(symbol: str, qlib_dir: Path, calendar: list[str]) -> tuple[float | None, str | None]:
    """旧数据中该股票在衔接点的调整收盘价及其日期。

    取旧数据最后一个非 NaN 的调整收盘价（衔接日停牌则回退到更早的交易日）。
    返回 (close_value, junction_date)。
    """
    uni = load_old_universe(qlib_dir)
    if symbol not in uni:
        return None, None
    start_date, end_date = uni[symbol]
    try:
        start_idx = calendar.index(start_date)
        end_idx = calendar.index(end_date)
    except ValueError:
        return None, None
    bin_path = Path(qlib_dir) / "features" / symbol.lower() / "close.day.bin"
    if not bin_path.exists():
        return None, None
    close = np.fromfile(bin_path, dtype="<f4")
    # bin 格式：[float32(start_index)] + 数据，数据第 i 个元素对应日历[start_idx+i]
    for i in range(end_idx - start_idx, -1, -1):
        pos = i + 1  # +1 跳过前缀
        if 0 <= pos < len(close) and not np.isnan(close[pos]):
            return float(close[pos]), calendar[start_idx + i]
    return None, None


def is_index(symbol: str) -> bool:
    """SH000xxx / SZ399xxx 为指数。"""
    code = symbol[2:]
    return code.startswith("000") or code.startswith("399")


def tx_rows(symbol: str, adj: str) -> pd.DataFrame:
    """腾讯 K 线直连（绕开 akshare 的 mini-racer / 每页等待）。

    行格式: [date, open, close, high, low, volume(手), {}, turnover, amount(万元), ...]
    按年分页（单次最多 640 行 ≈ 2.5 年），[2020, 2023, 2026] 覆盖 2020-09 之后全部交易日。
    """
    rows_by_date: dict[str, list] = {}
    for year in (2020, 2023, 2026):
        url = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
        params = {
            "_var": f"kline_day{adj}{year}",
            "param": f"{symbol.lower()},day,{year}-01-01,{year + 1}-12-31,640,{adj}",
            "r": "0.5",
        }
        for attempt in range(3):
            try:
                r = requests.get(url, params=params, timeout=15)
                txt = r.text
                if "={" not in txt:
                    raise ValueError("bad response")
                d = json.loads(txt[txt.find("={") + 1:])["data"][symbol.lower()]
                key = next(k for k in ("hfqday", "qfqday", "day") if k in d)
                for row in d[key]:
                    rows_by_date[row[0]] = row
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1.5 * (attempt + 1))
    return pd.DataFrame([rows_by_date[k] for k in sorted(rows_by_date)])


def fetch_symbol(symbol: str, qlib_dir: Path, calendar: list[str], start: str, end: str,
                 out_dir: Path, cache: dict, force: bool = False) -> tuple[str, str | None]:
    """抓取单个股票并写 CSV。返回 (symbol, None=成功 / 错误信息)。"""
    csv_path = out_dir / f"{symbol}.csv"
    if csv_path.exists() and not force:  # 断点续跑
        return symbol, None

    old_close, junction = old_junction_close(symbol, qlib_dir, calendar)
    adj = "" if is_index(symbol) else "hfq"
    df = tx_rows(symbol, adj)
    if df.empty:
        return symbol, "no data"

    df.columns = ["date", "open", "close", "high", "low", "volume", "_1", "turnover", "amount"] + \
        list(df.columns[9:])
    df = df[["date", "open", "close", "high", "low", "volume", "amount"]].copy()
    for col in ["open", "close", "high", "low", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # 单位：主板/创业板成交量是手→股（×100）；科创板与指数已是股；成交额万元→元
    if not symbol.startswith(("SH688", "SZ399", "SH000", "SZ000")):
        df["volume"] = df["volume"] * 100
    df["amount"] = df["amount"] * 10000

    # 衔接缩放：旧数据最后有效交易日的调整价 / 腾讯 hfq 同日价格
    scale = 1.0
    if old_close is not None:
        jun = df[df["date"] == junction]
        if not jun.empty:
            scale = old_close / float(jun["close"].iloc[0])

    if is_index(symbol):
        df["factor"] = scale  # 指数无复权：factor = scale，close/factor = 实际点位
    else:
        raw = tx_rows(symbol, "")
        if raw.empty:
            df["factor"] = scale
        else:
            raw = raw.iloc[:, [0, 2]].rename(columns={0: "date", 2: "raw_close"})
            raw["raw_close"] = pd.to_numeric(raw["raw_close"], errors="coerce")
            df = df.merge(raw, on="date", how="left")
            df["factor"] = scale * df["close"] / df["raw_close"].replace(0, np.nan)
        df = df.drop(columns=["raw_close"], errors="ignore")

    df["symbol"] = symbol
    # 价格×scale；factor 已含 scale（= scale*hfq/raw），不能再乘
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce") * scale
    df = df[df["date"] > (junction or JUNCTION)]  # 只保留衔接点之后的数据
    df = df.dropna(subset=["close"])
    if df.empty:
        return symbol, "no rows after junction"

    df[["date", "symbol", "open", "high", "low", "close", "volume", "amount", "factor"]].to_csv(
        csv_path, index=False
    )
    return symbol, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / ".qlib/cn_data_update"))
    ap.add_argument("--qlib-dir", default=str(Path.home() / ".qlib/qlib_data/cn_data"))
    ap.add_argument("--end", default=time.strftime("%Y%m%d"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 个符号（冒烟测试）")
    ap.add_argument("--force", action="store_true", help="忽略已有 CSV 重新抓取（修复衔接缩放后使用）")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    qlib_dir = Path(args.qlib_dir)

    calendar = (qlib_dir / "calendars" / "day.txt").read_text().splitlines()
    universe = load_old_universe(qlib_dir)
    symbols = sorted(universe.keys())

    # 补充 2020 之后的新上市股票：当前 A 股代码 - 旧股票池
    # 东财 push2 接口被限、新浪列表依赖 mini-racer（macOS 上会段错误），用交易所官方列表；
    # szse.cn 偶发 SSL 中断，加重试并缓存到本地（一天内不重新拉取）
    import akshare as ak
    cache_file = out_dir.parent / "cn_stock_list.csv"
    codes = []
    if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 86400:
        codes = [l.strip() for l in cache_file.read_text().splitlines() if l.strip()]
    if not codes:
        print("[*] 获取当前 A 股列表（交易所官方源）...", flush=True)
        sh = sz = None
        for attempt in range(3):
            try:
                sh = ak.stock_info_sh_name_code()["证券代码"].astype(str).str.zfill(6)
                sz = ak.stock_info_sz_name_code()["A股代码"].astype(str).str.zfill(6)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(3 * (attempt + 1))
        codes = sorted([f"SH{c}" for c in sh] + [f"SZ{c}" for c in sz])
        cache_file.write_text("\n".join(codes))
    new_symbols = [s for s in codes if s not in universe]
    symbols += new_symbols
    if args.limit:
        symbols = symbols[: args.limit]

    print(f"[*] 共 {len(symbols)} 个符号（旧池 {len(universe)} + 新上市 {len(new_symbols)}）", flush=True)

    ok, fail = 0, {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(fetch_symbol, s, qlib_dir, calendar, START_DATE, args.end, out_dir, {}, args.force): s
            for s in symbols
        }
        for i, fut in enumerate(as_completed(futs), 1):
            sym, err = fut.result()
            if err:
                fail[sym] = err
            else:
                ok += 1
            if i % 100 == 0:
                print(f"[*] {i}/{len(symbols)}  成功 {ok}  失败 {len(fail)}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"[*] 完成：成功 {ok}，失败 {len(fail)}（{time.time()-t0:.0f}s）", flush=True)
    if fail:
        print("[*] 失败样例：", list(fail.items())[:10], flush=True)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
