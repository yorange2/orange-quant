"""把 update_cn_data.py 生成的 CSV 合并进 qlib cn_data（替换卡死的 dump_update）。

qlib 自带 dump_update 的 ProcessPoolExecutor 在 macOS/Python3.12 上 pickle 巨大对象卡死，
这里在单进程内直接完成同等工作：

1. 旧日历 ∪ 所有 CSV 日期 → 新日历
2. 每只股票：
   - 旧 bin 统一比理论长度多 1 个元素（旧 dump 的尾巴=2020-09-28，丢弃）
   - 新段按日历对齐（停牌日填 NaN），重写 7 个字段的 .bin
3. 重写 calendars/day.txt、instruments/all.txt（顺延 end、追加新股）
4. 指数成分文件（csi100/csi300/csi500）当前快照的截止日顺延到新日历末日

用法:
    python scripts/merge_cn_update.py \
        --csv-dir ~/.qlib/cn_data_update \
        --qlib-dir ~/.qlib/qlib_data/cn_data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ["open", "high", "low", "close", "volume", "amount", "factor"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default=str(Path.home() / ".qlib/cn_data_update"))
    ap.add_argument("--qlib-dir", default=str(Path.home() / ".qlib/qlib_data/cn_data"))
    args = ap.parse_args()

    csv_dir = Path(args.csv_dir)
    qlib_dir = Path(args.qlib_dir)
    cal_dir = qlib_dir / "calendars"
    inst_dir = qlib_dir / "instruments"
    feat_dir = qlib_dir / "features"

    old_cal = (cal_dir / "day.txt").read_text().splitlines()
    # {SYMBOL: (start, end)} 旧股票池
    old_inst = {}
    for line in (inst_dir / "all.txt").read_text().splitlines():
        s, a, b = line.split("\t")
        old_inst[s] = (a.strip(), b.strip())

    # 读全部 CSV：{SYMBOL: DataFrame(date, open, ..., factor)}
    print("[*] 读取 CSV ...", flush=True)
    new_data: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(csv_dir.glob("*.csv")):
        df = pd.read_csv(csv_path, dtype={"date": str})
        df["date"] = df["date"].str.strip()
        new_data[csv_path.stem] = df

    # 新日历 = 旧 ∪ 所有新日期
    all_dates = set(old_cal)
    for df in new_data.values():
        all_dates.update(df["date"].tolist())
    new_cal = sorted(all_dates)
    cal_index = {d: i for i, d in enumerate(new_cal)}
    print(f"[*] 日历: {len(old_cal)} -> {len(new_cal)} 天", flush=True)

    # 写特征 bin
    n_ok = n_new = n_skip = 0
    for symbol, df in new_data.items():
        df = df.set_index("date").sort_index()
        dates = df.index.tolist()
        start_date, end_date = old_inst.get(symbol, (dates[0], None))
        if end_date is None:
            n_new += 1  # 新股
        else:
            start_date = old_inst[symbol][0]
        sym_dir = feat_dir / symbol.lower()
        sym_dir.mkdir(parents=True, exist_ok=True)

        start_idx = cal_index[start_date]
        old_end_idx = cal_index[end_date] if end_date in cal_index else start_idx - 1

        for field in FIELDS:
            bin_path = sym_dir / f"{field}.day.bin"
            if field == "factor" and "factor" not in df.columns:
                new_vals = np.full(len(dates), np.nan, dtype=np.float32)
            else:
                new_vals = pd.to_numeric(df[field], errors="coerce").to_numpy(dtype=np.float32)
            if end_date is not None and bin_path.exists():
                old_bin = np.fromfile(bin_path, dtype="<f4")
                theoretical = old_end_idx - start_idx + 1
                old_seg = old_bin[:theoretical]  # 旧 dump 多 1 个元素(2020-09-28)，截掉
                if len(old_seg) < theoretical:  # 理论长度缺失则补 NaN
                    old_seg = np.concatenate([old_seg, np.full(theoretical - len(old_seg), np.nan, np.float32)])
                # 新段按日历对齐：停牌日填 NaN
                new_seg = np.full(len(new_cal) - 1 - old_end_idx, np.nan, dtype=np.float32)
                for pos, d in enumerate(dates):
                    ci = cal_index[d]
                    if ci > old_end_idx:
                        new_seg[ci - old_end_idx - 1] = new_vals[pos]
                merged = np.concatenate([old_seg, new_seg])
            else:
                # 新股：CSV 日期在日历中不连续的间隙填 NaN
                merged = np.full(len(new_cal) - start_idx, np.nan, dtype=np.float32)
                for d, v in zip(dates, new_vals):
                    merged[cal_index[d] - start_idx] = v
            # qlib bin 格式：[float32(start_index)] + [float32 数据]，start_index 为日历中的起始位置
            np.hstack([[start_idx], merged]).astype("<f4", copy=False).tofile(bin_path)
        n_ok += 1

    print(f"[*] 特征写盘完成: 更新 {n_ok} 只（含新股 {n_new}）", flush=True)

    # 写日历
    (cal_dir / "day.txt").write_text("\n".join(new_cal) + "\n")

    # 写 all.txt：旧股票顺延 end，新股追加
    max_date = new_cal[-1]
    lines = []
    for symbol, (s, e) in old_inst.items():
        if symbol in new_data:
            e = max(e, new_data[symbol]["date"].max())
        lines.append(f"{symbol}\t{s}\t{e}")
    for symbol in sorted(set(new_data) - set(old_inst)):
        df = new_data[symbol]
        lines.append(f"{symbol}\t{df['date'].min()}\t{df['date'].max()}")
    (inst_dir / "all.txt").write_text("\n".join(sorted(lines)) + "\n")

    # 指数成分文件：把当前快照（截止日 = 文件最大截止日）的行顺延到 max_date
    for fname in ("csi100.txt", "csi300.txt", "csi500.txt"):
        path = inst_dir / fname
        if not path.exists():
            continue
        rows = [l.split("\t") for l in path.read_text().splitlines()]
        snap_end = max(r[2].strip() for r in rows)
        out = []
        for r in rows:
            if r[2].strip() == snap_end:
                r[2] = max_date
            out.append("\t".join(r))
        (inst_dir / fname).write_text("\n".join(out) + "\n")
        print(f"[*] {fname}: 快照截止日 {snap_end} -> {max_date}", flush=True)

    print("[*] 合并完成", flush=True)


if __name__ == "__main__":
    main()
