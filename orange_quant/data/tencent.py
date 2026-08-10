"""A-share daily bars from the Tencent K-line API (data/cn_raw, CSV per symbol).

The Tencent endpoint returns a *trailing* window anchored at the request's end
date (the start date is ignored, max 640 rows ≈ 2.6 years), so full-history
fetching uses end-anchored pagination: keep moving ``end`` back to the earliest
returned date minus one day until the requested start is covered. (The legacy
script's fixed 3-year loop [2020, 2023, 2026] has a documented hole in
2021-01~2021-05.)

Deliverables per symbol: ``data/cn_raw/{SYMBOL}.csv`` with columns
``date,symbol,open,high,low,close,volume,amount``. hfq (backward-adjusted)
prices are used directly as the investment series — no factor/raw split needed
now that there is no qlib semantics. Indices (SH000/SZ399) use the unadjusted
variant (key ``day``).

Volume units (verified 2026-08): main boards and ChiNext are in lots (×100);
STAR (SH688) and indices are already in shares. Amount is in 10k CNY (×10000).
NOTE: the legacy script excluded SZ000 from ×100 — that was a unit bug.

Usage:
    python -m orange_quant.data.tencent --symbols-file <tsv>   # fetch a list
    python -m orange_quant.data.tencent --symbols-file <tsv> --limit 5
    python -m orange_quant.data.tencent --index-only           # benchmark only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

URL = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
BATCH_MAX = 640  # rows per request ≈ 2.6 years
ADJ_HFQ = "hfq"
ADJ_INDEX = ""


def is_index(symbol: str) -> bool:
    """SH000xxx / SZ399xxx are indices (no adjustment, volume in shares)."""
    code = symbol[2:]
    return code.startswith("000") or code.startswith("399")


def _request(symbol: str, end_date: str, adj: str, retries: int = 3) -> Dict[str, list]:
    """One trailing-window request. Returns {date: raw row}."""
    url = URL
    params = {
        "_var": f"kline_day{adj or 'idx'}{end_date[:4]}",
        "param": f"{symbol.lower()},day,2000-01-01,{end_date},640,{adj}",
        "r": "0.5",
    }
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            txt = r.text
            if "={" not in txt:
                raise ValueError("bad response")
            d = json.loads(txt[txt.find("={") + 1:])["data"][symbol.lower()]
            key = next(k for k in ("hfqday", "qfqday", "day") if k in d)
            return {row[0]: row for row in d[key]}
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise AssertionError("unreachable")


def fetch_daily(symbol: str, start: str, end: str) -> pd.DataFrame:
    """End-anchored paginated fetch of [start, end] daily bars (inclusive).

    Returns DataFrame [date, open, close, high, low, volume, amount] with date
    as YYYY-MM-DD strings; missing days simply absent (suspension semantics).
    """
    adj = ADJ_INDEX if is_index(symbol) else ADJ_HFQ
    merged: Dict[str, list] = {}
    cur_end = end
    empty_streak = 0
    while True:
        batch = _request(symbol, cur_end, adj)
        if not batch:
            empty_streak += 1
            if empty_streak >= 2:
                break
            # retreat a few years and retry once
            y = int(cur_end[:4]) - 3
            cur_end = f"{y}-12-31"
            continue
        empty_streak = 0
        merged.update(batch)
        first = min(batch)
        if first <= start:
            break
        cur_end = (pd.Timestamp(first) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    rows = [merged[k] for k in sorted(merged) if start <= k <= end]
    if not rows:
        return pd.DataFrame(columns=["date", "open", "close", "high", "low", "volume", "amount"])
    df = pd.DataFrame(rows)
    df = df.iloc[:, :9]
    df.columns = ["date", "open", "close", "high", "low", "volume", "_1", "turnover", "amount"]
    df = df[["date", "open", "close", "high", "low", "volume", "amount"]]
    for col in ["open", "close", "high", "low", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _normalize_units(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Volume lots→shares (except STAR board and indices) and amount 万元→元.

    The unit rules are a documented past bug source — keep them in one place."""
    if not symbol.startswith(("SH688", "SZ399", "SH000")):
        df["volume"] = df["volume"] * 100
    df["amount"] = df["amount"] * 10000
    return df.dropna(subset=["close"])


def fetch_symbol(symbol: str, start: str, end: str, out_dir: Path,
                 force: bool = False) -> Tuple[str, Optional[str]]:
    """Fetch one symbol and write data/cn_raw/{SYMBOL}.csv. (symbol, err).

    Resumable: an existing CSV is extended incrementally from its last date
    (unless ``force``, which re-fetches everything); fully up-to-date files
    are skipped.
    """
    csv_path = out_dir / f"{symbol}.csv"
    if csv_path.exists() and not force:
        existing = pd.read_csv(csv_path, parse_dates=["date"])
        last = str(existing["date"].max().date())
        if last >= end:
            return symbol, None  # already up to date
        fetch_start = last  # fetch_daily is end-anchored, start is a floor
        try:
            df = fetch_daily(symbol, fetch_start, end)
        except Exception as e:  # noqa: BLE001 - per-symbol failure isolation
            return symbol, f"{type(e).__name__}: {e}"
        if df.empty:
            return symbol, None  # nothing new
        df = _normalize_units(df, symbol)
        fresh = df[df["date"] > last]
        if fresh.empty:
            return symbol, None
        fresh["symbol"] = symbol
        combined = pd.concat([existing, fresh]).drop_duplicates(
            subset="date", keep="last").sort_values("date")
        combined[["date", "symbol", "open", "high", "low", "close",
                  "volume", "amount"]].to_csv(csv_path, index=False)
        return symbol, None

    try:
        df = fetch_daily(symbol, start, end)
        if df.empty:
            return symbol, "no data"
        df = _normalize_units(df, symbol)
        df["symbol"] = symbol
        df[["date", "symbol", "open", "high", "low", "close", "volume", "amount"]].to_csv(
            csv_path, index=False
        )
        return symbol, None
    except Exception as e:  # noqa: BLE001 - per-symbol failure isolation
        return symbol, f"{type(e).__name__}: {e}"


def load_symbol_list(symbols_file: Optional[str], index_only: bool) -> List[str]:
    """Symbols to fetch: index_only → SH000300; else a file (TSV/CSV, 1st col)."""
    if index_only:
        return ["SH000300"]
    if not symbols_file:
        raise SystemExit("need --symbols-file <tsv|csv> (1st column = symbol) or --index-only")
    lines = Path(symbols_file).read_text().splitlines()
    syms = []
    for ln in lines:
        if not ln.strip() or ln.startswith("#"):
            continue
        first = ln.split()[0].split(",")[0].strip()
        if first.startswith(("SH", "SZ")):
            syms.append(first)
    return sorted(set(syms))


def main() -> None:
    ap = argparse.ArgumentParser(description="Tencent A-share daily fetch → data/cn_raw")
    ap.add_argument("--out", default="data/cn_raw")
    ap.add_argument("--start", default="2011-01-01")
    ap.add_argument("--end", default=time.strftime("%Y-%m-%d"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="first N symbols (smoke)")
    ap.add_argument("--force", action="store_true", help="re-fetch existing CSVs")
    ap.add_argument("--symbols-file", default=None,
                    help="TSV/CSV whose first column is SH/SZ symbols (e.g. qlib all.txt)")
    ap.add_argument("--index-only", action="store_true", help="fetch benchmark only")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols = load_symbol_list(args.symbols_file, args.index_only)
    if args.limit:
        symbols = symbols[: args.limit]
    print(f"[tencent] {len(symbols)} symbols → {out_dir} ({args.start} ~ {args.end})")

    ok, fail = 0, {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(fetch_symbol, s, args.start, args.end, out_dir, args.force): s
            for s in symbols
        }
        for i, fut in enumerate(as_completed(futs), 1):
            sym, err = fut.result()
            if err:
                fail[sym] = err
            else:
                ok += 1
            if i % 50 == 0:
                print(f"[tencent] {i}/{len(symbols)} ok={ok} fail={len(fail)} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    print(f"[tencent] done: ok={ok} fail={len(fail)} ({time.time() - t0:.0f}s)")
    if fail:
        print("[tencent] failures:", list(fail.items())[:10], flush=True)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
