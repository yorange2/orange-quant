#!/usr/bin/env python3
"""Live retrain scheduler (walk-forward engineering, ROADMAP R6).

Retrains the production policy on a rolling window — by default the trailing
3 years (train) with the last 6 months as validation — then atomically swaps
the model into ``models/<cfg>/policy_best.pth``. The live server picks the new
model up on its next daily run (live.py loads the checkpoint every run), so no
server restart is needed.

Run from cron, e.g. quarterly:
    0 3 1 1,4,7,10 *  cd /path/to/orange-quant && .venv/bin/python -m scripts.retrain_live --config binance-rl-rotation

Notes:
- The dataset npz is rebuilt first (--force) so training sees data through
  today; crypto bars are incrementally refreshed via the pipeline first.
- Z-scores come from the config's train segment (the same mild feature-level
  caveat as scripts/walkforward.py); NAV itself is unbiased.
- Every checkpoint write is atomic (write .tmp, os.replace).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from orange_quant.rl.dataset import (
    first_at_or_after, last_at_or_before, load_config, load_or_build,
)
from orange_quant.rl.train import train_policy


def refresh_data(cfg: dict) -> None:
    """Incremental bar refresh (CSV resumable) for the venue's raw store."""
    venue = cfg["market"]["venue"]
    freq = cfg["data"].get("freq", "1d")
    if venue == "tencent":
        # A-share: extend every existing CSV from its last date (incremental)
        from orange_quant.data.tencent import fetch_symbol
        from concurrent.futures import ThreadPoolExecutor, as_completed

        raw = Path(cfg["data"]["raw_dir"])
        raw.mkdir(parents=True, exist_ok=True)
        symbols = sorted(f.stem for f in raw.glob("*.csv"))
        end = cfg["data"]["end_time"]
        print(f"[retrain] tencent: incremental refresh of {len(symbols)} CSVs "
              f"(through {end})")
        ok, fail = 0, {}
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(fetch_symbol, s, "2011-01-01", end, raw): s
                    for s in symbols}
            for fut in as_completed(futs):
                sym, err = fut.result()
                if err:
                    fail[sym] = err
                else:
                    ok += 1
        if fail:
            print(f"[retrain] tencent refresh failures: {list(fail)[:5]}")
        print(f"[retrain] tencent refresh done: ok={ok}")
        return
    from orange_quant.data import pipeline
    from orange_quant.data.build import get_source

    source = get_source(venue).build_source()
    pipeline.rebuild_data(source, top=cfg["universe"]["top_n"],
                          start=cfg["data"]["start_time"], freq=freq)
    if freq == "1h":
        # hourly configs also need the daily store the npz is built from
        pipeline.rebuild_data(source, top=cfg["universe"]["top_n"],
                              start=cfg["data"]["start_time"], freq="1d")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-years", type=int, default=3)
    ap.add_argument("--valid-months", type=int, default=6)
    ap.add_argument("--max-epoch", type=int, default=50)
    ap.add_argument("--no-refresh", action="store_true",
                    help="skip data refresh/npz rebuild (test runs)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]

    if not args.no_refresh:
        refresh_data(cfg)
    ds = load_or_build(cfg, force=not args.no_refresh)

    # rolling window ending today (train) / 6 months ago (valid start)
    today = pd.Timestamp(datetime.now().date())
    tr_end = today - timedelta(days=1)
    tr_start = tr_end - pd.DateOffset(years=args.train_years) + timedelta(days=1)
    va_start = tr_end - pd.DateOffset(months=args.valid_months) + timedelta(days=1)
    a, b = first_at_or_after(ds.dates, tr_start), last_at_or_before(ds.dates, tr_end)
    va, vb = first_at_or_after(ds.dates, va_start), b
    if b <= a or vb <= va:
        raise SystemExit(f"[retrain] insufficient data for window "
                         f"{tr_start.date()}~{tr_end.date()}")
    # the valid segment must fit at least one training horizon — A-share
    # trading calendars have ~21 bars/month, so 6 months can be < horizon
    need = cfg["env"]["horizon"] * cfg["env"].get("decision_every", 1)
    while vb - va < need and va_start > tr_start:
        va_start -= pd.DateOffset(months=1)
        va = first_at_or_after(ds.dates, va_start)
    ds_w = replace(ds, split_idx={"train": (a, b), "valid": (va, vb),
                                  "test": (vb, b)})
    print(f"[retrain] {args.config}: train {ds.dates[a].astype('datetime64[D]')}"
          f"~{ds.dates[b].astype('datetime64[D]')} ({b - a + 1} bars), "
          f"valid {ds.dates[va].astype('datetime64[D]')}~{ds.dates[vb].astype('datetime64[D]')}")

    policy, best_reward, best_epoch = train_policy(
        cfg, ds_w, max_epoch=args.max_epoch,
        model_dir=paths["model_dir"], quiet=False)

    hist = Path(paths["model_dir"]) / "retrain_history.json"
    entries = json.loads(hist.read_text()) if hist.exists() else []
    entries.append({
        "time": datetime.now().isoformat(timespec="minutes"),
        "train": f"{tr_start.date()}~{tr_end.date()}",
        "valid_mean_reward": round(best_reward, 6),
        "best_epoch": best_epoch,
        "model": str(Path(paths["model_dir"]) / "policy_best.pth"),
    })
    hist.write_text(json.dumps(entries[-20:], indent=2))
    print(f"[retrain] done: best valid {best_reward:.6f} (epoch {best_epoch}); "
          f"model atomically swapped; history → {hist}")


if __name__ == "__main__":
    main()
