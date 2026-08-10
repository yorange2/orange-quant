#!/usr/bin/env python3
"""Walk-forward rolling retrain (ROADMAP R6).

For each out-of-sample window (default 6 months), train on the preceding 3
years (25 epochs, quiet) and backtest that window only; concatenate the OOS
returns into a continuous equity curve and report annualized metrics vs the
single-shot baseline.

Data note: features/z-scores come from the cached npz (fit on the original
train segment 2018-2023). Z-score parameters therefore include the early
windows' future — a mild feature-level leak; the NAV itself uses raw price
returns and is unbiased. Strict per-window z-score refits would need a full
dataset rebuild per window (not done here).

Usage: python -m scripts.walkforward [config] [--window-months 6] [--train-years 3]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

from orange_quant.rl.backtest import rollout
from orange_quant.rl.dataset import (
    first_at_or_after, last_at_or_before, load_config, load_or_build, RotationDataset,
)
from orange_quant.rl.env import RotationEnv
from orange_quant.rl.metrics import bars_per_year, return_metrics
from orange_quant.rl.train import train_policy


def walk_forward(cfg: dict, ds: RotationDataset, window_months: int,
                 train_years: int, max_epoch: int) -> dict:
    """Run rolling windows; returns {windows, oos rows, metrics}."""
    test_start = pd.Timestamp(cfg["test"]["start"])
    data_end = pd.Timestamp(cfg["data"]["end_time"])
    step = pd.DateOffset(months=window_months)

    windows = []
    cur = test_start
    while cur < data_end:
        te = min(cur + step - pd.Timedelta(days=1), data_end)
        windows.append((cur, te))
        cur += step

    oos_rows = []  # {date, ret, nav}
    per_window = []
    for wi, (ws, we) in enumerate(windows):
        tr_end = ws - pd.Timedelta(days=1)
        tr_start = tr_end - pd.DateOffset(years=train_years) + pd.Timedelta(days=1)
        val_start = tr_end - pd.DateOffset(months=6) + pd.Timedelta(days=1)
        a, b = first_at_or_after(ds.dates, tr_start), last_at_or_before(ds.dates, tr_end)
        va, vb = first_at_or_after(ds.dates, val_start), b
        ta, tb = first_at_or_after(ds.dates, ws), last_at_or_before(ds.dates, we)
        if tb <= ta or b <= a:
            print(f"[wf] window {ws.date()}~{we.date()} not covered, skip")
            continue

        ds_w = replace(ds, split_idx={
            "train": (a, b), "valid": (va, vb), "test": (ta, tb),
        })
        tag = f"wf-{ws.strftime('%Y%m')}"
        print(f"\n[wf] window {wi + 1}/{len(windows)}: train "
              f"{ds.dates[a].astype('datetime64[D]')}~{ds.dates[b].astype('datetime64[D]')} "
              f"({b - a + 1} bars) | test {ws.date()}~{we.date()} ({tb - ta + 1} bars)")
        policy, best_rew, best_ep = train_policy(
            cfg, ds_w, max_epoch=max_epoch, model_dir=f"models/{tag}", quiet=False)

        # ---- OOS rollout on this window ----
        env = RotationEnv(ds_w, segment="test", horizon=tb - ta,
                          tiers=cfg["env"]["tiers"], max_weight=cfg["env"]["max_weight"],
                          cost_rate=cfg["env"]["cost_rate"], turnover_penalty=0.0,
                          decision_every=cfg["env"].get("decision_every", 1),
                          start_idx=ta, seed=0)
        window_rows = rollout(policy, env)
        oos_rows.extend(window_rows.to_dict("records"))
        wret = float(window_rows["nav"].iloc[-1]) - 1.0
        per_window.append({"window": f"{ws.date()}~{we.date()}",
                           "oos_return": round(wret, 4), "n_days": len(window_rows),
                           "best_valid": round(best_rew, 4)})
        print(f"[wf] window OOS return: {wret:+.4f} ({len(window_rows)} days)")

    # ---- aggregate OOS curve ----
    df = pd.DataFrame(oos_rows)
    navs = df["nav"].to_numpy()
    turnover = df["turnover"].to_numpy()
    bpy = bars_per_year(cfg)
    m = return_metrics(navs, turnover=turnover, annualize=bpy, bars_per_year=bpy)
    summary = {
        "windows": per_window,
        "oos_total_return": float(m["total_return"]),
        "oos_annual_return": float(m["annual_return"]),
        "oos_sharpe": float(m["sharpe"]),
        "oos_max_drawdown": float(m["max_drawdown"]),
        "oos_annual_turnover": float(m["annual_turnover"]),
    }
    print("\n========== walk-forward OOS (aggregated) ==========")
    for k, v in summary.items():
        if k != "windows":
            print(f"  {k:<22} {v:.4f}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="binance-rl-rotation")
    ap.add_argument("--window-months", type=int, default=6)
    ap.add_argument("--train-years", type=int, default=3)
    ap.add_argument("--max-epoch", type=int, default=25)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = load_or_build(cfg)
    summary = walk_forward(cfg, ds, args.window_months, args.train_years, args.max_epoch)
    out = Path("outputs") / args.config / "walkforward.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[wf] summary → {out}")
    sys.exit(0)


if __name__ == "__main__":
    main()
