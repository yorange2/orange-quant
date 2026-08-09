"""Dataset layer: freeze the universe, build features, precompute numpy arrays.

Reads raw per-symbol CSV bars (A-shares from the Tencent feed via
``orange_quant.data.tencent``; crypto from ``orange_quant.data.pipeline``) and
caches the result as .npz for the training loop and environment — no qlib, no
live APIs at training time.

Look-ahead guards (all enforced here):
  * universe is frozen at ``freeze_date`` (last trading day before training),
    using only liquidity from before that date;
  * feature z-scores are fit on the train segment only;
  * the feature warmup window feeds features only — it never enters rewards;
  * r_gap/r_intra are per-symbol aligned on their own last traded close, then
    re-aligned to the shared calendar (suspended days → 0 return).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

_FEATURE_COLS = [
    "mom5", "mom20", "mom60", "ret1", "vol20", "std5",
    "vol_ratio", "vol_trend", "kdj_pos", "hilo20",
]


@dataclass
class RotationDataset:
    """Precomputed daily arrays for one frozen universe."""

    dates: np.ndarray            # (T,) datetime64[D]
    codes: List[str]             # (N,) sorted by liquidity desc
    feats: np.ndarray            # (T, N, F) float32, z-scored, clip ±3, NaN→0
    r_gap: np.ndarray            # (T, N) float32, open[t]/close[prev] − 1
    r_intra: np.ndarray          # (T, N) float32, close[t]/open[t] − 1
    split_idx: Dict[str, Tuple[int, int]]  # train/valid/test → (start, end) index
    zmean: np.ndarray = None     # (N, F) z-score mean, fit on train (live reuse)
    zstd: np.ndarray = None      # (N, F) z-score std, fit on train (live reuse)

    @property
    def n_stocks(self) -> int:
        return len(self.codes)

    @property
    def n_feats(self) -> int:
        return self.feats.shape[-1]


def load_config(config_name: str) -> dict:
    with open(f"config/{config_name}.yaml", "r") as f:
        return yaml.safe_load(f)


def bar_reader(config: dict) -> Callable[[str], Optional[pd.DataFrame]]:
    """Return a per-symbol CSV reader normalized to [date,open,high,low,close,volume].

    cn CSVs (Tencent) have columns date,symbol,open,high,low,close,volume,amount;
    crypto CSVs (pipeline.candles_to_csv) have date,open,close,high,low,volume.
    The reader returns a DataFrame indexed by date (datetime64), or None when the
    symbol has no file / is unparseable.
    """
    raw_dir = Path(config["data"]["raw_dir"])

    def _read(symbol: str) -> Optional[pd.DataFrame]:
        p = raw_dir / f"{symbol}.csv"
        if not p.exists():
            return None
        try:
            df = pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()
        except Exception:  # noqa: BLE001 - unparseable file treated as missing
            return None
        if df.empty:
            return None
        cols = {c: c for c in df.columns}
        # unify column names: crypto uses open,close,high,low (pipeline order)
        if "open" not in df and "close" in df and "high" in df and "low" in df:
            pass
        if "close" in df and "high" in df and "low" in df and "volume" in df:
            keep = ["open", "high", "low", "close", "volume"]
            if "open" not in df:  # crypto CSV has open but let's be safe
                pass
            return df[[c for c in keep if c in df.columns]]
        return None

    return _read


def _per_stock_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """10 hand-built factors from OHLCV only. Returns a copy, one col per feature."""
    c, o, h, l = ohlcv["close"], ohlcv["open"], ohlcv["high"], ohlcv["low"]
    v = ohlcv["volume"]
    ret = c.pct_change(fill_method=None)
    df = pd.DataFrame(index=ohlcv.index)
    df["mom5"] = c / c.shift(5) - 1
    df["mom20"] = c / c.shift(20) - 1
    df["mom60"] = c / c.shift(60) - 1
    df["ret1"] = ret
    df["vol20"] = ret.rolling(20).std()
    df["std5"] = ret.rolling(5).std()
    df["vol_ratio"] = v / v.rolling(20).mean()          # 量比
    df["vol_trend"] = v.rolling(5).mean() / v.rolling(20).mean() - 1  # 量能趋势
    lo9, hi9 = l.rolling(9).min(), h.rolling(9).max()
    df["kdj_pos"] = (c - lo9) / (hi9 - lo9)             # 随机指标位置
    lo20, hi20 = l.rolling(20).min(), h.rolling(20).max()
    df["hilo20"] = (c - lo20) / (hi20 - lo20)           # 20 日区间位置
    return df


def build_dataset(config: dict) -> RotationDataset:
    """Load raw CSV bars, compute features/returns, z-score, split, and cache."""

    data_end = config["data"]["end_time"]
    train_s, train_e = config["train"]["start"], config["train"]["end"]
    valid_s, valid_e = config["valid"]["start"], config["valid"]["end"]
    test_s, test_e = config["test"]["start"], config["test"]["end"]
    uni = config["universe"]
    warmup_start = config["features"]["warmup_start"]

    from orange_quant.data.universe import freeze_universe as freeze

    codes = freeze(uni["raw_dir"], uni["top_n"], uni["freeze_date"],
                   uni["liquidity_start"],
                   membership=uni.get("membership"),
                   min_history_days=uni.get("min_history_days", 250))
    print(f"[data] frozen universe: {len(codes)} names, "
          f"freeze_date={uni['freeze_date']}")

    read = bar_reader(config)
    w0, w1 = pd.Timestamp(warmup_start), pd.Timestamp(data_end)
    bars = {code: read(code) for code in codes}
    bars = {c: b for c, b in bars.items() if b is not None}
    cal = pd.DatetimeIndex(sorted({
        d for b in bars.values() for d in b.index if w0 <= d <= w1
    }))
    print(f"[data] calendar {cal[0].date()} ~ {cal[-1].date()} ({len(cal)} days)")

    feats = np.full((len(cal), len(codes), len(_FEATURE_COLS)), np.nan, np.float32)
    r_gap = np.zeros((len(cal), len(codes)), np.float32)
    r_intra = np.zeros((len(cal), len(codes)), np.float32)

    for j, code in enumerate(codes):
        if code not in bars:
            continue
        one = bars[code].reindex(cal)
        f = _per_stock_features(one)
        feats[:, j, :] = f[_FEATURE_COLS].to_numpy()

        # returns first on the stock's own (unaligned) series so a resumption
        # day's gap is measured from its last traded close, then re-aligned
        c, o = one["close"], one["open"]
        prev_close = c.shift(1)
        gap = (o / prev_close - 1.0).reindex(cal)
        intra = (c / o - 1.0).reindex(cal)
        r_gap[:, j] = gap.fillna(0.0).to_numpy()
        r_intra[:, j] = intra.fillna(0.0).to_numpy()

    def first_at_or_after(day: str) -> int:
        return int(cal.searchsorted(pd.Timestamp(day)))

    def last_at_or_before(day: str) -> int:
        return int(cal.searchsorted(pd.Timestamp(day), side="right")) - 1

    # z-score fit on the train segment only (per stock, per feature); the
    # parameters are cached for live inference (orange_quant.live). A symbol
    # with no history in the train segment (late listing) gets NaN stats here —
    # neutralize so the cached params never poison live normalization.
    t0 = first_at_or_after(train_s)
    t1 = last_at_or_before(train_e)
    tr = feats[t0 : t1 + 1]
    zmean = np.nan_to_num(np.nanmean(tr, axis=0), nan=0.0)   # (N, F)
    zstd = np.nan_to_num(np.nanstd(tr, axis=0), nan=1.0)     # (N, F)
    zstd[zstd < 1e-8] = 1.0
    feats = np.clip((feats - zmean[None]) / zstd[None], -3.0, 3.0)
    feats = np.nan_to_num(feats, nan=0.0).astype(np.float32)

    split_idx = {
        "train": (first_at_or_after(train_s), last_at_or_before(train_e)),
        "valid": (first_at_or_after(valid_s), last_at_or_before(valid_e)),
        "test": (first_at_or_after(test_s), last_at_or_before(test_e)),
    }
    print(f"[data] calendar {cal[0].date()} ~ {cal[-1].date()} ({len(cal)} days)")
    for name, (a, b) in split_idx.items():
        print(f"[data] {name}: {cal[a].date()} ~ {cal[b].date()} "
              f"({b - a + 1} days)")
    print(f"[data] feats shape={feats.shape}, r_gap shape={r_gap.shape}")

    return RotationDataset(
        dates=cal.to_numpy(dtype="datetime64[D]"),
        codes=codes,
        feats=feats,
        r_gap=r_gap,
        r_intra=r_intra,
        split_idx=split_idx,
        zmean=zmean.astype(np.float32),
        zstd=zstd.astype(np.float32),
    )


def load_or_build(config: dict, force: bool = False) -> RotationDataset:
    """Load from npz cache if present, else build and cache."""
    cache_dir = Path(config["paths"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / "features.npz"
    meta_path = cache_dir / "meta.json"
    if npz_path.exists() and meta_path.exists() and not force:
        print(f"[data] loading cached dataset: {npz_path}")
        meta = json.loads(meta_path.read_text())
        z = np.load(npz_path)
        return RotationDataset(
            dates=z["dates"],
            codes=meta["codes"],
            feats=z["feats"],
            r_gap=z["r_gap"],
            r_intra=z["r_intra"],
            split_idx={k: tuple(v) for k, v in meta["split_idx"].items()},
            zmean=z["zmean"],
            zstd=z["zstd"],
        )
    ds = build_dataset(config)
    np.savez(npz_path, dates=ds.dates, feats=ds.feats, r_gap=ds.r_gap,
             r_intra=ds.r_intra, zmean=ds.zmean, zstd=ds.zstd)
    meta_path.write_text(json.dumps({
        "codes": ds.codes,
        "split_idx": {k: list(v) for k, v in ds.split_idx.items()},
    }))
    print(f"[data] cached to {npz_path}")
    return ds


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build csi300 RL rotation data")
    parser.add_argument("config", help="config name without .yaml, e.g. csi300-rl-rotation")
    parser.add_argument("--force", action="store_true", help="rebuild even if cached")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = load_or_build(cfg, force=args.force)
    print(f"[data] OK: {ds.n_stocks} stocks × {ds.n_feats} features, "
          f"segments {list(ds.split_idx)}")


if __name__ == "__main__":
    main()
