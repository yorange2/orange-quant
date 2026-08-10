"""LGB dataset layer: frozen universe + Alpha158 features + label, npz cache.

Mirrors ``orange_quant.rl.dataset``: reads raw per-symbol CSV bars
(``orange_quant.data.pipeline`` layout), computes the full Alpha158 feature
set (``orange_quant.lgb.features``) on the shared calendar, and caches
features/label/returns as .npz for the LightGBM training loop.

Label semantics (qlib Alpha158 defaults, ported exactly):
  * raw label = ``close[t+2]/close[t+1] - 1`` (one-day-forward return), NaN
    where not computable — rows with NaN label are dropped at train time
    (qlib ``DropnaLabel``);
  * the stored label is the per-date cross-sectional z-score
    (``(x - mean)/std``, pandas ddof=1, stats per date over non-NaN coins —
    qlib ``CSZScoreNorm`` is stateless per date, so this is exact).

Look-ahead guards (same as the RL dataset): universe frozen at
``freeze_date`` using liquidity from before it; feature warmup only feeds
features, never labels; returns are per-coin on their own unaligned series
then re-aligned to the shared calendar (suspended days → 0 return, NaN
label).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from orange_quant.lgb.features import FEATURE_COLS, alpha158_features
from orange_quant.rl.dataset import load_bars_and_calendar, load_config, segment_idx


@dataclass
class LGBDataset:
    """Precomputed arrays for one frozen universe (raw features, no z-score)."""

    dates: np.ndarray            # (T,) datetime64[D]
    codes: List[str]             # (N,) sorted by liquidity desc
    feats: np.ndarray            # (T, N, 158) float32, raw with NaN
    label: np.ndarray            # (T, N) float32, per-date CS z-scored 1d-fwd ret
    ret: np.ndarray              # (T, N) float32, close-to-close, NaN→0
    close: np.ndarray            # (T, N) float32, close price, NaN where no bar
    split_idx: Dict[str, Tuple[int, int]]  # train/valid/test → (start, end)

    @property
    def n_stocks(self) -> int:
        return len(self.codes)

    @property
    def n_feats(self) -> int:
        return self.feats.shape[-1]


def _label_cs_zscore(label: np.ndarray) -> np.ndarray:
    """Per-date cross-sectional z-score over non-NaN entries (ddof=1).

    Matches qlib CSZScoreNorm applied after DropnaLabel: each day's mean/std
    are computed over the coins with a valid label that day only.
    """
    # Deliberately a per-date loop, not a vectorized nanmean/nanstd: the
    # vectorized form sums in a different order and shifts the stored float32
    # labels by ~1e-6, which would make every cached features.npz irreproducible
    # for no real gain (this runs once per dataset build).
    out = np.full_like(label, np.nan, dtype=np.float64)
    for t in range(label.shape[0]):
        row = label[t]
        valid = ~np.isnan(row)
        n = int(valid.sum())
        if n < 2:
            continue  # pandas std of < 2 → NaN → all NaN that day
        mean = row[valid].mean()
        std = row[valid].std(ddof=1)
        if std <= 0:
            continue  # constant cross-section → division by zero
        out[t, valid] = (row[valid] - mean) / std
    return out.astype(np.float32)


def build_dataset(config: dict) -> LGBDataset:
    """Load raw CSV bars, compute Alpha158 features/label/returns, cache."""
    train_s, train_e = config["train"]["start"], config["train"]["end"]
    valid_s, valid_e = config["valid"]["start"], config["valid"]["end"]
    test_s, test_e = config["test"]["start"], config["test"]["end"]

    codes, bars, cal = load_bars_and_calendar(config, "lgb-data")

    feats = np.full((len(cal), len(codes), len(FEATURE_COLS)), np.nan, np.float32)
    label = np.full((len(cal), len(codes)), np.nan, np.float32)
    ret = np.zeros((len(cal), len(codes)), np.float32)
    close = np.full((len(cal), len(codes)), np.nan, np.float32)

    for j, code in enumerate(codes):
        if code not in bars:
            continue
        one = bars[code].reindex(cal)
        f = alpha158_features(one)
        feats[:, j, :] = f[FEATURE_COLS].to_numpy(np.float32)

        # returns/labels on the coin's own unaligned series, then re-aligned,
        # so a resumption day measures from its last traded close.
        c = one["close"]
        close[:, j] = c.to_numpy(np.float32)
        ret_own = (c.shift(-1) / c - 1.0).reindex(cal)
        label_own = (c.shift(-2) / c.shift(-1) - 1.0).reindex(cal)
        ret[:, j] = ret_own.fillna(0.0).to_numpy(np.float32)
        label[:, j] = label_own.to_numpy(np.float32)

    label = _label_cs_zscore(label)

    split_idx = {
        "train": segment_idx(cal, train_s, train_e),
        "valid": segment_idx(cal, valid_s, valid_e),
        "test": segment_idx(cal, test_s, test_e),
    }
    for name, (a, b) in split_idx.items():
        print(f"[lgb-data] {name}: {cal[a].date()} ~ {cal[b].date()} "
              f"({b - a + 1} days)")
    nan_rate = float(np.isnan(feats).mean())
    print(f"[lgb-data] feats shape={feats.shape}, label shape={label.shape}, "
          f"feature NaN rate={nan_rate:.4f}")

    return LGBDataset(
        dates=cal.to_numpy(dtype="datetime64[D]"),
        codes=codes,
        feats=feats,
        label=label,
        ret=ret,
        close=close,
        split_idx=split_idx,
    )


def load_or_build(config: dict, force: bool = False) -> LGBDataset:
    """Load from npz cache if present, else build and cache."""
    cache_dir = Path(config["paths"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path = cache_dir / "features.npz"
    meta_path = cache_dir / "meta.json"
    if npz_path.exists() and meta_path.exists() and not force:
        print(f"[lgb-data] loading cached dataset: {npz_path}")
        meta = json.loads(meta_path.read_text())
        z = np.load(npz_path)
        return LGBDataset(
            dates=z["dates"],
            codes=meta["codes"],
            feats=z["feats"],
            label=z["label"],
            ret=z["ret"],
            close=z["close"],
            split_idx={k: tuple(v) for k, v in meta["split_idx"].items()},
        )
    ds = build_dataset(config)
    np.savez(npz_path, dates=ds.dates, feats=ds.feats, label=ds.label, ret=ds.ret,
             close=ds.close)
    meta_path.write_text(json.dumps({
        "codes": ds.codes,
        "split_idx": {k: list(v) for k, v in ds.split_idx.items()},
        "feature_names": FEATURE_COLS,
    }))
    print(f"[lgb-data] cached to {npz_path}")
    return ds


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build binance LGB momtopk data")
    parser.add_argument("config", help="config name without .yaml, e.g. binance-lgb-momtopk")
    parser.add_argument("--force", action="store_true", help="rebuild even if cached")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = load_or_build(cfg, force=args.force)
    print(f"[lgb-data] OK: {ds.n_stocks} coins × {ds.n_feats} features, "
          f"segments {list(ds.split_idx)}")


if __name__ == "__main__":
    main()
