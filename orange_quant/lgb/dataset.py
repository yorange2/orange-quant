"""LGB dataset layer: frozen universe + Alpha158 features + label, npz cache.

Mirrors ``orange_quant.rl.dataset``: reads raw per-symbol CSV bars
(``orange_quant.data.pipeline`` layout), computes the full Alpha158 feature
set (``orange_quant.lgb.features``) on the shared calendar, and caches
features/label/returns as .npz for the LightGBM training loop.

Label semantics (qlib Alpha158 defaults, ported exactly):
  * raw label = ``close[t+1+lag]/close[t+lag] - 1`` (one-day-forward return),
    NaN where not computable — rows with NaN label are dropped at train time
    (qlib ``DropnaLabel``). ``lag = label.exec_lag`` is the gap between the
    signal day and the fill, and must match how the portfolio is actually
    executed (see :func:`exec_lag`);
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


def exec_lag(config: dict) -> int:
    """Decision-day → fill-day gap, in bars. The label and the backtest both read it.

    ``1`` (default) is the legacy qlib timing: decide on close[t], fill at
    close[t+1], earn close[t+1]→close[t+2]. ``0`` matches what the live runner
    actually does — a market order minutes after the bar it decided on, so the
    fill is ≈close[t] and the position earns close[t]→close[t+1].

    Keeping this in ONE place is the point: a label built with one lag and a
    backtest run with another produces no error and no NaN, just a silently
    mismeasured strategy.
    """
    lag = int(config.get("label", {}).get("exec_lag", 1))
    if lag not in (0, 1):
        raise ValueError(f"label.exec_lag must be 0 or 1, got {lag}")
    return lag


def feature_lag(config: dict) -> int:
    """Bars the feature block is pushed back before being paired with the label.

    ``0`` (default) pairs label[t] with features computed through close[t].
    ``1`` pairs it with features through close[t−1], so ``exec_lag=0``'s label
    ``close[t+1]/close[t]−1`` no longer shares close[t] with its own features.

    That sharing is the thing worth isolating: nearly every Alpha158 column is
    built on close[t], and close[t] carries microstructure noise (bid-ask
    bounce, a late sweep that reverts) which enters the features as +e[t] and
    the exec_lag=0 label — close[t] is its denominator — as −e[t]. The induced
    correlation is pure noise, learnable, and not tradeable, since the fill is
    at that same polluted close[t]. Lagging the features severs it while
    leaving the predicted return period unchanged.
    """
    lag = int(config.get("features", {}).get("lag", 0))
    if lag < 0:
        raise ValueError(f"features.lag must be >= 0, got {lag}")
    return lag


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


def _cs_normalize(feats: np.ndarray, mode: str) -> np.ndarray:
    """Per-date cross-sectional feature normalization (roadmap C4).

    Both modes operate on the non-NaN cross-section of each date × feature
    (NaN stays NaN — LightGBM consumes missing values natively):
      * ``rank``   → percentile rank in (0, 1] (pandas pct rank);
      * ``zscore`` → (x − mean)/std, ddof=1, masked where the cross-section
        is constant (std ≈ 0).
    """
    T, N, F = feats.shape
    out = np.empty_like(feats)
    for f in range(F):
        df = pd.DataFrame(feats[:, :, f])
        if mode == "rank":
            out[:, :, f] = df.rank(axis=1, pct=True).to_numpy(np.float32)
        else:  # zscore
            # explicit sub/div(axis=0): `df - df.mean(axis=1)` is ambiguous —
            # the RangeIndex columns collide with the Series index and pandas
            # promotes the frame to (T, T)
            mean = df.mean(axis=1)
            std = df.std(axis=1)
            z = df.sub(mean, axis=0).div(std, axis=0).to_numpy(np.float32)
            z = np.where(np.isinf(z), np.nan, z)   # const cross-section → NaN
            out[:, :, f] = z
    return out


def _label_cs_zscore(label: np.ndarray, groups: list | None = None) -> np.ndarray:
    """Per-date cross-sectional z-score over non-NaN entries (ddof=1).

    Matches qlib CSZScoreNorm applied after DropnaLabel: each day's mean/std
    are computed over the coins with a valid label that day only. With
    ``groups`` (one industry name or None per column, roadmap C6), the z-score
    runs WITHIN each industry — ``None`` columns are excluded (delisted /
    unmapped names get NaN labels, dropped at train time).
    """
    # Deliberately a per-date loop, not a vectorized nanmean/nanstd: the
    # vectorized form sums in a different order and shifts the stored float32
    # labels by ~1e-6, which would make every cached features.npz irreproducible
    # for no real gain (this runs once per dataset build).
    out = np.full_like(label, np.nan, dtype=np.float64)
    per_date_cols = groups is None
    for t in range(label.shape[0]):
        row = label[t]
        valid = ~np.isnan(row)
        if per_date_cols:
            cols = [np.flatnonzero(valid)]
        else:
            cols = [np.flatnonzero(valid & (np.asarray(groups) == g))
                    for g in dict.fromkeys(g for g in groups if g is not None)]
        for col in cols:
            n = len(col)
            if n < 2:
                continue  # pandas std of < 2 → NaN → all NaN that day
            vals = row[col]
            mean = vals.mean()
            std = vals.std(ddof=1)
            if std <= 0:
                continue  # constant cross-section → division by zero
            out[t, col] = (vals - mean) / std
    return out.astype(np.float32)


def build_dataset(config: dict) -> LGBDataset:
    """Load raw CSV bars, compute Alpha158 features/label/returns, cache."""
    train_s, train_e = config["train"]["start"], config["train"]["end"]
    valid_s, valid_e = config["valid"]["start"], config["valid"]["end"]
    test_s, test_e = config["test"]["start"], config["test"]["end"]

    lag = exec_lag(config)
    print(f"[lgb-data] label.exec_lag={lag}: label[t] = "
          f"close[t+{lag + 1}]/close[t+{lag}] − 1")

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
        label_own = (c.shift(-(lag + 1)) / c.shift(-lag) - 1.0).reindex(cal)
        ret[:, j] = ret_own.fillna(0.0).to_numpy(np.float32)
        label[:, j] = label_own.to_numpy(np.float32)

    cs_norm = config.get("features", {}).get("cs_norm", "none")
    if cs_norm == "none":
        pass  # raw Alpha158 (legacy behavior, cached npz unchanged)
    elif cs_norm in ("rank", "zscore"):
        print(f"[lgb-data] per-date cross-sectional feature norm: {cs_norm}")
        feats = _cs_normalize(feats, cs_norm)
    else:
        raise ValueError(f"features.cs_norm must be rank|zscore|none, got {cs_norm!r}")

    flag = feature_lag(config)
    if flag:
        # np.concatenate, not an in-place overlapping slice assignment: the
        # latter copies low→high and would smear row 0 across the array.
        pad = np.full((flag,) + feats.shape[1:], np.nan, np.float32)
        feats = np.concatenate([pad, feats[:-flag]])
        print(f"[lgb-data] features.lag={flag}: label[t] now pairs with "
              f"features through close[t−{flag}]")

    ind_groups = None
    if config.get("label", {}).get("industry_neutral"):
        from orange_quant.data.industry import load_industry_map

        ind = load_industry_map()
        ind_groups = [ind.get(c) for c in codes]      # None → dropped at train
        n_covered = sum(g is not None for g in ind_groups)
        print(f"[lgb-data] industry-neutral label: {n_covered}/{len(codes)} covered")
    label = _label_cs_zscore(label, ind_groups)

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
        # A cache built under a different exec_lag has a label array that looks
        # perfectly valid but measures a different holding period — the exact
        # failure exec_lag() exists to prevent. Caches predating this field were
        # all built at lag 1.
        for field, want, default in (("exec_lag", exec_lag(config), 1),
                                     ("feature_lag", feature_lag(config), 0)):
            cached = int(meta.get(field, default))
            if cached != want:
                raise ValueError(
                    f"{npz_path} was built with {field}={cached} but the config "
                    f"asks for {want}. Rebuild with --force, or point "
                    f"paths.cache_dir at a separate directory.")
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
        "exec_lag": exec_lag(config),
        "feature_lag": feature_lag(config),
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
