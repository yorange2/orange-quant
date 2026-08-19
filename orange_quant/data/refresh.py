"""Pre-trade incremental bar refresh for the live runners.

The live path reads per-symbol CSVs from ``data.raw_dir`` (via
``rl.dataset.bar_reader``); nothing in the trading server updates them, so a
container that only runs ``orange_quant.server`` will happily keep trading on
whatever bars were last downloaded by hand. This module closes that gap: it
appends the missing closed bars for the *strategy universe only* right before
the signal is computed, then reports how fresh the data actually is so the
caller can refuse to trade on stale bars.

Deliberately narrower than ``pipeline.rebuild_data``: that one re-ranks the
venue's top-N by live 24h volume and rewrites the whole file set, which is a
research/build operation. Here the universe is frozen by the config and only
those coins matter, so a bad ranking call can never change what gets traded.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from orange_quant.data.pipeline import candles_to_csv

_REQUEST_DELAY = 0.3
_SYMBOL_LOOKUP_N = 300  # deep enough that a frozen-universe coin is always in it


def _utc_today() -> datetime:
    return datetime.now(timezone.utc)


def _resolve_symbols(source, coins: List[str]) -> tuple:
    """Map coin → venue symbol. Returns (mapping, unresolved).

    Venues spell symbols differently (Binance ``BTCUSDT`` vs Hyperliquid ccxt
    ``BTC/USDC``), so the mapping comes from the venue's own listing rather
    than string concatenation. A network failure here is not fatal — it just
    means nothing gets refreshed and the freshness check has the final say.
    """
    try:
        pairs = source.get_top_symbols(_SYMBOL_LOOKUP_N)
    except Exception as e:  # noqa: BLE001 - offline/ratelimited: stay non-fatal
        print(f"[refresh] symbol lookup failed: {e}")
        return {}, list(coins)
    by_coin = {coin: sym for sym, coin in pairs}
    mapping = {c: by_coin[c] for c in coins if c in by_coin}
    return mapping, [c for c in coins if c not in by_coin]


def _stamp_to_ms(stamp: str) -> int:
    """UTC epoch ms for ``YYYY-MM-DD`` or ``YYYY-MM-DD HH:MM:SS``.

    Hourly CSVs carry the hour, so a date-only parse raises here exactly as it
    did in ``pipeline._date_to_ms`` — which is why hourly incremental updates
    never worked before.
    """
    stamp = str(stamp).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return int(datetime.strptime(stamp, fmt)
                       .replace(tzinfo=timezone.utc).timestamp() * 1000)
        except ValueError:
            continue
    raise ValueError(f"unparseable bar timestamp: {stamp!r}")


def _append_bars(csv_file: Path, candles: list, freq: str = "1d") -> int:
    """Merge freshly fetched candles into an existing CSV. Returns rows added.

    ``freq`` must match the file: ``candles_to_csv`` writes date-only stamps for
    ``1d`` and keeps the hour for ``1h``. Formatting hourly bars as ``1d`` would
    collapse 24 rows onto one date and then ``drop_duplicates`` would keep a
    single one — silent data loss, so the frequency is threaded through rather
    than assumed.
    """
    new_df = pd.read_csv(pd.io.common.StringIO(candles_to_csv(candles, freq)))
    if not csv_file.exists():
        new_df.to_csv(csv_file, index=False)
        return len(new_df)
    existing = pd.read_csv(csv_file)
    combined = (pd.concat([existing, new_df])
                .drop_duplicates(subset="date", keep="last")
                .sort_values("date"))
    combined.to_csv(csv_file, index=False)
    return len(combined) - len(existing)


def latest_bar_dates(raw_dir: Path, coins: Iterable[str]) -> Dict[str, Optional[str]]:
    """{coin: newest bar date in its CSV} — None when the file is missing/empty."""
    out: Dict[str, Optional[str]] = {}
    for coin in coins:
        p = Path(raw_dir) / f"{coin}.csv"
        if not p.exists():
            out[coin] = None
            continue
        try:
            d = pd.read_csv(p, usecols=["date"])["date"]
            out[coin] = str(d.iloc[-1]) if len(d) else None
        except Exception:  # noqa: BLE001 - unreadable file == no data
            out[coin] = None
    return out


def refresh_bars(cfg: dict, coins: List[str]) -> dict:
    """Append missing closed bars for ``coins``. Never raises.

    Frequency comes from ``data.freq`` (``1d`` default, ``1h`` for the
    rolling-24h strategy, whose views each need their own clock hour present).
    Returns ``{"added": {coin: n}, "errors": {coin: msg}, "unresolved": [...]}``.
    Per-coin isolation: one dead symbol never blocks the rest, and the caller
    decides what to do about the resulting staleness.
    """
    from orange_quant.data.build import get_source

    venue = cfg.get("market", {}).get("venue", "binance")
    source = get_source(venue)
    raw_dir = Path(cfg["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    freq = str(cfg.get("data", {}).get("freq", "1d"))
    if freq not in ("1d", "1h"):
        raise ValueError(f"data.freq must be 1d or 1h, got {freq!r}")
    hourly = freq == "1h"
    step_ms = 3600000 if hourly else 86400000
    fetch = source.fetch_hourly if hourly else source.fetch_daily
    if fetch is None:
        raise ValueError(f"venue {venue} has no {freq} fetch hook")

    mapping, unresolved = _resolve_symbols(source, coins)
    if unresolved:
        print(f"[refresh] no venue symbol for: {', '.join(unresolved)}")

    end_ms = int(time.time() * 1000)
    added: Dict[str, int] = {}
    errors: Dict[str, str] = {}

    for coin, symbol in mapping.items():
        csv_file = raw_dir / f"{coin}.csv"
        try:
            if csv_file.exists():
                last_date = str(pd.read_csv(csv_file, usecols=["date"])["date"].iloc[-1])
                start_ms = _stamp_to_ms(last_date) + step_ms
                if start_ms >= end_ms:
                    continue                       # already through the last closed bar
            else:
                start = cfg.get("data", {}).get("start_time", "2020-01-01")
                start_ms = _stamp_to_ms(start)
            candles = fetch(symbol, start_ms, end_ms)
            if candles:
                n = _append_bars(csv_file, candles, freq)
                if n:
                    added[coin] = n
            time.sleep(_REQUEST_DELAY)
        except Exception as e:  # noqa: BLE001 - per-coin isolation
            errors[coin] = str(e)
            print(f"[refresh] {coin} failed: {e}")

    if added:
        print(f"[refresh] +bars: {', '.join(f'{c}+{n}' for c, n in sorted(added.items()))}")
    else:
        print("[refresh] no new bars (already current)")
    return {"added": added, "errors": errors, "unresolved": unresolved, "freq": freq}


# kept so existing daily callers/imports keep working after the rename
refresh_daily = refresh_bars


def refresh_and_gate(cfg: dict, coins: List[str], enabled: bool,
                     max_age_days: int, min_live: Optional[int] = None) -> tuple:
    """Refresh then gate: ``(ok_to_trade, report)``. Both runners share this.

    Disabled (the research/backtest default) is a pass-through, so nothing but
    the live configs ever touches the network here.
    """
    if not enabled:
        return True, {"refreshed": False, "live": list(coins), "dropped": []}
    report = refresh_bars(cfg, coins)
    ok, detail = check_freshness(cfg, coins, max_age_days, min_live)
    return ok, {"refreshed": True, **report, **detail}


def check_freshness(cfg: dict, coins: List[str], max_age_days: int,
                    min_live: Optional[int] = None) -> tuple:
    """Are the universe's bars recent enough to trade on? Returns (ok, detail).

    ``max_age_days`` is measured from today (UTC) to the newest bar. Today's
    bar is still open and never downloaded, so 1 means "yesterday's close is
    present" — the freshest state reachable on a daily series.

    Two modes:

    * ``min_live=None`` (default) — every name must be fresh. Correct when the
      universe is guaranteed current; the RL configs keep this.
    * ``min_live=N`` — **prune, then gate on what survives**. A universe frozen
      at a past date drifts as coins are delisted from the venue, and under the
      strict rule one dead name red-lights the gate forever. The right question
      is not "is every name in a historical snapshot fresh" but "is the set I
      am about to trade fresh, and is it still big enough" — so dark names are
      dropped from ``detail["live"]`` and the gate checks ``len(live) >= N``.
      The caller must rank only ``live``; ``detail["dropped"]`` is what it must
      exclude.
    """
    raw_dir = Path(cfg["data"]["raw_dir"])
    latest = latest_bar_dates(raw_dir, coins)
    today = _utc_today().date()
    cutoff = today - timedelta(days=max_age_days)

    stale, missing = [], []
    for coin, d in latest.items():
        if d is None:
            missing.append(coin)
        elif datetime.strptime(d[:10], "%Y-%m-%d").date() < cutoff:
            stale.append((coin, d))

    newest = max((d for d in latest.values() if d), default=None)
    dropped = sorted(missing + [c for c, _ in stale])
    live = [c for c in coins if c not in set(dropped)]
    detail = {"latest": latest, "newest": newest, "stale": stale,
              "missing": missing, "cutoff": str(cutoff),
              "live": live, "dropped": dropped, "min_live": min_live}

    if missing:
        print(f"[refresh] NO DATA for: {', '.join(missing)}")
    if stale:
        print("[refresh] STALE: " + ", ".join(f"{c}@{d}" for c, d in stale)
              + f" (need >= {cutoff})")

    if min_live is None:
        ok = not dropped
        print(f"[refresh] freshness ok: newest bar {newest} (cutoff {cutoff})"
              if ok else f"[refresh] gate FAILED: {len(dropped)} dark names")
        return ok, detail

    ok = len(live) >= min_live
    if dropped:
        print(f"[refresh] pruned {len(dropped)} delisted/dark names from the "
              f"universe: {', '.join(dropped)}")
    print(f"[refresh] live universe {len(live)}/{len(coins)} "
          f"(floor {min_live}) — {'ok' if ok else 'TOO THIN, not trading'}, "
          f"newest bar {newest}")
    return ok, detail
