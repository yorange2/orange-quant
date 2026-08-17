"""Live RL rotation runner: today's bars → features → policy → tiers → orders.

Market-agnostic execution loop for crypto venues. The policy was trained by
``orange_quant.rl.train``; this module rebuilds the observation with the exact
same feature pipeline and z-score parameters cached by the dataset build, then
converts target tiers into quote-currency target amounts and diffs against
current holdings to place market orders through the injected broker
(``BinanceBroker``/``HyperliquidBroker``/``PaperBroker``).

Idempotency: a state file records the date the strategy last acted on; running
twice on the same date is a no-op unless ``--force``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tianshou.data import Batch

from orange_quant.data.refresh import refresh_and_gate
from orange_quant.rl.dataset import load_config, bar_reader, _per_stock_features, _FEATURE_COLS
from orange_quant.rl.env import RotationEnv


class RLRotationRunner:
    def __init__(self, config_name: str, broker, force: bool = False) -> None:
        self.cfg = load_config(config_name)
        self.broker = broker
        self.force = force
        trading = self.cfg.get("trading", {})
        self.risk_degree = float(trading.get("risk_degree", 0.95))
        self.min_notional_safety = float(trading.get("min_notional_safety", 20.0))
        self.sweep_out_of_universe = bool(trading.get("sweep_out_of_universe", False))
        self.sweep_min_notional = float(trading.get("sweep_min_notional", 5.0))
        self.refresh_data = bool(trading.get("refresh_data", False))
        self.max_bar_age_days = int(trading.get("max_bar_age_days", 2))
        self.state_file = Path(trading.get("state_file", "data/live_state/state.json"))
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ steps
    def _current_obs(self, ds, date: str) -> np.ndarray:
        """Obs for today: [z-scored features, current tiers/3]."""
        read = bar_reader(self.cfg)
        bars = {}
        for j, code in enumerate(ds.codes):
            b = read(code)
            if b is not None:
                bars[code] = b
        today = pd.Timestamp(date)
        feats_raw = np.zeros((len(ds.codes), ds.n_feats), np.float32)
        for j, code in enumerate(ds.codes):
            if code not in bars:
                continue
            one = bars[code][bars[code].index <= today]
            if len(one) < 70:  # need warmup history for the factors
                continue
            # longest factor window is 60 bars — only the tail matters for the
            # last row (same trick as lgb/runner.py's lookback slice)
            f = _per_stock_features(one.iloc[-100:])
            feats_raw[j] = f[_FEATURE_COLS].iloc[-1].fillna(0.0).to_numpy(dtype=np.float32)
        feats_z = np.clip((feats_raw - ds.zmean) / ds.zstd, -3.0, 3.0)
        obs = np.concatenate([feats_z.reshape(-1),
                              self._prev_tiers.astype(np.float32) / 3.0]).astype(np.float32)
        assert obs.shape == (len(ds.codes) * ds.n_feats + len(ds.codes),)
        return obs

    def _today(self) -> str:
        return pd.Timestamp.utcnow().strftime("%Y-%m-%d")

    # ------------------------------------------------------------------ flow
    def run_once(self) -> dict:
        from orange_quant.rl.backtest import load_policy
        from orange_quant.rl.dataset import load_or_build

        ds = load_or_build(self.cfg)
        policy = load_policy(self.cfg, ds, torch.device(self.cfg["model"]["device"]))

        # idempotency: already acted today?
        state = {}
        if self.state_file.exists():
            state = json.loads(self.state_file.read_text())
        today = self._today()
        if state.get("date") == today and not self.force:
            print(f"[live] already executed on {today}, skipping (--force to rerun)")
            return {"skipped": True, "date": today}

        self._prev_tiers = np.asarray(state.get("tiers", [0] * ds.n_stocks),
                                      dtype=np.int64)
        if len(self._prev_tiers) != ds.n_stocks:
            # stale state from a different universe (e.g. after a rebuild) —
            # reset to all-cash rather than crash on the obs shape assert
            print(f"[live] state tiers len {len(self._prev_tiers)} != universe "
                  f"{ds.n_stocks}, resetting to all-cash")
            self._prev_tiers = np.zeros(ds.n_stocks, dtype=np.int64)

        # pull today's missing bars before building the observation — the CSVs
        # are the only price source on the live path, and nothing else updates
        # them. Stale bars mean trading yesterday's signal, so refuse rather
        # than act on them; the state file is left untouched so the next run
        # (or the next day) retries cleanly.
        fresh, refresh_report = refresh_and_gate(
            self.cfg, ds.codes, self.refresh_data, self.max_bar_age_days)
        if not fresh:
            print(f"[live] stale bars, not trading on {today}")
            return {"skipped": True, "reason": "stale_data", "date": today,
                    "refresh": refresh_report}

        obs = self._current_obs(ds, today)

        policy.eval()
        with torch.no_grad():
            act = policy(Batch(obs=obs[None])).act[0]
        tiers = np.asarray(act, dtype=np.int64)
        w = RotationEnv._normalize_tiers(
            tiers, np.asarray(self.cfg["env"]["tiers"]),
            self.cfg["env"]["max_weight"]) * self.risk_degree
        target = {ds.codes[i]: float(w[i]) for i in range(ds.n_stocks) if w[i] > 0}

        result = {}
        if self.sweep_out_of_universe:
            from orange_quant.trading.execute import sweep_out_of_universe as sweep
            result.update(sweep(self.broker, ds.codes, self.sweep_min_notional))
        result.update(self._execute(target, ds.codes))
        result.update({"date": today, "tiers": tiers.tolist(),
                       "weights": {ds.codes[i]: round(float(w[i]), 4) for i in range(ds.n_stocks)}})
        self.state_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"[live] state written: {self.state_file}")
        return result

    def _execute(self, target_w: Dict[str, float], codes: List[str]) -> dict:
        """Diff target weights against holdings and place market orders."""
        from orange_quant.trading.execute import rebalance

        return rebalance(target_w, codes, self.broker, self.min_notional_safety)


def main() -> None:
    ap = argparse.ArgumentParser(description="RL rotation live runner")
    ap.add_argument("--config", required=True, help="config name, e.g. binance-rl-rotation")
    ap.add_argument("--broker", default="paper", choices=["paper", "binance", "hyperliquid"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    from orange_quant.trading import make_broker

    broker = make_broker(cfg, args.broker)
    runner = RLRotationRunner(args.config, broker, force=args.force)
    result = runner.run_once()
    print(json.dumps({k: v for k, v in result.items() if k != "orders"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
