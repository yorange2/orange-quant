"""Live LGB rotation runner: today's bars → features → model → top-k → orders.

Mirrors ``orange_quant.live.RLRotationRunner``: idempotent per day via a
state file, market orders through the injected broker (BinanceBroker /
PaperBroker), reduce-only blacklist applied to the rankings.

Deliberate deviation from the backtest (same as the legacy live runner): the
backtest rebalances one day after the signal (qlib ``shift=1`` + close fills),
while live trades on the signal day and does FULL daily rotation to top-k —
``n_drop``/``hold_thresh`` from the config are backtest-only knobs.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from orange_quant.data.refresh import refresh_and_gate
from orange_quant.lgb.dataset import load_or_build, load_config
from orange_quant.lgb.features import FEATURE_COLS, alpha158_features
from orange_quant.rl.dataset import bar_reader


class LGBRotationRunner:
    def __init__(self, config_name: str, broker, force: bool = False) -> None:
        self.cfg = load_config(config_name)
        self.broker = broker
        self.force = force
        trading = self.cfg.get("trading", {})
        self.risk_degree = float(trading.get("risk_degree", 0.95))
        self.topk = int(trading.get("topk", 20))
        self.min_trade = float(trading.get("min_trade", 20.0))
        self.max_position_pct = float(trading.get("max_position_pct", 0.25))
        self.lookback = int(trading.get("lookback", 160))
        self.min_notional_safety = float(trading.get("min_notional_safety", 20.0))
        self.sweep_out_of_universe = bool(trading.get("sweep_out_of_universe", False))
        self.sweep_min_notional = float(trading.get("sweep_min_notional", 5.0))
        self.refresh_data = bool(trading.get("refresh_data", False))
        self.max_bar_age_days = int(trading.get("max_bar_age_days", 2))
        self.blacklist_path = trading.get("blacklist")
        self.state_file = Path(trading.get("state_file", "data/live_state/state.json"))
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        self.ds = load_or_build(self.cfg)          # cached codes + splits (cheap)
        with open(Path(self.cfg["paths"]["model_dir"]) / "model.pkl", "rb") as f:
            self.model = pickle.load(f)
        print(f"[lgb-live] loaded model + {len(self.ds.codes)}-coin universe")

    # ------------------------------------------------------------------ steps
    def _features_today(self, date: str) -> np.ndarray:
        """Raw Alpha158 features for the latest closed bar ≤ date, per coin."""
        read = bar_reader(self.cfg)
        today = pd.Timestamp(date)
        X = np.full((len(self.ds.codes), len(FEATURE_COLS)), np.nan, np.float32)
        for j, code in enumerate(self.ds.codes):
            b = read(code)
            if b is None:
                continue
            one = b[b.index <= today]
            if len(one) < 60:                       # feature warmup floor
                continue
            f = alpha158_features(one.iloc[-self.lookback:])
            X[j] = f[FEATURE_COLS].iloc[-1].to_numpy(np.float32)
        return X

    def _target_weights(self, pred: np.ndarray) -> Dict[str, float]:
        """Full daily rotation: top-k by pred desc, equal-weight × risk_degree."""
        from orange_quant.blacklist import load as load_blacklist

        ranked = np.argsort(-pred)                  # best first; -inf ranks last
        codes = self.ds.codes
        black = load_blacklist(self.blacklist_path) if self.blacklist_path else set()
        w = min(self.risk_degree / self.topk, self.max_position_pct)
        target: Dict[str, float] = {}
        for c in ranked:
            coin = codes[c]
            if coin in black or np.isnan(pred[c]):
                continue
            target[coin] = w
            if len(target) >= self.topk:
                break
        return target

    def _today(self) -> str:
        return pd.Timestamp.utcnow().strftime("%Y-%m-%d")

    # ------------------------------------------------------------------ flow
    def run_once(self) -> dict:
        state = {}
        if self.state_file.exists():
            state = json.loads(self.state_file.read_text())
        today = self._today()
        if state.get("date") == today and not self.force:
            print(f"[lgb-live] already executed on {today}, skipping (--force to rerun)")
            return {"skipped": True, "date": today}

        # pull today's missing bars before building features — see the same
        # gate in live.RLRotationRunner.run_once for why stale bars abort the
        # run instead of silently trading yesterday's signal
        fresh, refresh_report = refresh_and_gate(
            self.cfg, self.ds.codes, self.refresh_data, self.max_bar_age_days)
        if not fresh:
            print(f"[lgb-live] stale bars, not trading on {today}")
            return {"skipped": True, "reason": "stale_data", "date": today,
                    "refresh": refresh_report}

        X = self._features_today(today)
        pred = self.model.predict(X)
        pred = np.where(np.isnan(X).all(axis=1), -np.inf, pred)  # no-history coins
        target = self._target_weights(pred)

        from orange_quant.trading.execute import rebalance, sweep_out_of_universe as sweep

        result = {}
        if self.sweep_out_of_universe:
            result.update(sweep(self.broker, self.ds.codes, self.sweep_min_notional))
        result.update(rebalance(target, self.ds.codes, self.broker,
                                self.min_notional_safety))
        result.update({
            "date": today,
            "targets": {k: round(v, 4) for k, v in target.items()},
        })
        self.state_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"[lgb-live] state written: {self.state_file}")
        return result


def main() -> None:
    ap = argparse.ArgumentParser(description="LGB rotation live runner")
    ap.add_argument("--config", required=True, help="config name, e.g. binance-lgb-momtopk")
    ap.add_argument("--broker", default="paper", choices=["paper", "binance"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    from orange_quant.trading import make_broker

    broker = make_broker(cfg, args.broker)
    runner = LGBRotationRunner(args.config, broker, force=args.force)
    result = runner.run_once()
    print(json.dumps({k: v for k, v in result.items() if k != "orders"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
