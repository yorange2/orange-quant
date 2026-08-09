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
from typing import Dict, List, Optional

import gym
import numpy as np
import pandas as pd
import torch
from tianshou.data import Batch

from orange_quant.rl.dataset import load_config, bar_reader, _per_stock_features, _FEATURE_COLS
from orange_quant.rl.env import RotationEnv
from orange_quant.rl.network import MultiDiscreteActor, RotationCritic
from orange_quant.rl.policy import MultiDiscretePPO


def _build_policy(cfg: dict, ds) -> MultiDiscretePPO:
    """Rebuild the policy graph from config and load the best checkpoint."""
    from orange_quant.rl.backtest import load_policy as _load

    return _load(cfg, ds, torch.device(cfg["model"]["device"]))


class RLRotationRunner:
    def __init__(self, config_name: str, broker, force: bool = False) -> None:
        self.cfg = load_config(config_name)
        self.broker = broker
        self.force = force
        trading = self.cfg.get("trading", {})
        self.risk_degree = float(trading.get("risk_degree", 0.95))
        self.min_notional_safety = float(trading.get("min_notional_safety", 20.0))
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
            f = _per_stock_features(one)
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
        from orange_quant.rl.dataset import load_or_build

        ds = load_or_build(self.cfg)
        policy = _build_policy(self.cfg, ds)

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
        obs = self._current_obs(ds, today)

        policy.eval()
        with torch.no_grad():
            act = policy(Batch(obs=obs[None])).act[0]
        tiers = np.asarray(act, dtype=np.int64)
        w = RotationEnv._normalize_tiers(
            tiers, np.asarray(self.cfg["env"]["tiers"]),
            self.cfg["env"]["max_weight"]) * self.risk_degree
        target = {ds.codes[i]: float(w[i]) for i in range(ds.n_stocks) if w[i] > 0}

        result = self._execute(target, ds.codes)
        result.update({"date": today, "tiers": tiers.tolist(),
                       "weights": {ds.codes[i]: round(float(w[i]), 4) for i in range(ds.n_stocks)}})
        self.state_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"[live] state written: {self.state_file}")
        return result

    def _execute(self, target_w: Dict[str, float], codes: List[str]) -> dict:
        """Diff target weights against holdings and place market orders."""
        balances = self.broker.get_balances() or {}
        quote = self.cfg["market"]["quote_ccy"]
        prices = self.broker.get_current_prices(codes) or {}
        value = float(balances.get(quote, 0.0)) + sum(
            float(balances.get(c, 0.0)) * prices.get(c, 0.0) for c in codes)
        if value <= 0:
            return {"orders": [], "message": "zero portfolio value"}

        orders: List[dict] = []
        for coin in codes:
            tgt = target_w.get(coin, 0.0) * value
            held = float(balances.get(coin, 0.0)) * prices.get(coin, 0.0)
            delta = tgt - held
            if abs(delta) < self.min_notional_safety:
                continue
            if delta > 0:
                orders.append({"coin": coin, "side": "buy",
                               "amount_quote": round(delta, 2)})
            else:
                qty = -delta / max(prices.get(coin, 1.0), 1e-12)
                orders.append({"coin": coin, "side": "sell", "amount": qty})

        placed = []
        for o in orders:
            try:
                if o["side"] == "buy":
                    r = self.broker.market_buy(o["coin"], o["amount_quote"])
                else:
                    r = self.broker.market_sell(o["coin"], o["amount"])
                placed.append({**o, "result": r})
            except Exception as e:  # noqa: BLE001 - per-order isolation
                placed.append({**o, "error": str(e)})
        print(f"[live] value={value:.2f} {quote}, {len(placed)} orders placed")
        return {"orders": placed, "portfolio_value": round(value, 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description="RL rotation live runner")
    ap.add_argument("--config", required=True, help="config name, e.g. binance-rl-rotation")
    ap.add_argument("--broker", default="paper", choices=["paper", "binance", "hyperliquid"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    quote = cfg["market"]["quote_ccy"]
    if args.broker == "paper":
        from orange_quant.trading.paper_broker import PaperBroker

        if "binance" in cfg["market"]["venue"]:
            from orange_quant.trading.binance_broker import _make_public_exchange as mk
        else:
            from orange_quant.trading.hyperliquid_broker import _make_exchange as mk
        broker = PaperBroker([], quote, mk)
    elif args.broker == "binance":
        from orange_quant.trading.binance_broker import BinanceBroker
        broker = BinanceBroker()
    else:
        from orange_quant.trading.hyperliquid_broker import HyperliquidBroker
        broker = HyperliquidBroker()

    runner = RLRotationRunner(args.config, broker, force=args.force)
    result = runner.run_once()
    print(json.dumps({k: v for k, v in result.items() if k != "orders"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
