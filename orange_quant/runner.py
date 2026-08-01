"""
Automated trading strategy runner (exchange-agnostic).

Loads a trained LightGBM model, fetches market data daily, generates predicted
signals, and executes rebalancing. Trade behaviour is a strict superset of both
live strategies and is selected by config, so each exchange reproduces its current
decisions exactly:

- Full daily rotation to top-k  <= ``n_drop=None`` (∞) and ``hold_thresh=0``.
- Partial rotation               <= ``n_drop``/``hold_thresh`` from the model YAML.
- Reduce-only blacklist filter    <= ``use_blacklist=True`` (drops coins + substitutes
                                     only for blacklisted names on a rejected buy).
- Liquidity filter                <= ``liquidity_multiple > 0``.

The broker is the coin-based interface (``fetch_ohlcv(coin)``,
``get_current_prices(coins)``, ``market_buy(coin, usd)`` …) implemented by every
adapter's live broker and the shared PaperBroker.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import date, datetime, timedelta

import pandas as pd
import numpy as np

from orange_quant import blacklist


class StrategyRunner:
    def __init__(
        self,
        broker,
        coins: List[str],
        quote_ccy: str,
        provider_uri: str,
        topk: int = 5,
        n_drop: Optional[int] = None,
        hold_thresh: int = 0,
        lookback_days: int = 160,
        rebalance_interval_hours: int = 24,
        min_trade: float = 20.0,
        max_position_pct: float = 0.25,
        risk_degree: float = 0.95,
        liquidity_multiple: float = 0.0,
        use_blacklist: bool = False,
        blacklist_path: Optional[str] = None,
        model_path: Optional[str] = None,
        state_path: str = "data/entry_dates.json",
        cash_threshold: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        n_drop : int or None
            Max positions rotated out per rebalance. ``None`` means unlimited
            (full rotation to top-k).
        hold_thresh : int
            Minimum days a position is held before it may be rotated out.
        liquidity_multiple : float
            A coin's 24h quote volume must be >= this multiple of the per-coin
            budget or it is dropped as too thin. ``0`` disables the check.
        use_blacklist : bool
            Apply the reduce-only blacklist to signals, and (on a rejected buy)
            substitute only for blacklisted coins. When False, substitute for any
            unfillable buy.
        """
        self.broker = broker
        self.coins = coins
        self.quote_ccy = quote_ccy
        self.provider_uri = provider_uri
        self.topk = topk
        self.n_drop = n_drop
        self.hold_thresh = hold_thresh
        self.lookback_days = lookback_days
        self.rebalance_interval_hours = rebalance_interval_hours
        self.min_trade = min_trade
        self.max_position_pct = max_position_pct
        self.risk_degree = risk_degree
        self.liquidity_multiple = liquidity_multiple
        self.use_blacklist = use_blacklist
        self.blacklist_path = blacklist_path
        self.model_path = model_path
        self.state_path = Path(state_path)
        self.cash_threshold = cash_threshold

        self.positions: Dict[str, float] = {}
        self.last_rebalance: Optional[datetime] = None

        self.predictor = None
        if model_path:
            from orange_quant.model_predictor import ModelPredictor
            self.predictor = ModelPredictor(model_path, provider_uri)

    # ----- signals -----

    def compute_signals(self) -> pd.DataFrame:
        """Compute signals (model takes priority over momentum)."""
        if self.predictor is not None:
            return self.predictor.predict(self.broker, self.coins, self.lookback_days)

        rows = []
        print(f"[runner] Fetching market data for {len(self.coins)} coins...")
        for coin in self.coins:
            try:
                df = self.broker.fetch_ohlcv(coin, "1d", limit=self.lookback_days + 5)
                if len(df) < self.lookback_days:
                    print(f"  {coin}: insufficient data ({len(df)} days)")
                    continue

                close = df["close"]
                momentum_7d = close.iloc[-1] / close.iloc[-7] - 1 if len(close) >= 7 else 0
                momentum_14d = close.iloc[-1] / close.iloc[-14] - 1 if len(close) >= 14 else 0
                momentum_30d = close.iloc[-1] / close.iloc[-30] - 1 if len(close) >= 30 else 0
                vol = close.pct_change().tail(30).std()

                score = 0.4 * momentum_7d + 0.35 * momentum_14d + 0.25 * momentum_30d
                if vol and vol > 0:
                    score = score / (vol * np.sqrt(365))

                rows.append({
                    "coin": coin,
                    "price": float(close.iloc[-1]),
                    "momentum_7d": momentum_7d,
                    "momentum_30d": momentum_30d,
                    "score": score,
                })
            except Exception as e:
                print(f"  {coin}: fetch failed - {e}")

        df = pd.DataFrame(rows)
        if not df.empty:
            df["rank"] = df["score"].rank(ascending=False)
            df = df.sort_values("score", ascending=False)
        return df

    # ----- entry-date state (for hold_thresh) -----

    def _load_entry_dates(self) -> Dict[str, str]:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text())
            except Exception as e:
                print(f"[runner] ⚠ Unreadable entry-date state ({e}), treating holdings as opened today")
        return {}

    def _save_entry_dates(self, entries: Dict[str, str]):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(entries, indent=2, sort_keys=True))

    def _held_days(self, entries: Dict[str, str], coin: str, today: date) -> int:
        entered = entries.get(coin)
        if not entered:
            return 0
        try:
            return (today - date.fromisoformat(entered)).days
        except ValueError:
            return 0

    # ----- filters -----

    def _apply_blacklist(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Drop reduce-only assets so their budget goes to the next-ranked coin."""
        if not self.use_blacklist:
            return signals
        excluded = blacklist.load(self.blacklist_path) if self.blacklist_path else blacklist.load()
        if excluded:
            blocked = sorted(set(signals["coin"]) & excluded)
            if blocked:
                print(f"[runner] ⛔ Excluding reduce-only assets: {blocked}")
            signals = signals[~signals["coin"].isin(excluded)]
        return signals

    def _filter_illiquid(self, signals: pd.DataFrame, budget_per_coin: float) -> pd.DataFrame:
        """Drop coins whose 24h volume is too thin to absorb our orders."""
        if self.liquidity_multiple <= 0:
            return signals
        min_vol = budget_per_coin * self.liquidity_multiple
        try:
            volumes = self.broker.get_quote_volumes(list(signals["coin"]))
        except Exception as e:
            print(f"[runner] ⚠ Liquidity check failed ({e}), keeping all coins")
            return signals
        thin = [c for c in signals["coin"] if volumes.get(c, 0) < min_vol]
        if thin:
            print(f"[runner] 💧 Excluding thin markets (24h volume < ${min_vol:,.0f}): {thin}")
        return signals[~signals["coin"].isin(thin)]

    # ----- rebalance -----

    def run_once(self, dry_run: bool = True) -> Dict:
        """Execute a single rebalance."""
        print(f"\n{'='*50}")
        print(f"🔄 Rebalance check — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*50}")

        # Current holdings + total equity
        balances = self.broker.get_balances()
        cash_balance = balances.get(self.quote_ccy, 0.0)
        current_holdings = {
            c: balances.get(c, 0.0)
            for c in self.coins
            if c in balances
        }

        holding_coins = [c for c, a in current_holdings.items() if a > 0]
        prices = {}
        if holding_coins:
            prices = self.broker.get_current_prices(holding_coins)
            # Price-fetch guard: if we hold coins but can't get their prices (e.g.
            # a rate-limit 429 returns {} or zeros), equity is badly under-counted
            # and every downstream budget/order size is wrong. Abort this rebalance
            # rather than trade on bad data — the next scheduled run will retry.
            missing = [c for c in holding_coins if prices.get(c, 0) <= 0]
            if missing:
                print(f"[runner] ❌ Aborting rebalance: no valid price for "
                      f"{len(missing)}/{len(holding_coins)} held coins {missing[:8]} "
                      f"(likely rate-limited). No orders placed.")
                return {"status": "price_error", "missing": missing}

        holdings_value = 0.0
        print(f"\n💰 {self.quote_ccy} balance: {cash_balance:.2f}")
        print(f"📦 Current holdings: {len(holding_coins)} coins")
        for coin, amt in current_holdings.items():
            if amt > 0:
                price = prices.get(coin, 0)
                val = amt * price
                holdings_value += val
                print(f"  {coin}: {amt:.4f} (≈${val:.2f})")

        total_equity = cash_balance + holdings_value
        print(f"💎 Total equity: ${total_equity:,.2f}")

        # Ranking
        signals = self.compute_signals()
        if signals.empty:
            return {"status": "no_data"}

        signals = self._apply_blacklist(signals)
        if signals.empty:
            return {"status": "no_data"}

        # Estimate the per-coin budget for the liquidity check
        budget_est = (total_equity * self.risk_degree) / max(self.topk, 1)
        budget_est = min(budget_est, total_equity * self.max_position_pct)
        signals = self._filter_illiquid(signals, budget_est)
        if signals.empty:
            return {"status": "no_data"}

        print(f"\n📊 Momentum ranking (Top {self.topk}):")
        for _, row in signals.head(self.topk).iterrows():
            print(f"  {row['rank']:.0f}. {row['coin']:8s}  "
                  f"score={row['score']:.4f}  price=${row['price']:.4f}")

        # Decide buys/sells (dust: positions worth < min_trade are treated as not held)
        ranked = list(signals["coin"])
        # Cash floor: only coins scoring above the threshold are eligible to hold;
        # the rest of the top-k budget stays in cash (de-risking when few coins look
        # attractive). None => every coin eligible (fully invested, unchanged).
        if self.cash_threshold is not None:
            score_by = dict(zip(signals["coin"], signals["score"]))
            eligible = [c for c in ranked
                        if score_by.get(c, float("-inf")) > self.cash_threshold]
        else:
            eligible = ranked
        current_coins = {
            c for c, amt in current_holdings.items()
            if amt * prices.get(c, 0) >= self.min_trade
        }

        today = datetime.utcnow().date()
        entries = self._load_entry_dates()
        for coin in current_coins:
            entries.setdefault(coin, today.isoformat())
        entries = {c: d for c, d in entries.items() if c in current_coins}

        # Only holdings that fell out of the target top-k are rotation candidates,
        # worst-ranked first, and only after they have aged past hold_thresh.
        # topk_coins is drawn from the eligible list, so holdings that fell below the
        # cash threshold drop out of the target and become rotation candidates.
        topk_coins = set(eligible[:self.topk])
        rank_of = {c: i for i, c in enumerate(ranked)}
        droppable = sorted(
            (c for c in current_coins if c not in topk_coins),
            key=lambda c: rank_of.get(c, len(ranked)),
            reverse=True,
        )
        held_too_briefly = [
            c for c in droppable
            if self._held_days(entries, c, today) < self.hold_thresh
        ]
        # n_drop None => unlimited (full rotation)
        max_drop = self.n_drop if self.n_drop is not None else len(ranked)
        to_sell = set(
            [c for c in droppable if c not in held_too_briefly][:max_drop]
        )

        # Refill up to topk with the best-ranked ELIGIBLE names we don't already hold.
        # If fewer than topk coins are eligible, the empty slots stay in cash
        # (budget is divided by topk in _execute when a cash floor is set).
        slots = max(self.topk - (len(current_coins) - len(to_sell)), 0)
        to_buy = set([c for c in eligible if c not in current_coins][:slots])
        target_coins = (current_coins - to_sell) | to_buy

        print(f"\n📋 Rebalance plan (topk={self.topk} n_drop={self.n_drop} hold_thresh={self.hold_thresh}d):")
        print(f"  Target holdings: {target_coins}")
        print(f"  Buy: {to_buy if to_buy else 'none'}")
        print(f"  Sell: {to_sell if to_sell else 'none'}")
        if held_too_briefly:
            ages = {c: self._held_days(entries, c, today) for c in held_too_briefly}
            print(f"  ⏳ Held < {self.hold_thresh}d, not rotated yet: {ages}")

        trades = []
        if dry_run:
            print(f"\n⚠ DRY RUN — analysis only, no orders placed")
        else:
            self._execute(to_sell, to_buy, target_coins, current_coins, current_holdings,
                          prices, signals, ranked, total_equity, cash_balance, entries, today, trades)

        self.last_rebalance = datetime.now()
        return {
            "status": "ok",
            "dry_run": dry_run,
            "quote_balance": cash_balance,
            "target_coins": list(target_coins),
            "signals": signals,
            "trades": trades,
        }

    def _execute(self, to_sell, to_buy, target_coins, current_coins, current_holdings,
                 prices, signals, ranked, total_equity, cash_balance, entries, today, trades):
        """Place the sell/trim/buy orders for a live rebalance."""
        for coin in to_sell:
            if coin in current_holdings:
                amt = current_holdings[coin]
                price = prices.get(coin, 0)
                min_notional = self.broker.get_min_notional(coin)
                if amt * price < min_notional:
                    print(f"[runner] ⏭ Skipping sell of {coin} {amt:.6f} "
                          f"(≈${amt*price:.2f}, below minimum ${min_notional})")
                    continue
                if self.broker.market_sell(coin, amt):
                    trades.append(("SELL", coin, amt))

        # Target per-coin budget: deploy risk_degree of equity, equal-weighted.
        # With a cash floor, divide by the full topk (not the number of names held),
        # so unfilled slots stay in cash instead of over-weighting the survivors.
        denom = self.topk if self.cash_threshold is not None else len(target_coins)
        budget_per_coin = (total_equity * self.risk_degree) / max(denom, 1)
        budget_per_coin = min(budget_per_coin, total_equity * self.max_position_pct)

        # Trim held target positions far above budget, freeing cash for new entrants
        for coin in sorted(target_coins & current_coins):
            price = prices.get(coin, 0)
            if price <= 0:
                continue
            val = current_holdings.get(coin, 0) * price
            excess = val - budget_per_coin
            min_notional = self.broker.get_min_notional(coin)
            if val > budget_per_coin * 1.3 and excess >= max(self.min_trade, min_notional):
                if self.broker.market_sell(coin, excess / price):
                    trades.append(("TRIM", coin, excess))

        time.sleep(1)
        new_balances = self.broker.get_balances()
        updated_cash = new_balances.get(self.quote_ccy, cash_balance)

        if to_buy:
            substitutes = [
                c for c in signals["coin"]
                if c not in target_coins and c not in current_coins
            ]
            buy_queue = sorted(to_buy)
            while buy_queue:
                coin = buy_queue.pop(0)
                if budget_per_coin > self.min_trade and updated_cash >= budget_per_coin:
                    if self.broker.market_buy(coin, budget_per_coin):
                        trades.append(("BUY", coin, budget_per_coin))
                        updated_cash -= budget_per_coin
                    elif substitutes and self._should_substitute(coin):
                        sub = substitutes.pop(0)
                        reason = "reduce-only" if self.use_blacklist else "unfillable"
                        print(f"[runner] ↪ Substituting {sub} for {reason} {coin}")
                        buy_queue.append(sub)

        # Age positions from their fill date, so hold_thresh survives restarts
        for action, coin, _ in trades:
            if action == "SELL":
                entries.pop(coin, None)
            elif action == "BUY":
                entries[coin] = today.isoformat()
        self._save_entry_dates(entries)

    def _should_substitute(self, coin: str) -> bool:
        """Binance only falls back for reduce-only rejects; others for any unfillable buy.

        Re-reads the blacklist so a coin the broker just added on a reduce-only
        reject (mid-loop) is picked up, matching the original per-check load.
        """
        if not self.use_blacklist:
            return True
        excluded = blacklist.load(self.blacklist_path) if self.blacklist_path else blacklist.load()
        return coin in excluded

    def run_loop(self, dry_run: bool = True):
        """Run the rebalance loop continuously."""
        print(f"\n🚀 Automated trading system starting")
        print(f"   Environment: MAINNET")
        print(f"   Mode: {'DRY RUN (observe)' if dry_run else '⚠ LIVE (real trading)'}")
        print(f"   Coins: {len(self.coins)}")
        print(f"   Positions held: {self.topk}")
        print(f"   Rebalance interval: {self.rebalance_interval_hours}h")
        print(f"   Press Ctrl+C to stop\n")

        while True:
            try:
                self.run_once(dry_run=dry_run)
            except Exception as e:
                print(f"[runner] ❌ Rebalance error: {e}")

            next_run = datetime.now() + timedelta(hours=self.rebalance_interval_hours)
            print(f"\n⏰ Next rebalance: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Waiting {self.rebalance_interval_hours}h...\n")
            time.sleep(self.rebalance_interval_hours * 3600)
