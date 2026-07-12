"""
Automated trading strategy runner

Loads a trained LightGBM model, fetches market data daily,
generates predicted signals, and executes rebalancing.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from .broker import HyperliquidBroker, PaperBroker


class StrategyRunner:
    """
    Automated trading strategy runner.

    Usage:

        runner = StrategyRunner(
            broker=broker,
            coins=["BTC", "ETH", "SOL"],
            topk=5,
        )
        runner.run_once()   # single rebalance
    """

    def __init__(
        self,
        broker: Union[HyperliquidBroker, PaperBroker],
        coins: List[str],
        topk: int = 5,
        lookback_days: int = 160,
        rebalance_interval_hours: int = 24,
        min_trade_usdc: float = 20.0,
        max_position_pct: float = 0.25,
        model_path: Optional[str] = None,
    ):
        self.broker = broker
        self.coins = coins
        self.topk = topk
        self.lookback_days = lookback_days
        self.rebalance_interval_hours = rebalance_interval_hours
        self.min_trade_usdc = min_trade_usdc
        self.max_position_pct = max_position_pct
        self.model_path = model_path

        self.positions: Dict[str, float] = {}
        self.last_rebalance: Optional[datetime] = None

        self.predictor = None
        if model_path:
            from .model_predictor import ModelPredictor
            self.predictor = ModelPredictor(model_path)

    def compute_signals(self) -> pd.DataFrame:
        """Compute signals (model takes priority over momentum)"""
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

    def run_once(self, dry_run: bool = True) -> Dict:
        """Execute a single rebalance"""
        print(f"\n{'='*50}")
        print(f"🔄 Rebalance check — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*50}")

        # Get current holdings
        balances = self.broker.get_balances()
        usdc_balance = balances.get("USDC", 0.0)
        current_holdings = {
            c: balances.get(c, 0.0)
            for c in self.coins
            if c in balances
        }

        holding_coins = [c for c, a in current_holdings.items() if a > 0]
        prices = {}
        if holding_coins:
            prices = self.broker.get_current_prices(holding_coins)

        holdings_value = 0.0
        print(f"\n💰 USDC balance: {usdc_balance:.2f}")
        print(f"📦 Current holdings: {len(holding_coins)} coins")
        for coin, amt in current_holdings.items():
            if amt > 0:
                price = prices.get(coin, 0)
                val = amt * price
                holdings_value += val
                print(f"  {coin}: {amt:.4f} (≈${val:.2f})")

        total_equity = usdc_balance + holdings_value
        print(f"💎 Total equity: ${total_equity:,.2f}")

        # Compute ranking
        signals = self.compute_signals()
        if signals.empty:
            return {"status": "no_data"}

        print(f"\n📊 Momentum ranking (Top {self.topk}):")
        for _, row in signals.head(self.topk).iterrows():
            print(f"  {row['rank']:.0f}. {row['coin']:8s}  "
                  f"score={row['score']:.4f}  price=${row['price']:.4f}")

        # Decide buys/sells
        target_coins = set(signals.head(self.topk)["coin"])
        current_coins = {
            c for c, amt in current_holdings.items()
            if amt * prices.get(c, 0) >= self.min_trade_usdc
        }

        to_buy = target_coins - current_coins
        to_sell = current_coins - target_coins

        print(f"\n📋 Rebalance plan:")
        print(f"  Target holdings: {target_coins}")
        print(f"  Buy: {to_buy if to_buy else 'none'}")
        print(f"  Sell: {to_sell if to_sell else 'none'}")

        trades = []
        if dry_run:
            print(f"\n⚠ DRY RUN — analysis only, no orders placed")
        else:
            for coin in to_sell:
                if coin in current_holdings:
                    amt = current_holdings[coin]
                    price = prices.get(coin, 0)
                    # Check whether the minimum trade notional is met (skip dust)
                    min_notional = self.broker.get_min_notional(coin)
                    if amt * price < min_notional:
                        print(f"[runner] ⏭ Skipping sell of {coin} {amt:.6f} (≈${amt*price:.2f}, below minimum ${min_notional})")
                        continue
                    result = self.broker.market_sell(coin, amt)
                    if result:
                        trades.append(("SELL", coin, amt))

            time.sleep(1)
            new_balances = self.broker.get_balances()
            updated_usdc = new_balances.get("USDC", usdc_balance)

            if to_buy:
                n_buy = len(to_buy)
                n_total = len(target_coins)
                if n_total > 0:
                    budget_per_coin = (total_equity * 0.95) / n_total
                else:
                    budget_per_coin = (updated_usdc * 0.95) / n_buy
                budget_per_coin = min(budget_per_coin, total_equity * self.max_position_pct)

                for coin in to_buy:
                    if budget_per_coin > self.min_trade_usdc and updated_usdc >= budget_per_coin:
                        result = self.broker.market_buy(coin, budget_per_coin)
                        if result:
                            trades.append(("BUY", coin, budget_per_coin))
                            updated_usdc -= budget_per_coin

        self.last_rebalance = datetime.now()
        return {
            "status": "ok",
            "dry_run": dry_run,
            "usdc_balance": usdc_balance,
            "target_coins": list(target_coins),
            "signals": signals,
            "trades": trades,
        }

    def run_loop(self, dry_run: bool = True):
        """Run the rebalance loop continuously"""
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
