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

from . import blacklist
from .broker import BinanceBroker, PaperBroker


class StrategyRunner:
    """
    Automated trading strategy runner.

    Usage:

        runner = StrategyRunner(
            broker=broker,
            coins=["BTC", "ETH", "SOL", "BNB", "XRP"],
            topk=5,
            rebalance_interval_hours=24,
        )
        runner.run_once()   # single rebalance
        # runner.run_loop() # run continuously
    """

    def __init__(
        self,
        broker: Union[BinanceBroker, PaperBroker],
        coins: List[str],
        topk: int = 5,
        lookback_days: int = 160,
        rebalance_interval_hours: int = 24,
        min_trade_usdt: float = 15.0,
        max_position_pct: float = 0.25,
        risk_degree: float = 0.95,
        model_path: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        broker : BinanceBroker or PaperBroker
        coins : list[str]
            Traded coins (without the USDT suffix).
        topk : int
            Number of positions to hold.
        lookback_days : int
            Momentum/model lookback window in days.
        rebalance_interval_hours : int
            Rebalance interval, default 24h.
        min_trade_usdt : float
            Minimum trade size per order.
        max_position_pct : float
            Maximum position size as a fraction of equity, per coin.
        risk_degree : float
            Fraction of total equity to deploy (rest stays in USDT),
            mirroring the backtest strategy's risk_degree.
        model_path : str or None
            LightGBM model path. None uses the momentum strategy.
        """
        self.broker = broker
        self.coins = coins
        self.symbols = [f"{c}/USDT" for c in coins]
        self.topk = topk
        self.lookback_days = lookback_days
        self.rebalance_interval_hours = rebalance_interval_hours
        self.min_trade_usdt = min_trade_usdt
        self.max_position_pct = max_position_pct
        self.risk_degree = risk_degree
        self.model_path = model_path

        self.positions: Dict[str, float] = {}
        self.last_rebalance: Optional[datetime] = None

        # Load the model (if provided)
        self.predictor = None
        if model_path:
            from .model_predictor import ModelPredictor
            self.predictor = ModelPredictor(model_path)

    def compute_signals(self) -> pd.DataFrame:
        """
        Compute signals (model takes priority over momentum).

        If a LightGBM model is loaded, use its predictions;
        otherwise use a simple momentum factor ranking.

        Returns
        -------
        pd.DataFrame
            columns: coin, price, score, rank
        """
        # Prefer the model if available
        if self.predictor is not None:
            return self.predictor.predict(self.broker, self.coins, self.lookback_days)
        rows = []
        print(f"[runner] Fetching market data for {len(self.symbols)} coins...")
        for sym, coin in zip(self.symbols, self.coins):
            try:
                df = self.broker.fetch_ohlcv(sym, "1d", limit=self.lookback_days + 5)
                if len(df) < self.lookback_days:
                    print(f"  {coin}: insufficient data ({len(df)} days)")
                    continue

                close = df["close"]
                # Momentum = recent returns (weighted across periods)
                momentum_7d = close.iloc[-1] / close.iloc[-7] - 1 if len(close) >= 7 else 0
                momentum_14d = close.iloc[-1] / close.iloc[-14] - 1 if len(close) >= 14 else 0
                momentum_30d = close.iloc[-1] / close.iloc[-30] - 1 if len(close) >= 30 else 0
                # Volatility adjustment
                vol = close.pct_change().tail(30).std()

                score = 0.4 * momentum_7d + 0.35 * momentum_14d + 0.25 * momentum_30d
                # Volatility penalty
                if vol and vol > 0:
                    score = score / (vol * np.sqrt(365))

                rows.append({
                    "coin": coin,
                    "symbol": sym,
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
        """
        Execute a single rebalance.

        Parameters
        ----------
        dry_run : bool
            True = analyze only without trading, False = place real orders.

        Returns
        -------
        dict
            Summary of the rebalance result.
        """
        print(f"\n{'='*50}")
        print(f"🔄 Rebalance check — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*50}")

        # 1. Get current holdings + compute total equity
        balances = self.broker.get_balances()
        usdt_balance = balances.get("USDT", 0.0)
        current_holdings = {
            c: balances.get(c, 0.0)
            for c in self.coins
            if c in balances
        }

        # Get prices for all held coins, compute total equity
        holding_coins = [c for c, a in current_holdings.items() if a > 0]
        prices = {}
        if holding_coins:
            symbols = [f"{c}/USDT" for c in holding_coins]
            prices = self.broker.get_current_prices(symbols)

        holdings_value = 0.0
        print(f"\n💰 USDT balance: {usdt_balance:.2f}")
        print(f"📦 Current holdings: {len(holding_coins)} coins")
        for coin, amt in current_holdings.items():
            if amt > 0:
                price = prices.get(f"{coin}/USDT", 0)
                val = amt * price
                holdings_value += val
                print(f"  {coin}: {amt:.4f} (≈${val:.2f})")

        total_equity = usdt_balance + holdings_value
        print(f"💎 Total equity: ${total_equity:,.2f}")

        # 2. Compute momentum ranking
        signals = self.compute_signals()
        if signals.empty:
            return {"status": "no_data"}

        # Drop reduce-only assets so their budget goes to the next-ranked coin
        excluded = blacklist.load()
        if excluded:
            blocked = sorted(set(signals["coin"]) & excluded)
            if blocked:
                print(f"[runner] ⛔ Excluding reduce-only assets: {blocked}")
            signals = signals[~signals["coin"].isin(excluded)]
            if signals.empty:
                return {"status": "no_data"}

        print(f"\n📊 Momentum ranking (Top {self.topk}):")
        for _, row in signals.head(self.topk).iterrows():
            print(f"  {row['rank']:.0f}. {row['coin']:8s}  "
                  f"score={row['score']:.4f}  price=${row['price']:.4f}")

        # 3. Decide buys/sells (exclude dust: positions worth less than the min trade size are treated as not held)
        target_coins = set(signals.head(self.topk)["coin"])
        current_coins = {
            c for c, amt in current_holdings.items()
            if amt * prices.get(f"{c}/USDT", 0) >= self.min_trade_usdt
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
            # Sell (skip dust)
            for coin in to_sell:
                if coin in current_holdings:
                    amt = current_holdings[coin]
                    sym = f"{coin}/USDT"
                    price = prices.get(sym, 0)
                    # Check whether the minimum trade notional is met
                    min_notional = _get_min_notional(self.broker.exchange, sym)
                    if amt * price < min_notional:
                        print(f"[runner] ⏭ Skipping sell of {coin} {amt:.6f} (≈${amt*price:.2f}, below minimum ${min_notional})")
                        continue
                    result = self.broker.market_sell(sym, amt)
                    if result:
                        trades.append(("SELL", coin, amt))

            # Target per-coin budget: deploy risk_degree of equity, equal-weighted
            budget_per_coin = (total_equity * self.risk_degree) / max(len(target_coins), 1)
            # Cap each coin at max_position_pct
            budget_per_coin = min(budget_per_coin, total_equity * self.max_position_pct)

            # Trim held target positions that are far above the per-coin budget,
            # freeing cash so new entrants can actually be bought
            for coin in sorted(target_coins & current_coins):
                sym = f"{coin}/USDT"
                price = prices.get(sym, 0)
                if price <= 0:
                    continue
                val = current_holdings.get(coin, 0) * price
                excess = val - budget_per_coin
                min_notional = _get_min_notional(self.broker.exchange, sym)
                if val > budget_per_coin * 1.3 and excess >= max(self.min_trade_usdt, min_notional):
                    result = self.broker.market_sell(sym, excess / price)
                    if result:
                        trades.append(("TRIM", coin, excess))

            # Refresh balance (USDT increases after selling)
            time.sleep(1)
            new_balances = self.broker.get_balances()
            updated_usdt = new_balances.get("USDT", usdt_balance)

            # Buy: size positions based on total equity
            if to_buy:
                # Next-ranked coins to fall back on if a buy is rejected as reduce-only
                substitutes = [
                    c for c in signals["coin"]
                    if c not in target_coins and c not in current_coins
                ]
                buy_queue = sorted(to_buy)
                while buy_queue:
                    coin = buy_queue.pop(0)
                    if budget_per_coin > self.min_trade_usdt and updated_usdt >= budget_per_coin:
                        sym = f"{coin}/USDT"
                        result = self.broker.market_buy(sym, budget_per_coin)
                        if result:
                            trades.append(("BUY", coin, budget_per_coin))
                            updated_usdt -= budget_per_coin
                        elif coin in blacklist.load() and substitutes:
                            sub = substitutes.pop(0)
                            print(f"[runner] ↪ Substituting {sub} for reduce-only {coin}")
                            buy_queue.append(sub)

        self.last_rebalance = datetime.now()
        return {
            "status": "ok",
            "dry_run": dry_run,
            "usdt_balance": usdt_balance,
            "target_coins": list(target_coins),
            "signals": signals,
            "trades": trades,
        }

    def run_loop(self, dry_run: bool = True):
        """
        Run the rebalance loop continuously.

        Parameters
        ----------
        dry_run : bool
            True = simulate only, no orders placed.
        """
        print(f"\n🚀 Automated trading system starting")
        print(f"   Environment: MAINNET")
        print(f"   Mode: {'DRY RUN (observe)' if dry_run else '⚠ LIVE (real trading)'}")
        print(f"   Coins: {len(self.coins)}")
        print(f"   Positions held: {self.topk}")
        print(f"   Rebalance interval: {self.rebalance_interval_hours}h")
        print(f"   Press Ctrl+C to stop\n")

        while True:
            try:
                result = self.run_once(dry_run=dry_run)
                if result["status"] == "ok":
                    pass
            except Exception as e:
                print(f"[runner] ❌ Rebalance error: {e}")

            # Wait for the next rebalance
            next_run = datetime.now() + timedelta(hours=self.rebalance_interval_hours)
            print(f"\n⏰ Next rebalance: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Waiting {self.rebalance_interval_hours}h...\n")
            time.sleep(self.rebalance_interval_hours * 3600)


def _get_min_notional(exchange, symbol: str) -> float:
    """Get the minimum notional value for a trading pair"""
    try:
        market = exchange.market(symbol)
        min_notional = market.get("limits", {}).get("cost", {}).get("min", 0)
        if min_notional is None:
            min_notional = 10.0  # Binance default $10
        return float(min_notional)
    except Exception:
        return 10.0
