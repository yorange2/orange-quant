"""
Simulated exchange (paper trading) — exchange-agnostic.

Uses a public ccxt client for market data and simulates the account locally; no
orders are sent to the venue. Parametrized by quote currency and a ccxt-client
factory so a single implementation serves every exchange. The public surface is
the coin-based broker interface the core runner/predictor expect
(``fetch_ohlcv(coin)``, ``get_current_prices(coins)``, ``market_buy(coin, usd)`` …).
"""

from typing import Callable, Dict, List, Optional
from datetime import datetime

import pandas as pd

_DEFAULT_MIN_NOTIONAL = 10.0


class PaperBroker:
    def __init__(
        self,
        coins: List[str],
        quote_ccy: str,
        make_exchange: Callable[[], object],
        initial_cash: float = 100000.0,
    ):
        """
        Parameters
        ----------
        coins : list[str]
            Traded coins (ccxt base codes, without the quote suffix).
        quote_ccy : str
            Settlement currency, e.g. "USDT" or "USDC".
        make_exchange : callable
            Zero-arg factory returning a public ccxt client for the venue.
        initial_cash : float
            Starting quote-currency balance.
        """
        self.quote_ccy = quote_ccy
        self.exchange = make_exchange()

        self._balance = {c: 0.0 for c in coins}
        self._balance[quote_ccy] = initial_cash
        self._trades: list = []

        self._verify_connection()

    def _verify_connection(self):
        try:
            self.exchange.load_markets()
            print(f"[broker] ✅ Paper Trading mode "
                  f"(initial ${self._balance.get(self.quote_ccy, 0):,.0f})")
        except Exception as e:
            print(f"[broker] ❌ Connection failed: {e}")
            raise

    def _symbol(self, coin: str) -> str:
        return f"{coin}/{self.quote_ccy}"

    def get_balances(self) -> Dict[str, float]:
        return {k: v for k, v in self._balance.items() if v > 0}

    def get_quote_balance(self) -> float:
        return self._balance.get(self.quote_ccy, 0.0)

    def get_current_prices(self, coins: List[str]) -> Dict[str, float]:
        symbols = [self._symbol(c) for c in coins]
        tickers = self.exchange.fetch_tickers(symbols)
        result = {}
        for sym, t in tickers.items():
            price = t.get("last") or t.get("close")
            if price:
                result[sym.split("/")[0]] = float(price)
        return result

    def fetch_ohlcv(self, coin: str, timeframe: str = "1d", limit: int = 365) -> pd.DataFrame:
        duration_ms = self.exchange.parse_timeframe(timeframe) * 1000
        since = self.exchange.milliseconds() - limit * duration_ms
        ohlcv = self.exchange.fetch_ohlcv(self._symbol(coin), timeframe, since=since, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["datetime", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
        df.set_index("datetime", inplace=True)
        return df

    def get_quote_volumes(self, coins: List[str]) -> Dict[str, float]:
        symbols = [self._symbol(c) for c in coins]
        tickers = self.exchange.fetch_tickers(symbols)
        result = {}
        for sym, t in tickers.items():
            result[sym.split("/")[0]] = float(t.get("quoteVolume") or 0)
        return result

    def get_min_notional(self, coin: str) -> float:
        try:
            market = self.exchange.market(self._symbol(coin))
            min_cost = market.get("limits", {}).get("cost", {}).get("min")
            return float(min_cost) if min_cost else _DEFAULT_MIN_NOTIONAL
        except Exception:
            return _DEFAULT_MIN_NOTIONAL

    def market_buy(self, coin: str, amount_usd: float) -> Optional[dict]:
        symbol = self._symbol(coin)
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker.get("last") or ticker.get("close")
            amount = amount_usd / price
            cost = amount * price

            if self._balance.get(self.quote_ccy, 0) >= cost:
                self._balance[self.quote_ccy] -= cost
                self._balance[coin] = self._balance.get(coin, 0) + amount
                self._trades.append({
                    "time": datetime.now(), "side": "BUY", "symbol": symbol,
                    "amount": amount, "price": price, "cost": cost,
                })
            print(f"[broker] 📝 Paper BUY  {symbol} {amount:.6f} @ ${price:.4f} = ${cost:.2f}")
            return {"symbol": symbol, "side": "buy", "amount": amount, "price": price, "status": "paper"}
        except Exception as e:
            print(f"[broker] ❌ Failed to buy {coin}: {e}")
            return None

    def market_sell(self, coin: str, amount: float) -> Optional[dict]:
        symbol = self._symbol(coin)
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker.get("last") or ticker.get("close")

            if self._balance.get(coin, 0) >= amount:
                self._balance[coin] -= amount
                revenue = amount * price
                self._balance[self.quote_ccy] = self._balance.get(self.quote_ccy, 0) + revenue
                self._trades.append({
                    "time": datetime.now(), "side": "SELL", "symbol": symbol,
                    "amount": amount, "price": price, "revenue": revenue,
                })
            print(f"[broker] 📝 Paper SELL {symbol} {amount:.6f} @ ${price:.4f} = ${amount*price:.2f}")
            return {"symbol": symbol, "side": "sell", "amount": amount, "price": price, "status": "paper"}
        except Exception as e:
            print(f"[broker] ❌ Failed to sell {coin}: {e}")
            return None

    def get_open_orders(self, coin: Optional[str] = None) -> list:
        return []

    def cancel_all_orders(self, coin: Optional[str] = None):
        pass
