"""
Binance exchange interface

Connects to Binance via ccxt, providing account queries, market data, and order execution.
"""

import os
from typing import Dict, List, Optional
from datetime import datetime

import ccxt
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class BinanceBroker:
    """
    Binance live trading wrapper.

    Usage:

        broker = BinanceBroker()
        broker.market_buy("BTC/USDT", 100)  # buy 100 USDT worth of BTC
    """

    def __init__(self):
        """Requires the BINANCE_API_KEY / BIANCE_SECRET_KEY environment variables to be set"""
        api_key = os.getenv("BINANCE_API_KEY", "")
        secret_key = os.getenv("BIANCE_SECRET_KEY", "")

        self.exchange = ccxt.binance({"type": "spot",
            "apiKey": api_key,
            "secret": secret_key,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })

        self._verify_connection()

    def _verify_connection(self):
        """Verify the connection"""
        try:
            self.exchange.load_markets()
            print(f"[broker] ✅ Connected to Binance MAINNET")
        except Exception as e:
            print(f"[broker] ❌ Connection failed: {e}")
            raise

    def get_balances(self) -> Dict[str, float]:
        """Get account balances, returns {coin: available balance}"""
        balance = self.exchange.fetch_balance()
        result = {}
        for asset, info in balance["total"].items():
            if info and info > 0:
                result[asset] = info
        return result

    def get_usdt_balance(self) -> float:
        """Get the USDT balance"""
        balances = self.get_balances()
        return balances.get("USDT", 0.0)

    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices, returns {symbol: price}"""
        tickers = self.exchange.fetch_tickers(symbols)
        return {s: t["last"] for s, t in tickers.items() if t.get("last")}

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1d", limit: int = 365) -> pd.DataFrame:
        """
        Fetch OHLCV candle data.
        Returns pd.DataFrame with columns: datetime, open, high, low, close, volume
        """
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(
            ohlcv, columns=["datetime", "open", "high", "low", "close", "volume"]
        )
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
        df.set_index("datetime", inplace=True)
        return df

    def market_buy(self, symbol: str, amount_usdt: float) -> Optional[dict]:
        """Market buy"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker["last"]
            market = self.exchange.market(symbol)
            amount = self.exchange.amount_to_precision(symbol, amount_usdt / price)
            order = self.exchange.create_market_buy_order(symbol, float(amount))
            print(f"[broker] ✅ Bought {symbol} {amount} @ ~{price:.2f} = ~${amount_usdt:.2f}")
            return order
        except Exception as e:
            print(f"[broker] ❌ Failed to buy {symbol}: {e}")
            if "reduce-only" in str(e):
                from . import blacklist
                blacklist.add(symbol.split("/")[0])
            return None

    def market_sell(self, symbol: str, amount: float) -> Optional[dict]:
        """Market sell"""
        try:
            amount = self.exchange.amount_to_precision(symbol, amount)
            order = self.exchange.create_market_sell_order(symbol, float(amount))
            print(f"[broker] ✅ Sold {symbol} {amount}")
            return order
        except Exception as e:
            print(f"[broker] ❌ Failed to sell {symbol}: {e}")
            return None

    def get_open_orders(self, symbol: Optional[str] = None) -> list:
        """Get open (unfilled) orders"""
        return self.exchange.fetch_open_orders(symbol)

    def cancel_all_orders(self, symbol: Optional[str] = None):
        """Cancel all open orders"""
        orders = self.get_open_orders(symbol)
        for o in orders:
            self.exchange.cancel_order(o["id"], o["symbol"])
        print(f"[broker] Cancelled {len(orders)} open orders")


class PaperBroker:
    """
    Simulated exchange (paper trading).

    Uses the public API for market data, simulates the account locally, no orders sent to the exchange.

    Usage:

        broker = PaperBroker(coins=["BTC", "ETH"], initial_usdt=100000)
    """

    def __init__(self, coins: List[str], initial_usdt: float = 100000.0):
        """
        Parameters
        ----------
        coins : list[str]
            Traded coins list (without the USDT suffix).
        initial_usdt : float
            Initial USDT amount.
        """
        self.exchange = ccxt.binance({"type": "spot",
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })

        self._balance = {c: 0.0 for c in coins}
        self._balance["USDT"] = initial_usdt
        self._trades = []

        self._verify_connection()

    def _verify_connection(self):
        """Verify the connection"""
        try:
            self.exchange.load_markets()
            print(f"[broker] ✅ Binance Paper Trading mode (initial ${self._balance.get('USDT', 0):,.0f})")
        except Exception as e:
            print(f"[broker] ❌ Connection failed: {e}")
            raise

    def get_balances(self) -> Dict[str, float]:
        """Get account balances"""
        return {k: v for k, v in self._balance.items() if v > 0}

    def get_usdt_balance(self) -> float:
        """Get the USDT balance"""
        return self._balance.get("USDT", 0.0)

    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices"""
        tickers = self.exchange.fetch_tickers(symbols)
        return {s: t["last"] for s, t in tickers.items() if t.get("last")}

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1d", limit: int = 365) -> pd.DataFrame:
        """Fetch OHLCV candle data"""
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(
            ohlcv, columns=["datetime", "open", "high", "low", "close", "volume"]
        )
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
        df.set_index("datetime", inplace=True)
        return df

    def market_buy(self, symbol: str, amount_usdt: float) -> Optional[dict]:
        """Simulated market buy"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker["last"]
            amount = amount_usdt / price
            coin = symbol.split("/")[0]
            cost = amount * price

            if self._balance.get("USDT", 0) >= cost:
                self._balance["USDT"] -= cost
                self._balance[coin] = self._balance.get(coin, 0) + amount
                self._trades.append({
                    "time": datetime.now(), "side": "BUY", "symbol": symbol,
                    "amount": amount, "price": price, "cost": cost,
                })
            print(f"[broker] 📝 Paper BUY  {symbol} {amount:.6f} @ ${price:.2f} = ${cost:.2f}")
            return {"symbol": symbol, "side": "buy", "amount": amount, "price": price, "status": "paper"}
        except Exception as e:
            print(f"[broker] ❌ Failed to buy {symbol}: {e}")
            return None

    def market_sell(self, symbol: str, amount: float) -> Optional[dict]:
        """Simulated market sell"""
        try:
            coin = symbol.split("/")[0]
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker["last"]

            if self._balance.get(coin, 0) >= amount:
                self._balance[coin] -= amount
                revenue = amount * price
                self._balance["USDT"] = self._balance.get("USDT", 0) + revenue
                self._trades.append({
                    "time": datetime.now(), "side": "SELL", "symbol": symbol,
                    "amount": amount, "price": price, "revenue": revenue,
                })
            print(f"[broker] 📝 Paper SELL {symbol} {amount:.6f} @ ${price:.2f} = ${amount*price:.2f}")
            return {"symbol": symbol, "side": "sell", "amount": amount, "price": price, "status": "paper"}
        except Exception as e:
            print(f"[broker] ❌ Failed to sell {symbol}: {e}")
            return None

    def get_open_orders(self, symbol: Optional[str] = None) -> list:
        """Simulated account has no open orders"""
        return []

    def cancel_all_orders(self, symbol: Optional[str] = None):
        """Simulated account needs no cancellation"""
        pass
