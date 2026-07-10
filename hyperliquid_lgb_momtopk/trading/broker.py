"""
Hyperliquid exchange interface

Connects to the Hyperliquid decentralized spot exchange via its REST API,
providing account queries, market data, and order execution.

Hyperliquid API:
    - Public data: POST https://api.hyperliquid.xyz/info
    - Spot trading: requires the official SDK (hyperliquid-python-sdk)
"""

import os
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

_HL_INFO = "https://api.hyperliquid.xyz/info"


class HyperliquidBroker:
    """
    Hyperliquid live spot trading wrapper.

    Usage:

        broker = HyperliquidBroker()
        broker.market_buy("PURR", 100)  # buy 100 USDC worth of PURR
    """

    def __init__(self):
        """Requires the HYPERLIQUID_PRIVATE_KEY environment variable to be set"""
        self.private_key = os.getenv("HYPERLIQUID_PRIVATE_KEY", "")
        self.address = os.getenv("HYPERLIQUID_ADDRESS", "")

        try:
            from hyperliquid.info import Info
            from hyperliquid.exchange import Exchange
            self.info = Info(base_url=_HL_INFO)
            self.exchange = Exchange(
                wallet=None,
                base_url="https://api.hyperliquid.xyz",
                private_key=self.private_key,
            )
        except ImportError:
            print("[broker] ⚠ hyperliquid-python-sdk not installed, market data only")
            self.exchange = None

        self._verify_connection()

    def _verify_connection(self):
        """Verify the connection"""
        try:
            meta = self._api_post({"type": "spotMeta"})
            print(f"[broker] ✅ Connected to Hyperliquid Spot MAINNET ({len(meta['tokens'])} spot tokens)")
        except Exception as e:
            print(f"[broker] ❌ Connection failed: {e}")
            raise

    def _api_post(self, payload: dict) -> dict:
        """Send a POST request to the Hyperliquid Info API"""
        resp = requests.post(_HL_INFO, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_balances(self) -> Dict[str, float]:
        """Get account balances (spot balances + USDC)"""
        if not self.address:
            return {"USDC": 0.0}
        try:
            # Get spot balances
            balances = self._api_post({"type": "spotClearinghouseState", "user": self.address})
            result = {}
            for b in balances.get("balances", []):
                coin = b.get("coin", "")
                amount = float(b.get("total", 0))
                if amount > 0:
                    result[coin] = amount
            # Also query the USDC balance
            result["USDC"] = float(balances.get("withdrawable", 0))
            return result
        except Exception as e:
            print(f"[broker] ❌ Failed to get balances: {e}")
            return {"USDC": 0.0}

    def get_usdt_balance(self) -> float:
        """Get the USDC balance"""
        return self.get_balances().get("USDC", 0.0)

    def get_current_prices(self, coins: List[str]) -> Dict[str, float]:
        """Get current mid prices"""
        try:
            all_mids = self._api_post({"type": "allMids"})
            return {coin: float(price) for coin, price_str in all_mids.items()
                    if coin in coins and price_str}
        except Exception as e:
            print(f"[broker] ❌ Failed to get prices: {e}")
            return {}

    def fetch_ohlcv(self, coin: str, timeframe: str = "1d", limit: int = 365) -> pd.DataFrame:
        """
        Fetch OHLCV candle data.
        Returns pd.DataFrame with columns: datetime, open, high, low, close, volume
        """
        import time
        end_time = int(time.time() * 1000)
        start_time = end_time - limit * 86400000

        req = {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": timeframe,
                "startTime": start_time,
                "endTime": end_time,
            },
        }
        data = self._api_post(req)
        if not data:
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["t"], unit="ms")
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        df = df[["datetime", "open", "high", "low", "close", "volume"]]
        df.set_index("datetime", inplace=True)
        return df

    def market_buy(self, coin: str, amount_usdc: float) -> Optional[dict]:
        """Market-buy spot (IOC limit order, 1% slippage)"""
        try:
            if not self.exchange:
                print(f"[broker] ❌ SDK not installed, cannot place orders")
                return None
            price = self._get_mid(coin)
            if price <= 0:
                return None
            sz = amount_usdc / price
            # IOC = fill immediately or cancel; limitPx set 1% above mid to ensure a fill
            order = self.exchange.order(
                coin, True, sz,
                {"limit": {"tif": "Ioc"}},
                None,
                round(price * 1.01, 6),
            )
            print(f"[broker] ✅ Bought {coin} sz={sz:.4f} @ ~${price:.2f} = ~${amount_usdc:.2f}")
            return order
        except Exception as e:
            print(f"[broker] ❌ Failed to buy {coin}: {e}")
            return None

    def market_sell(self, coin: str, amount: float) -> Optional[dict]:
        """Market-sell spot (IOC limit order, 1% slippage)"""
        try:
            if not self.exchange:
                print(f"[broker] ❌ SDK not installed, cannot place orders")
                return None
            price = self._get_mid(coin)
            if price <= 0:
                return None
            order = self.exchange.order(
                coin, False, amount,
                {"limit": {"tif": "Ioc"}},
                None,
                round(price * 0.99, 6),
            )
            print(f"[broker] ✅ Sold {coin} sz={amount:.4f} @ ~${price:.2f}")
            return order
        except Exception as e:
            print(f"[broker] ❌ Failed to sell {coin}: {e}")
            return None

    def _get_mid(self, coin: str) -> float:
        """Get the mid price for a single coin"""
        return self.get_current_prices([coin]).get(coin, 0)

    def get_open_orders(self) -> list:
        """Get open (unfilled) orders"""
        if not self.address:
            return []
        try:
            return self._api_post({"type": "openOrders", "user": self.address})
        except Exception:
            return []

    def cancel_all_orders(self):
        """Cancel all open orders"""
        if not self.exchange:
            return
        orders = self.get_open_orders()
        for o in orders:
            try:
                self.exchange.cancel(o["coin"], o["oid"])
            except Exception:
                pass
        print(f"[broker] Cancelled {len(orders)} open orders")


class PaperBroker:
    """
    Simulated exchange (paper trading).

    Uses the public API for market data, simulates the account locally.

    Usage:

        broker = PaperBroker(coins=["BTC", "ETH"], initial_usdc=10000)
    """

    def __init__(self, coins: List[str], initial_usdc: float = 10000.0):
        self.coins = coins
        self._balance = {c: 0.0 for c in coins}
        self._balance["USDC"] = initial_usdc
        self._trades = []

        self._verify_connection()

    def _verify_connection(self):
        """Verify the connection"""
        try:
            meta = self._api_post({"type": "spotMeta"})
            print(f"[broker] ✅ Hyperliquid Paper Trading mode "
                  f"(initial ${self._balance.get('USDC', 0):,.0f}, {len(meta['tokens'])} spot tokens)")
        except Exception as e:
            print(f"[broker] ❌ Connection failed: {e}")
            raise

    def _api_post(self, payload: dict) -> dict:
        resp = requests.post(_HL_INFO, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_balances(self) -> Dict[str, float]:
        return {k: v for k, v in self._balance.items() if v > 0}

    def get_usdt_balance(self) -> float:
        return self._balance.get("USDC", 0.0)

    def get_current_prices(self, coins: List[str]) -> Dict[str, float]:
        try:
            all_mids = self._api_post({"type": "allMids"})
            return {coin: float(price) for coin, price_str in all_mids.items()
                    if coin in coins and price_str}
        except Exception as e:
            print(f"[broker] ❌ Failed to get prices: {e}")
            return {}

    def fetch_ohlcv(self, coin: str, timeframe: str = "1d", limit: int = 365) -> pd.DataFrame:
        import time
        end_time = int(time.time() * 1000)
        start_time = end_time - limit * 86400000

        req = {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": timeframe,
                "startTime": start_time,
                "endTime": end_time,
            },
        }
        data = self._api_post(req)
        if not data:
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["t"], unit="ms")
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        df = df[["datetime", "open", "high", "low", "close", "volume"]]
        df.set_index("datetime", inplace=True)
        return df

    def market_buy(self, coin: str, amount_usdc: float) -> Optional[dict]:
        try:
            price = self.get_current_prices([coin]).get(coin, 0)
            if price <= 0:
                return None
            sz = amount_usdc / price
            cost = sz * price

            if self._balance.get("USDC", 0) >= cost:
                self._balance["USDC"] -= cost
                self._balance[coin] = self._balance.get(coin, 0) + sz
                self._trades.append({
                    "time": datetime.now(), "side": "BUY", "coin": coin,
                    "size": sz, "price": price, "cost": cost,
                })
            print(f"[broker] 📝 Paper BUY  {coin} sz={sz:.4f} @ ${price:.2f} = ${cost:.2f}")
            return {"coin": coin, "side": "buy", "size": sz, "price": price, "status": "paper"}
        except Exception as e:
            print(f"[broker] ❌ Failed to buy {coin}: {e}")
            return None

    def market_sell(self, coin: str, amount: float) -> Optional[dict]:
        try:
            price = self.get_current_prices([coin]).get(coin, 0)
            if self._balance.get(coin, 0) >= amount:
                self._balance[coin] -= amount
                revenue = amount * price
                self._balance["USDC"] = self._balance.get("USDC", 0) + revenue
                self._trades.append({
                    "time": datetime.now(), "side": "SELL", "coin": coin,
                    "size": amount, "price": price, "revenue": revenue,
                })
            print(f"[broker] 📝 Paper SELL {coin} sz={amount:.4f} @ ${price:.2f} = ${amount*price:.2f}")
            return {"coin": coin, "side": "sell", "size": amount, "price": price, "status": "paper"}
        except Exception as e:
            print(f"[broker] ❌ Failed to sell {coin}: {e}")
            return None

    def get_open_orders(self) -> list:
        return []

    def cancel_all_orders(self):
        pass
