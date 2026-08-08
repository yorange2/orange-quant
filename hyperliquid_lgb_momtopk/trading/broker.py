"""
Hyperliquid exchange interface

Connects to the Hyperliquid decentralized spot exchange via ccxt,
providing account queries, market data, and order execution.

Coin names follow ccxt's normalized codes (UBTC -> BTC, UETH -> ETH, USOL -> SOL),
and all spot pairs are quoted in USDC.

Auth: set HYPERLIQUID_ADDRESS (main wallet address) and HYPERLIQUID_PRIVATE_KEY
(the private key of the wallet or an API wallet) in the environment / .env file.
"""

import os
from typing import Dict, List, Optional

import ccxt
import pandas as pd
from dotenv import load_dotenv

from orange_quant.trading.broker import Broker
from orange_quant.trading.paper_broker import PaperBroker as _CorePaperBroker

load_dotenv()

_QUOTE = "USDC"
_TIMEOUT_MS = 30000
# Hyperliquid spot minimum order value
_DEFAULT_MIN_NOTIONAL = 10.0


def _make_exchange(**extra) -> ccxt.hyperliquid:
    return ccxt.hyperliquid({
        "enableRateLimit": True,
        "timeout": _TIMEOUT_MS,
        **extra,
    })


class HyperliquidBroker(Broker):
    """
    Hyperliquid live spot trading wrapper.

    Usage:

        broker = HyperliquidBroker()
        broker.market_buy("HYPE", 100)  # buy 100 USDC worth of HYPE
    """

    def __init__(self):
        """Requires the HYPERLIQUID_ADDRESS / HYPERLIQUID_PRIVATE_KEY environment variables to be set"""
        address = os.getenv("HYPERLIQUID_ADDRESS", "")
        private_key = os.getenv("HYPERLIQUID_PRIVATE_KEY", "")

        self.exchange = _make_exchange(
            walletAddress=address,
            privateKey=private_key,
        )

        self._verify_connection()

    def _verify_connection(self):
        """Verify the connection"""
        try:
            markets = self.exchange.load_markets()
            n_spot = sum(1 for m in markets.values() if m.get("spot"))
            print(f"[broker] ✅ Connected to Hyperliquid Spot MAINNET ({n_spot} spot pairs)")
        except Exception as e:
            print(f"[broker] ❌ Connection failed: {e}")
            raise

    def _symbol(self, coin: str) -> str:
        return f"{coin}/{_QUOTE}"

    def get_balances(self) -> Dict[str, float]:
        """Get spot account balances, returns {coin: total balance} (USDC included)"""
        balance = self.exchange.fetch_balance(params={"type": "spot"})
        result = {}
        for asset, amount in balance["total"].items():
            if amount and amount > 0:
                result[asset] = float(amount)
        return result

    def get_usdc_balance(self) -> float:
        """Get the USDC balance"""
        return self.get_balances().get(_QUOTE, 0.0)

    def get_current_prices(self, coins: List[str]) -> Dict[str, float]:
        """Get current prices, returns {coin: price}"""
        symbols = [self._symbol(c) for c in coins]
        try:
            tickers = self.exchange.fetch_tickers(symbols)
        except Exception as e:
            print(f"[broker] ❌ Failed to get prices: {e}")
            return {}
        result = {}
        for sym, t in tickers.items():
            price = t.get("last") or t.get("close")
            if price:
                result[sym.split("/")[0]] = float(price)
        return result

    def fetch_ohlcv(self, coin: str, timeframe: str = "1d", limit: int = 365) -> pd.DataFrame:
        """
        Fetch OHLCV candle data.
        Returns pd.DataFrame with columns: datetime, open, high, low, close, volume
        """
        duration_ms = self.exchange.parse_timeframe(timeframe) * 1000
        since = self.exchange.milliseconds() - limit * duration_ms
        ohlcv = self.exchange.fetch_ohlcv(self._symbol(coin), timeframe, since=since, limit=limit)
        df = pd.DataFrame(
            ohlcv, columns=["datetime", "open", "high", "low", "close", "volume"]
        )
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
        df.set_index("datetime", inplace=True)
        return df

    def get_quote_volumes(self, coins: List[str]) -> Dict[str, float]:
        """Get 24h quote volume (USDC) per coin, returns {coin: volume}"""
        symbols = [self._symbol(c) for c in coins]
        tickers = self.exchange.fetch_tickers(symbols)
        result = {}
        for sym, t in tickers.items():
            result[sym.split("/")[0]] = float(t.get("quoteVolume") or 0)
        return result

    def get_min_notional(self, coin: str) -> float:
        """Get the minimum order value (USDC) for a coin's spot pair"""
        try:
            market = self.exchange.market(self._symbol(coin))
            min_cost = market.get("limits", {}).get("cost", {}).get("min")
            return float(min_cost) if min_cost else _DEFAULT_MIN_NOTIONAL
        except Exception:
            return _DEFAULT_MIN_NOTIONAL

    def market_buy(self, coin: str, amount_usdc: float) -> Optional[dict]:
        """Market buy (amount specified in USDC notional)"""
        symbol = self._symbol(coin)
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker.get("last") or ticker.get("close")
            if not price:
                raise ValueError("no price available")
            amount = float(self.exchange.amount_to_precision(symbol, amount_usdc / price))
            # Hyperliquid market orders need a reference price (slippage is capped around it)
            order = self.exchange.create_order(symbol, "market", "buy", amount, price)
            print(f"[broker] ✅ Bought {symbol} {amount} @ ~{price} = ~${amount_usdc:.2f}")
            return order
        except Exception as e:
            print(f"[broker] ❌ Failed to buy {coin}: {e}")
            return None

    def market_sell(self, coin: str, amount: float) -> Optional[dict]:
        """Market sell (amount specified in base units)"""
        symbol = self._symbol(coin)
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker.get("last") or ticker.get("close")
            if not price:
                raise ValueError("no price available")
            amount = float(self.exchange.amount_to_precision(symbol, amount))
            order = self.exchange.create_order(symbol, "market", "sell", amount, price)
            print(f"[broker] ✅ Sold {symbol} {amount}")
            return order
        except Exception as e:
            print(f"[broker] ❌ Failed to sell {coin}: {e}")
            return None

    def get_open_orders(self, coin: Optional[str] = None) -> list:
        """Get open (unfilled) orders"""
        symbol = self._symbol(coin) if coin else None
        return self.exchange.fetch_open_orders(symbol)

    def cancel_all_orders(self, coin: Optional[str] = None):
        """Cancel all open orders"""
        orders = self.get_open_orders(coin)
        for o in orders:
            self.exchange.cancel_order(o["id"], o["symbol"])
        print(f"[broker] Cancelled {len(orders)} open orders")


def PaperBroker(coins: List[str], initial_usdc: float = 100000.0):
    """Back-compatible factory: the shared paper broker quoted in USDC on Hyperliquid."""
    return _CorePaperBroker(coins, _QUOTE, _make_exchange, initial_cash=initial_usdc)
