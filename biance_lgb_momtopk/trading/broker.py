"""
Binance exchange interface (coin-based).

Connects to Binance via ccxt, providing account queries, market data, and order
execution. Exposes the shared coin-based broker interface the core runner/predictor
expect: methods take a bare coin code (e.g. "BTC") and the "/USDT" quoting is
internal.
"""

import os
from typing import Dict, List, Optional

import ccxt
import pandas as pd
from dotenv import load_dotenv

from orange_quant import blacklist
from orange_quant.trading.broker import Broker
from orange_quant.trading.paper_broker import PaperBroker as _CorePaperBroker

load_dotenv()

_QUOTE = "USDT"
_DEFAULT_MIN_NOTIONAL = 10.0  # Binance spot default $10
BLACKLIST_PATH = "data/reduce_only_blacklist.json"


def _make_public_exchange() -> ccxt.binance:
    return ccxt.binance({
        "type": "spot",
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })


class BinanceBroker(Broker):
    """Binance live spot trading wrapper (coin-based interface)."""

    def __init__(self):
        """Requires BINANCE_API_KEY / BIANCE_SECRET_KEY in the environment / .env."""
        api_key = os.getenv("BINANCE_API_KEY", "")
        secret_key = os.getenv("BIANCE_SECRET_KEY", "")

        self.exchange = ccxt.binance({
            "type": "spot",
            "apiKey": api_key,
            "secret": secret_key,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        self.quote_ccy = _QUOTE
        self._verify_connection()

    def _verify_connection(self):
        try:
            self.exchange.load_markets()
            print(f"[broker] ✅ Connected to Binance MAINNET")
        except Exception as e:
            print(f"[broker] ❌ Connection failed: {e}")
            raise

    def _symbol(self, coin: str) -> str:
        return f"{coin}/{_QUOTE}"

    def get_balances(self) -> Dict[str, float]:
        """Get account balances, returns {coin: total balance} (USDT included)."""
        balance = self.exchange.fetch_balance()
        result = {}
        for asset, info in balance["total"].items():
            if info and info > 0:
                result[asset] = float(info)
        return result

    def get_current_prices(self, coins: List[str]) -> Dict[str, float]:
        """Get current prices, returns {coin: price}."""
        symbols = [self._symbol(c) for c in coins]
        try:
            tickers = self.exchange.fetch_tickers(symbols)
        except Exception as e:
            print(f"[broker] ❌ Failed to get prices: {e}")
            return {}
        result = {}
        for sym, t in tickers.items():
            if t.get("last"):
                result[sym.split("/")[0]] = float(t["last"])
        return result

    def fetch_ohlcv(self, coin: str, timeframe: str = "1d", limit: int = 365) -> pd.DataFrame:
        """Fetch OHLCV; columns: datetime(index), open, high, low, close, volume."""
        ohlcv = self.exchange.fetch_ohlcv(self._symbol(coin), timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["datetime", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
        df.set_index("datetime", inplace=True)
        return df

    def get_quote_volumes(self, coins: List[str]) -> Dict[str, float]:
        """Get 24h quote volume (USDT) per coin, returns {coin: volume}."""
        symbols = [self._symbol(c) for c in coins]
        tickers = self.exchange.fetch_tickers(symbols)
        result = {}
        for sym, t in tickers.items():
            result[sym.split("/")[0]] = float(t.get("quoteVolume") or 0)
        return result

    def get_min_notional(self, coin: str) -> float:
        """Get the minimum order value (USDT) for a coin's spot pair."""
        try:
            market = self.exchange.market(self._symbol(coin))
            min_notional = market.get("limits", {}).get("cost", {}).get("min")
            return float(min_notional) if min_notional else _DEFAULT_MIN_NOTIONAL
        except Exception:
            return _DEFAULT_MIN_NOTIONAL

    def market_buy(self, coin: str, amount_usdt: float) -> Optional[dict]:
        """Market buy (amount specified in USDT notional)."""
        symbol = self._symbol(coin)
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker["last"]
            amount = self.exchange.amount_to_precision(symbol, amount_usdt / price)
            order = self.exchange.create_market_buy_order(symbol, float(amount))
            print(f"[broker] ✅ Bought {symbol} {amount} @ ~{price:.2f} = ~${amount_usdt:.2f}")
            return order
        except Exception as e:
            print(f"[broker] ❌ Failed to buy {coin}: {e}")
            if "reduce-only" in str(e):
                blacklist.add(coin, BLACKLIST_PATH)
            return None

    def market_sell(self, coin: str, amount: float) -> Optional[dict]:
        """Market sell (amount specified in base units)."""
        symbol = self._symbol(coin)
        try:
            amount = self.exchange.amount_to_precision(symbol, amount)
            order = self.exchange.create_market_sell_order(symbol, float(amount))
            print(f"[broker] ✅ Sold {symbol} {amount}")
            return order
        except Exception as e:
            print(f"[broker] ❌ Failed to sell {coin}: {e}")
            return None

    def get_open_orders(self, coin: Optional[str] = None) -> list:
        symbol = self._symbol(coin) if coin else None
        return self.exchange.fetch_open_orders(symbol)

    def cancel_all_orders(self, coin: Optional[str] = None):
        orders = self.get_open_orders(coin)
        for o in orders:
            self.exchange.cancel_order(o["id"], o["symbol"])
        print(f"[broker] Cancelled {len(orders)} open orders")


def PaperBroker(coins: List[str], initial_usdt: float = 100000.0):
    """Back-compatible factory: the shared paper broker quoted in USDT on Binance."""
    return _CorePaperBroker(coins, _QUOTE, _make_public_exchange, initial_cash=initial_usdt)
