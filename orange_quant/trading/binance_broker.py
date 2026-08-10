"""
Binance exchange interface (coin-based).

Connects to Binance via ccxt, providing account queries, market data, and order
execution. Exposes the shared coin-based broker interface the core runner/predictor
expect: methods take a bare coin code (e.g. "BTC") and the "/USDT" quoting is
internal.
"""

import os
from typing import Dict, Optional

import ccxt
from dotenv import load_dotenv

from orange_quant import blacklist
from orange_quant.trading.broker import CcxtBroker

load_dotenv()

_QUOTE = "USDT"
BLACKLIST_PATH = "data/reduce_only_blacklist.json"


def _make_public_exchange() -> ccxt.binance:
    return ccxt.binance({
        "type": "spot",
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })


class BinanceBroker(CcxtBroker):
    """Binance live spot trading wrapper (coin-based interface)."""

    def __init__(self, blacklist_path: str = BLACKLIST_PATH):
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
        self.blacklist_path = blacklist_path
        self._verify_connection()

    def _verify_connection(self):
        try:
            self.exchange.load_markets()
            print(f"[broker] ✅ Connected to Binance MAINNET")
        except Exception as e:
            print(f"[broker] ❌ Connection failed: {e}")
            raise

    def get_balances(self) -> Dict[str, float]:
        """Get account balances, returns {coin: total balance} (USDT included)."""
        balance = self.exchange.fetch_balance()
        result = {}
        for asset, info in balance["total"].items():
            if info and info > 0:
                result[asset] = float(info)
        return result

    def market_buy(self, coin: str, amount_usdt: float,
                   price: Optional[float] = None) -> Optional[dict]:
        """Market buy (amount specified in USDT notional)."""
        symbol = self._symbol(coin)
        try:
            price = self._reference_price(symbol, price)
            amount = self.exchange.amount_to_precision(symbol, amount_usdt / price)
            order = self.exchange.create_market_buy_order(symbol, float(amount))
            print(f"[broker] ✅ Bought {symbol} {amount} @ ~{price:.2f} = ~${amount_usdt:.2f}")
            return order
        except Exception as e:
            print(f"[broker] ❌ Failed to buy {coin}: {e}")
            if "reduce-only" in str(e):
                blacklist.add(coin, self.blacklist_path)
            return None

    def market_sell(self, coin: str, amount: float,
                    price: Optional[float] = None) -> Optional[dict]:
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
