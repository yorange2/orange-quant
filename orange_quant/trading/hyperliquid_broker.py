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
from typing import Dict, Optional

import ccxt
from dotenv import load_dotenv

from orange_quant.trading.broker import CcxtBroker

load_dotenv()

_QUOTE = "USDC"
_TIMEOUT_MS = 30000


def _make_exchange(**extra) -> ccxt.hyperliquid:
    return ccxt.hyperliquid({
        "enableRateLimit": True,
        "timeout": _TIMEOUT_MS,
        **extra,
    })


class HyperliquidBroker(CcxtBroker):
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
        self.quote_ccy = _QUOTE

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

    def market_buy(self, coin: str, amount_usdc: float,
                   price: Optional[float] = None) -> Optional[dict]:
        """Market buy (amount specified in USDC notional)"""
        symbol = self._symbol(coin)
        try:
            price = self._reference_price(symbol, price)
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

    def market_sell(self, coin: str, amount: float,
                    price: Optional[float] = None) -> Optional[dict]:
        """Market sell (amount specified in base units)"""
        symbol = self._symbol(coin)
        try:
            price = self._reference_price(symbol, price)
            if not price:
                raise ValueError("no price available")
            amount = float(self.exchange.amount_to_precision(symbol, amount))
            order = self.exchange.create_order(symbol, "market", "sell", amount, price)
            print(f"[broker] ✅ Sold {symbol} {amount}")
            return order
        except Exception as e:
            print(f"[broker] ❌ Failed to sell {coin}: {e}")
            return None
