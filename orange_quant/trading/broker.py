"""Coin-based broker interface shared by every venue.

The core runner/predictor only ever talk to a :class:`Broker` — implemented by
``BinanceBroker``, ``HyperliquidBroker`` and the simulated :class:`PaperBroker`
(``orange_quant.trading.paper_broker``).  The interface is deliberately thin:
coin-based market data + order placement, no venue-specific concepts.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd


class Broker(ABC):
    """Coin-based broker interface (market data + orders).

    All methods take/return coin names without exchange prefix (``"BTC"`` not
    ``"BTC/USDT"``); the quote currency is a venue detail handled by each
    implementation.
    """

    @abstractmethod
    def _verify_connection(self) -> None:
        """Raise if the venue cannot be reached."""

    @abstractmethod
    def get_balances(self) -> Dict[str, float]:
        """Return {coin: free balance} (quote currency included)."""

    @abstractmethod
    def get_current_prices(self, coins: List[str]) -> Dict[str, float]:
        """Return {coin: last price} for the given coins."""

    @abstractmethod
    def fetch_ohlcv(self, coin: str, timeframe: str = "1d", limit: int = 365) -> pd.DataFrame:
        """Return closed OHLCV bars for ``coin`` as a DataFrame with a datetime index."""

    @abstractmethod
    def get_quote_volumes(self, coins: List[str]) -> Dict[str, float]:
        """Return {coin: quote-currency volume over a recent window} (top-symbol ranking)."""

    @abstractmethod
    def get_min_notional(self, coin: str) -> float:
        """Return the venue's minimum order size in quote currency for ``coin``."""

    @abstractmethod
    def market_buy(self, coin: str, amount_quote: float) -> Optional[dict]:
        """Market-buy ``amount_quote`` worth of ``coin``; return the venue order dict or None."""

    @abstractmethod
    def market_sell(self, coin: str, amount: float) -> Optional[dict]:
        """Market-sell ``amount`` units of ``coin``; return the venue order dict or None."""

    @abstractmethod
    def get_open_orders(self, coin: Optional[str] = None) -> list:
        """List open orders, optionally filtered by coin."""

    @abstractmethod
    def cancel_all_orders(self, coin: Optional[str] = None) -> None:
        """Cancel open orders, optionally filtered by coin."""
