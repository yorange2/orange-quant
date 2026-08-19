"""Coin-based broker interface shared by every venue.

The core runner/predictor only ever talk to a :class:`Broker` — implemented by
``BinanceBroker``, ``HyperliquidBroker`` and the simulated :class:`PaperBroker`
(``orange_quant.trading.paper_broker``).  The interface is deliberately thin:
coin-based market data + order placement, no venue-specific concepts.

:class:`CcxtBroker` supplies the market-data methods once for every ccxt-backed
implementation (all three), so venue classes only implement auth + order
placement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd

_DEFAULT_MIN_NOTIONAL = 10.0


class Broker(ABC):
    """Coin-based broker interface (market data + orders).

    All methods take/return coin names without exchange prefix (``"BTC"`` not
    ``"BTC/USDT"``); the quote currency is exposed as ``quote_ccy`` and the
    symbol mapping is a venue detail handled by each implementation.
    """

    #: Settlement currency of the venue (e.g. "USDT" / "USDC"); set by __init__.
    quote_ccy: str

    #: Whether fills are simulated. Live brokers leave this False; the runners
    #: key their idempotency state file off it so a paper run cannot mark the
    #: day done for the live one (and vice versa).
    is_paper: bool = False

    @abstractmethod
    def _verify_connection(self) -> None:
        """Raise if the venue cannot be reached."""

    @abstractmethod
    def get_balances(self) -> Dict[str, float]:
        """Return {coin: free balance} (quote currency included)."""

    @abstractmethod
    def get_free_balances(self) -> Dict[str, float]:
        """Return {coin: free balance} — only settled funds that can be traded.

        Distinct from :meth:`get_balances` for venues that report locked /
        on-order funds (Binance total vs free); venues without locking may
        return the same dict.
        """

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
    def market_buy(self, coin: str, amount_quote: float,
                   price: Optional[float] = None) -> Optional[dict]:
        """Market-buy ``amount_quote`` worth of ``coin``; return the venue order dict or None.

        ``price`` is an optional recent reference price (e.g. from the batch
        ticker fetch the rebalance already did); when omitted the broker
        fetches the ticker itself.
        """

    @abstractmethod
    def market_sell(self, coin: str, amount: float,
                    price: Optional[float] = None) -> Optional[dict]:
        """Market-sell ``amount`` units of ``coin``; return the venue order dict or None."""

    @abstractmethod
    def get_open_orders(self, coin: Optional[str] = None) -> list:
        """List open orders, optionally filtered by coin."""

    @abstractmethod
    def cancel_all_orders(self, coin: Optional[str] = None) -> None:
        """Cancel open orders, optionally filtered by coin."""


class CcxtBroker(Broker):
    """Shared ccxt-backed market-data methods.

    Subclasses must set ``self.exchange`` (a ccxt client) and
    ``self.quote_ccy`` before use.
    """

    def _symbol(self, coin: str) -> str:
        return f"{coin}/{self.quote_ccy}"

    def get_free_balances(self) -> Dict[str, float]:
        """Default: same as :meth:`get_balances`; venues with locked funds override."""
        return self.get_balances()

    @staticmethod
    def _ticker_price(t: dict) -> Optional[float]:
        p = t.get("last") or t.get("close")
        return float(p) if p else None

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
            price = self._ticker_price(t)
            if price:
                result[sym.split("/")[0]] = price
        return result

    def fetch_ohlcv(self, coin: str, timeframe: str = "1d", limit: int = 365) -> pd.DataFrame:
        """Fetch OHLCV; columns: datetime(index), open, high, low, close, volume."""
        duration_ms = self.exchange.parse_timeframe(timeframe) * 1000
        since = self.exchange.milliseconds() - limit * duration_ms
        ohlcv = self.exchange.fetch_ohlcv(self._symbol(coin), timeframe, since=since, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["datetime", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
        df.set_index("datetime", inplace=True)
        return df

    def get_quote_volumes(self, coins: List[str]) -> Dict[str, float]:
        """Get 24h quote volume per coin, returns {coin: volume}."""
        symbols = [self._symbol(c) for c in coins]
        tickers = self.exchange.fetch_tickers(symbols)
        result = {}
        for sym, t in tickers.items():
            result[sym.split("/")[0]] = float(t.get("quoteVolume") or 0)
        return result

    def get_min_notional(self, coin: str) -> float:
        """Get the minimum order value (quote ccy) for a coin's spot pair."""
        try:
            market = self.exchange.market(self._symbol(coin))
            min_cost = market.get("limits", {}).get("cost", {}).get("min")
            return float(min_cost) if min_cost else _DEFAULT_MIN_NOTIONAL
        except Exception:
            return _DEFAULT_MIN_NOTIONAL

    def _reference_price(self, symbol: str, price: Optional[float]) -> Optional[float]:
        """The caller-supplied price, or a fresh ticker fetch when absent."""
        if price:
            return float(price)
        return self._ticker_price(self.exchange.fetch_ticker(symbol))

    def get_open_orders(self, coin: Optional[str] = None) -> list:
        symbol = self._symbol(coin) if coin else None
        return self.exchange.fetch_open_orders(symbol)

    def cancel_all_orders(self, coin: Optional[str] = None):
        orders = self.get_open_orders(coin)
        for o in orders:
            self.exchange.cancel_order(o["id"], o["symbol"])
        print(f"[broker] Cancelled {len(orders)} open orders")
