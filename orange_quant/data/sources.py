"""Exchange data-source hooks: venue-specific fetch logic only.

Everything else (incremental download loop, qlib-binary build, phase
resampling) lives in ``orange_quant.data.pipeline`` / ``orange_quant.data.hourly``
and is exchange-agnostic. Each venue contributes a :class:`DataSourceHooks`
subclass supplying the same uniform rows
``[timestamp_ms, open, high, low, close, volume]`` (closed bars only).
"""

import os
import time
from pathlib import Path

import requests

from orange_quant.data import pipeline

_REQUEST_DELAY = 0.3


class DataSourceHooks:
    """Venue-specific hooks consumed by the shared build pipeline."""

    label: str
    raw_dir: Path
    h1_raw_dir: Path

    def get_top_symbols(self, n: int = 50) -> list:
        """Return [(symbol, coin)] ranked by quote volume (live 24h snapshot)."""
        raise NotImplementedError

    def fetch_daily(self, symbol: str, start_ms: int, end_ms: int) -> list:
        """Fetch daily bars as uniform rows (closed bars only)."""
        raise NotImplementedError

    # -- pipeline wiring -----------------------------------------------------

    def build_source(self) -> pipeline.DataSource:
        return pipeline.DataSource(
            label=self.label,
            raw_dir=self.raw_dir,
            h1_raw_dir=getattr(self, "h1_raw_dir", None),
            get_top_symbols=self.get_top_symbols,
            fetch_daily=self.fetch_daily,
            fetch_hourly=getattr(self, "fetch_hourly", None),
            fallback_coins=self.fallback_coins,
        )


class BinanceSource(DataSourceHooks):
    """Binance USDT spot: REST kline API, top-N by 24h quote volume."""

    label = "Binance"
    raw_dir = Path("data/binance_raw")
    h1_raw_dir = Path("data/binance_h1_raw")

    _BINANCE_API = "https://api.binance.com/api/v3"
    _SKIP = {
        "USDCUSDT", "USDTUSDT", "TUSDUSDT", "BUSDUSDT", "DAIUSDT",
        "PAXUSDT", "USD1USDT", "FDUSDUSDT", "RLUSDUSDT", "EURUSDT",
        "XAUTUSDT", "PAXGUSDT",
        "UUSDT",  # trade-restricted on Binance (reduce-only), orders get rejected
    }
    fallback_coins = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX",
                      "LINK", "DOT", "LTC", "UNI", "NEAR", "AAVE", "FIL", "INJ",
                      "TRX", "FET", "XLM", "ZEC"]

    def get_top_symbols(self, n: int = 50) -> list:
        """Top-N USDT spot pairs by 24h quote volume. Returns [(symbol, coin)]."""
        tickers = requests.get(f"{self._BINANCE_API}/ticker/24hr", timeout=10).json()
        usdt = [(t["symbol"], float(t["quoteVolume"]))
                for t in tickers if t["symbol"].endswith("USDT")]
        usdt.sort(key=lambda x: x[1], reverse=True)

        result = []
        for symbol, _vol in usdt:
            base = symbol.replace("USDT", "")
            if symbol in self._SKIP:
                continue
            if any(x in base for x in ("UP", "DOWN", "BULL", "BEAR")):
                continue
            result.append((symbol, base))
            if len(result) >= n:
                break
        return result

    def _fetch_klines(self, symbol: str, interval: str, step_ms: int,
                      start_ms: int, end_ms: int, retries: int = 3) -> list:
        """Fetch klines from the Binance API (auto-paginated, retried)."""
        all_candles = []
        batch_start = start_ms
        while batch_start < end_ms:
            params = {"symbol": symbol, "interval": interval,
                      "startTime": batch_start, "endTime": end_ms, "limit": 1000}
            data = None
            for attempt in range(retries):
                try:
                    resp = requests.get(f"{self._BINANCE_API}/klines", params=params, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except Exception as e:
                    if attempt == retries - 1:
                        print(f"  API err: {symbol} {interval} page at {batch_start} "
                              f"failed after {retries} attempts: {e}")
                    else:
                        time.sleep(_REQUEST_DELAY * (attempt + 1) * 5)
            if data is None:
                break
            if not data or not isinstance(data, list):
                break
            all_candles.extend(data)
            last_time = data[-1][0]
            if last_time <= batch_start:
                break
            batch_start = last_time + step_ms
            time.sleep(_REQUEST_DELAY)

        # Keep only closed bars, reshape kline -> uniform [ts, o, h, l, c, v]
        now_ms = int(time.time() * 1000)
        return [[c[0], c[1], c[2], c[3], c[4], c[5]] for c in all_candles if c[6] <= now_ms]

    def fetch_daily(self, symbol: str, start_ms: int, end_ms: int) -> list:
        return self._fetch_klines(symbol, "1d", 86400000, start_ms, end_ms)

    def fetch_hourly(self, symbol: str, start_ms: int, end_ms: int) -> list:
        return self._fetch_klines(symbol, "1h", 3600000, start_ms, end_ms)


class HyperliquidSource(DataSourceHooks):
    """Hyperliquid spot (USDC): ccxt candles, liquidity-filtered universe."""

    label = "Hyperliquid"
    raw_dir = Path("data/hyperliquid_raw")
    h1_raw_dir = Path("data/hyperliquid_h1_raw")

    _SKIP_BASES = {"USDT", "USDE", "USDH", "USDHL", "FEUSD", "USR", "DAI", "BUIDL", "USDXL"}
    fallback_coins = ["HYPE", "PURR", "BTC", "ETH", "SOL"]

    # Liquidity floor: drop coins whose 24h USDC quote volume is below this.
    # Most of Hyperliquid spot is near-zero-volume zombie pairs (median ~$900/day).
    _MIN_QUOTE_VOLUME = float(os.environ.get("HL_MIN_QUOTE_VOLUME", "25000"))
    # Safety floor so a market-wide volume dip can never collapse the universe.
    _MIN_UNIVERSE = int(os.environ.get("HL_MIN_UNIVERSE", "15"))

    def __init__(self):
        self._exchange = None

    def _get_exchange(self):
        if self._exchange is None:
            import ccxt
            self._exchange = ccxt.hyperliquid({"enableRateLimit": True})
        return self._exchange

    def get_top_symbols(self, n: int = 50) -> list:
        """Top-N spot pairs by 24h quote volume (liquidity floor applied)."""
        ex = self._get_exchange()
        tickers = ex.fetch_tickers()
        pairs = []
        for sym, t in tickers.items():
            base = sym.split("/")[0]
            if base in self._SKIP_BASES or "/" not in sym:
                continue
            vol = float(t.get("quoteVolume") or 0)
            pairs.append((sym, base, vol))
        pairs.sort(key=lambda x: x[2], reverse=True)

        result = [(sym, base) for sym, base, _vol in pairs if _vol >= self._MIN_QUOTE_VOLUME]
        if len(result) < self._MIN_UNIVERSE:  # volume dip: keep top-N regardless
            result = [(sym, base) for sym, base, _vol in pairs[: max(n, self._MIN_UNIVERSE)]]
        return result[:n]

    def _fetch_ohlcv(self, symbol: str, timeframe: str, step_ms: int,
                     start_ms: int, end_ms: int) -> list:
        ex = self._get_exchange()
        all_rows = []
        batch_start = start_ms
        while batch_start < end_ms:
            try:
                rows = ex.fetch_ohlcv(symbol, timeframe, since=batch_start, limit=5000)
            except Exception as e:
                print(f"  API err: {e}")
                break
            if not rows:
                break
            all_rows.extend(rows)
            last_time = rows[-1][0]
            if last_time <= batch_start:
                break
            batch_start = last_time + step_ms
            time.sleep(_REQUEST_DELAY)

        now_ms = int(time.time() * 1000)
        return [r for r in all_rows if r[0] + step_ms <= now_ms]

    def fetch_daily(self, symbol: str, start_ms: int, end_ms: int) -> list:
        return self._fetch_ohlcv(symbol, "1d", 86400000, start_ms, end_ms)

    def fetch_hourly(self, symbol: str, start_ms: int, end_ms: int) -> list:
        # NOTE: Hyperliquid only retains ~5000 hourly candles (~208 days). Phase
        # datasets built from this cover a recent window only; use Binance for
        # the phase study. Kept for recent-window sanity checks on HL.
        return self._fetch_ohlcv(symbol, "1h", 3600000, start_ms, end_ms)
