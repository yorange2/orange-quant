"""
Binance 交易所接口

通过 ccxt 连接 Binance，提供账户查询、行情获取、订单执行等功能。
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
    Binance 实盘交易所封装。

    使用方式：

        broker = BinanceBroker()
        broker.market_buy("BTC/USDT", 100)  # 买入100 USDT的BTC
    """

    def __init__(self):
        """需要设置 BINANCE_API_KEY / BIANCE_SECRET_KEY 环境变量"""
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
        """验证连接"""
        try:
            self.exchange.load_markets()
            print(f"[broker] ✅ Binance MAINNET 连接成功")
        except Exception as e:
            print(f"[broker] ❌ 连接失败: {e}")
            raise

    def get_balances(self) -> Dict[str, float]:
        """获取账户余额，返回 {币种: 可用余额}"""
        balance = self.exchange.fetch_balance()
        result = {}
        for asset, info in balance["total"].items():
            if info and info > 0:
                result[asset] = info
        return result

    def get_usdt_balance(self) -> float:
        """获取 USDT 余额"""
        balances = self.get_balances()
        return balances.get("USDT", 0.0)

    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """获取当前价格，返回 {symbol: price}"""
        tickers = self.exchange.fetch_tickers(symbols)
        return {s: t["last"] for s, t in tickers.items() if t.get("last")}

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1d", limit: int = 365) -> pd.DataFrame:
        """
        获取 K 线数据。
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
        """市价买入"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker["last"]
            market = self.exchange.market(symbol)
            amount = self.exchange.amount_to_precision(symbol, amount_usdt / price)
            order = self.exchange.create_market_buy_order(symbol, float(amount))
            print(f"[broker] ✅ 买入 {symbol} {amount} @ ~{price:.2f} = ~${amount_usdt:.2f}")
            return order
        except Exception as e:
            print(f"[broker] ❌ 买入 {symbol} 失败: {e}")
            return None

    def market_sell(self, symbol: str, amount: float) -> Optional[dict]:
        """市价卖出"""
        try:
            amount = self.exchange.amount_to_precision(symbol, amount)
            order = self.exchange.create_market_sell_order(symbol, float(amount))
            print(f"[broker] ✅ 卖出 {symbol} {amount}")
            return order
        except Exception as e:
            print(f"[broker] ❌ 卖出 {symbol} 失败: {e}")
            return None

    def get_open_orders(self, symbol: Optional[str] = None) -> list:
        """获取未成交订单"""
        return self.exchange.fetch_open_orders(symbol)

    def cancel_all_orders(self, symbol: Optional[str] = None):
        """取消所有未成交订单"""
        orders = self.get_open_orders(symbol)
        for o in orders:
            self.exchange.cancel_order(o["id"], o["symbol"])
        print(f"[broker] 已取消 {len(orders)} 个挂单")


class PaperBroker:
    """
    模拟交易所（Paper Trading）。

    使用公开 API 获取行情，本地模拟账户，不下单到交易所。

    使用方式：

        broker = PaperBroker(coins=["BTC", "ETH"], initial_usdt=100000)
    """

    def __init__(self, coins: List[str], initial_usdt: float = 100000.0):
        """
        Parameters
        ----------
        coins : list[str]
            交易币种列表（不含 USDT 后缀）。
        initial_usdt : float
            初始 USDT 金额。
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
        """验证连接"""
        try:
            self.exchange.load_markets()
            print(f"[broker] ✅ Binance Paper Trading 模式 (初始 ${self._balance.get('USDT', 0):,.0f})")
        except Exception as e:
            print(f"[broker] ❌ 连接失败: {e}")
            raise

    def get_balances(self) -> Dict[str, float]:
        """获取账户余额"""
        return {k: v for k, v in self._balance.items() if v > 0}

    def get_usdt_balance(self) -> float:
        """获取 USDT 余额"""
        return self._balance.get("USDT", 0.0)

    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """获取当前价格"""
        tickers = self.exchange.fetch_tickers(symbols)
        return {s: t["last"] for s, t in tickers.items() if t.get("last")}

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1d", limit: int = 365) -> pd.DataFrame:
        """获取 K 线数据"""
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(
            ohlcv, columns=["datetime", "open", "high", "low", "close", "volume"]
        )
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
        df.set_index("datetime", inplace=True)
        return df

    def market_buy(self, symbol: str, amount_usdt: float) -> Optional[dict]:
        """模拟市价买入"""
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
            print(f"[broker] ❌ 买入 {symbol} 失败: {e}")
            return None

    def market_sell(self, symbol: str, amount: float) -> Optional[dict]:
        """模拟市价卖出"""
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
            print(f"[broker] ❌ 卖出 {symbol} 失败: {e}")
            return None

    def get_open_orders(self, symbol: Optional[str] = None) -> list:
        """模拟账户无挂单"""
        return []

    def cancel_all_orders(self, symbol: Optional[str] = None):
        """模拟账户无需取消"""
        pass
