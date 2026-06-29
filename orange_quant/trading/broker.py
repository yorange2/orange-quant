"""
Binance 交易所接口

通过 ccxt 连接 Binance（支持 testnet 和 mainnet），
提供账户查询、行情获取、订单执行等功能。
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
    Binance 交易所封装。

    使用方式：

        # Testnet
        broker = BinanceBroker(testnet=True)
        print(broker.get_balances())

        # Mainnet（需要改 .env 中的 key 为实盘 key）
        broker = BinanceBroker(testnet=False)
        broker.market_buy("BTC/USDT", 100)  # 买入100 USDT的BTC
    """

    def __init__(self, testnet: bool = True, paper: bool = True):
        """
        Parameters
        ----------
        testnet : bool
            True 使用 Binance 测试网，False 使用实盘。
        paper : bool
            True = paper trading 模式（仅公开 API，不下单）。
            需要真实交易时设为 False。
        """
        api_key = os.getenv("BINANCE_API_KEY", "")
        secret_key = os.getenv("BIANCE_SECRET_KEY", "")

        self.testnet = testnet
        self.paper = paper

        if paper:
            # Paper trading: 用公开 API 获取行情，本地模拟账户
            self.exchange = ccxt.binance({
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            })
            self._paper_balance = {c: 0.0 for c in "BTC ETH SOL BNB XRP DOGE ADA AVAX LINK DOT LTC UNI NEAR AAVE FIL INJ TRX FET XLM ZEC".split()}
            self._paper_balance["USDT"] = 100000.0  # 初始 10万 USDT
            self._paper_trades = []
        else:
            if testnet:
                self.exchange = ccxt.binance({
                    "apiKey": api_key,
                    "secret": secret_key,
                    "enableRateLimit": True,
                    "urls": {
                        "api": {
                            "public": "https://testnet.binance.vision/api/v3",
                            "private": "https://testnet.binance.vision/api/v3",
                        },
                    },
                    "options": {"defaultType": "spot"},
                })
                self.exchange.set_sandbox_mode(True)
            else:
                self.exchange = ccxt.binance({
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
            if self.paper:
                print(f"[broker] ✅ Binance Paper Trading 模式 (初始 ${self._paper_balance.get('USDT', 0):,.0f})")
            else:
                env = "TESTNET" if self.testnet else "MAINNET"
                print(f"[broker] ✅ Binance {env} 连接成功")
        except Exception as e:
            print(f"[broker] ❌ 连接失败: {e}")
            raise

    def get_balances(self) -> Dict[str, float]:
        """
        获取账户余额。

        Returns
        -------
        dict
            {币种: 可用余额}，只返回余额 > 0 的。
        """
        if self.paper:
            return {k: v for k, v in self._paper_balance.items() if v > 0}

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
        """
        获取当前价格。

        Parameters
        ----------
        symbols : list[str]
            交易对列表，如 ["BTC/USDT", "ETH/USDT"]。

        Returns
        -------
        dict
            {symbol: price}
        """
        tickers = self.exchange.fetch_tickers(symbols)
        return {s: t["last"] for s, t in tickers.items() if t.get("last")}

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1d", limit: int = 365) -> pd.DataFrame:
        """
        获取 K 线数据。

        Parameters
        ----------
        symbol : str
            交易对，如 "BTC/USDT"。
        timeframe : str
            K 线周期: "1m", "1h", "1d"。
        limit : int
            获取条数。

        Returns
        -------
        pd.DataFrame
            columns: datetime, open, high, low, close, volume
        """
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(
            ohlcv, columns=["datetime", "open", "high", "low", "close", "volume"]
        )
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
        df.set_index("datetime", inplace=True)
        return df

    def market_buy(self, symbol: str, amount_usdt: float) -> Optional[dict]:
        """
        市价买入。

        Parameters
        ----------
        symbol : str
            交易对，如 "BTC/USDT"。
        amount_usdt : float
            买入金额（USDT）。

        Returns
        -------
        dict or None
            订单信息。
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker["last"]
            amount = amount_usdt / price
            coin = symbol.split("/")[0]

            if self.paper:
                cost = amount * price
                if self._paper_balance.get("USDT", 0) >= cost:
                    self._paper_balance["USDT"] -= cost
                    self._paper_balance[coin] = self._paper_balance.get(coin, 0) + amount
                    self._paper_trades.append({"time": datetime.now(), "side": "BUY", "symbol": symbol, "amount": amount, "price": price, "cost": cost})
                print(f"[broker] 📝 Paper BUY  {symbol} {amount:.6f} @ ${price:.2f} = ${cost:.2f}")
                return {"symbol": symbol, "side": "buy", "amount": amount, "price": price, "status": "paper"}
            else:
                market = self.exchange.market(symbol)
                amount = self.exchange.amount_to_precision(symbol, amount)
                order = self.exchange.create_market_buy_order(symbol, float(amount))
                print(f"[broker] ✅ 买入 {symbol} {amount} @ ~{price:.2f} = ~${amount_usdt:.2f}")
                return order
        except Exception as e:
            print(f"[broker] ❌ 买入 {symbol} 失败: {e}")
            return None

    def market_sell(self, symbol: str, amount: float) -> Optional[dict]:
        """
        市价卖出。

        Parameters
        ----------
        symbol : str
            交易对。
        amount : float
            卖出数量。
        """
        try:
            coin = symbol.split("/")[0]
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker["last"]

            if self.paper:
                if self._paper_balance.get(coin, 0) >= amount:
                    self._paper_balance[coin] -= amount
                    revenue = amount * price
                    self._paper_balance["USDT"] = self._paper_balance.get("USDT", 0) + revenue
                    self._paper_trades.append({"time": datetime.now(), "side": "SELL", "symbol": symbol, "amount": amount, "price": price, "revenue": revenue})
                print(f"[broker] 📝 Paper SELL {symbol} {amount:.6f} @ ${price:.2f} = ${amount*price:.2f}")
                return {"symbol": symbol, "side": "sell", "amount": amount, "price": price, "status": "paper"}
            else:
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
