"""
Hyperliquid 交易所接口

通过 REST API 连接 Hyperliquid 去中心化永续合约交易所，
提供账户查询、行情获取、订单执行等功能。

Hyperliquid API:
    - 公开数据: POST https://api.hyperliquid.xyz/info
    - 交易: 需要使用官方 SDK (hyperliquid-python-sdk)
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
    Hyperliquid 实盘交易所封装。

    使用方式：

        broker = HyperliquidBroker()
        broker.market_buy("BTC", 100)  # 买入100 USDT的BTC永续合约
    """

    def __init__(self):
        """需要设置 HYPERLIQUID_PRIVATE_KEY 环境变量"""
        self.private_key = os.getenv("HYPERLIQUID_PRIVATE_KEY", "")
        self.address = os.getenv("HYPERLIQUID_ADDRESS", "")

        # 使用官方 SDK
        try:
            from hyperliquid.info import Info
            from hyperliquid.exchange import Exchange
            self.info = Info(base_url=_HL_INFO)
            self.exchange = Exchange(
                wallet=None,  # 需要配置 wallet
                base_url="https://api.hyperliquid.xyz",
                private_key=self.private_key,
            )
        except ImportError:
            print("[broker] ⚠ hyperliquid-python-sdk 未安装，仅支持行情查询")
            self.exchange = None

        self._verify_connection()

    def _verify_connection(self):
        """验证连接"""
        try:
            meta = self._api_post({"type": "meta"})
            print(f"[broker] ✅ Hyperliquid MAINNET 连接成功 (永续合约: {len(meta['universe'])} 个)")
        except Exception as e:
            print(f"[broker] ❌ 连接失败: {e}")
            raise

    def _api_post(self, payload: dict) -> dict:
        """发送 POST 请求到 Hyperliquid Info API"""
        resp = requests.post(_HL_INFO, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_balances(self) -> Dict[str, float]:
        """获取账户余额"""
        if not self.address:
            return {"USDC": 0.0}
        try:
            state = self._api_post({"type": "clearinghouseState", "user": self.address})
            return {"USDC": float(state.get("marginSummary", {}).get("accountValue", 0))}
        except Exception as e:
            print(f"[broker] ❌ 获取余额失败: {e}")
            return {"USDC": 0.0}

    def get_usdt_balance(self) -> float:
        """获取 USDC 余额（Hyperliquid 使用 USDC 作为保证金）"""
        return self.get_balances().get("USDC", 0.0)

    def get_current_prices(self, coins: List[str]) -> Dict[str, float]:
        """获取当前中间价"""
        try:
            all_mids = self._api_post({"type": "allMids"})
            return {coin: float(price) for coin, price_str in all_mids.items()
                    if coin in coins and price_str}
        except Exception as e:
            print(f"[broker] ❌ 获取价格失败: {e}")
            return {}

    def fetch_ohlcv(self, coin: str, timeframe: str = "1d", limit: int = 365) -> pd.DataFrame:
        """
        获取 K 线数据。
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
        """市价买入"""
        try:
            if not self.exchange:
                print(f"[broker] ❌ SDK 未安装，无法下单")
                return None
            price = self._get_mid(coin)
            sz = amount_usdc / price
            order = self.exchange.market_open(coin, True, sz)
            print(f"[broker] ✅ 买入 {coin} sz={sz:.4f} @ ~${price:.2f} = ~${amount_usdc:.2f}")
            return order
        except Exception as e:
            print(f"[broker] ❌ 买入 {coin} 失败: {e}")
            return None

    def market_sell(self, coin: str, amount: float) -> Optional[dict]:
        """市价卖出（平仓）"""
        try:
            if not self.exchange:
                print(f"[broker] ❌ SDK 未安装，无法下单")
                return None
            order = self.exchange.market_close(coin, amount)
            print(f"[broker] ✅ 卖出 {coin} sz={amount:.4f}")
            return order
        except Exception as e:
            print(f"[broker] ❌ 卖出 {coin} 失败: {e}")
            return None

    def _get_mid(self, coin: str) -> float:
        """获取单个币种的中间价"""
        return self.get_current_prices([coin]).get(coin, 0)

    def get_open_orders(self) -> list:
        """获取未成交订单"""
        if not self.address:
            return []
        try:
            return self._api_post({"type": "openOrders", "user": self.address})
        except Exception:
            return []

    def cancel_all_orders(self):
        """取消所有未成交订单"""
        if not self.exchange:
            return
        orders = self.get_open_orders()
        for o in orders:
            try:
                self.exchange.cancel(o["coin"], o["oid"])
            except Exception:
                pass
        print(f"[broker] 已取消 {len(orders)} 个挂单")


class PaperBroker:
    """
    模拟交易所（Paper Trading）。

    使用公开 API 获取行情，本地模拟账户。

    使用方式：

        broker = PaperBroker(coins=["BTC", "ETH"], initial_usdc=10000)
    """

    def __init__(self, coins: List[str], initial_usdc: float = 10000.0):
        self.coins = coins
        self._balance = {c: 0.0 for c in coins}
        self._balance["USDC"] = initial_usdc
        self._trades = []

        self._verify_connection()

    def _verify_connection(self):
        """验证连接"""
        try:
            meta = self._api_post({"type": "meta"})
            print(f"[broker] ✅ Hyperliquid Paper Trading 模式 "
                  f"(初始 ${self._balance.get('USDC', 0):,.0f}, {len(meta['universe'])} 个永续合约)")
        except Exception as e:
            print(f"[broker] ❌ 连接失败: {e}")
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
            print(f"[broker] ❌ 获取价格失败: {e}")
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
            print(f"[broker] ❌ 买入 {coin} 失败: {e}")
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
            print(f"[broker] ❌ 卖出 {coin} 失败: {e}")
            return None

    def get_open_orders(self) -> list:
        return []

    def cancel_all_orders(self):
        pass
