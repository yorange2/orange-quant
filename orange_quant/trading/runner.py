"""
自动交易策略执行器

加载训练好的 LightGBM 模型，每日获取行情数据，
生成预测信号，执行调仓操作。
"""

import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from .broker import BinanceBroker


class StrategyRunner:
    """
    自动交易策略执行器。

    使用方式：

        runner = StrategyRunner(
            broker=broker,
            coins=["BTC", "ETH", "SOL", "BNB", "XRP"],
            topk=5,
            rebalance_interval_hours=24,
        )
        runner.run_once()   # 单次调仓
        # runner.run_loop() # 持续运行
    """

    def __init__(
        self,
        broker: BinanceBroker,
        coins: List[str],
        topk: int = 5,
        lookback_days: int = 30,
        rebalance_interval_hours: int = 24,
        min_trade_usdt: float = 15.0,
        max_position_pct: float = 0.25,
    ):
        """
        Parameters
        ----------
        broker : BinanceBroker
            交易所接口。
        coins : list[str]
            交易币种（不含 USDT 后缀），如 ["BTC", "ETH"]。
        topk : int
            持仓数量，持有排名前 topk 的币种。
        lookback_days : int
            动量计算回看天数。
        rebalance_interval_hours : int
            调仓间隔（小时），默认 24h。
        min_trade_usdt : float
            单笔最小交易金额（USDT），低于此值不交易。
        max_position_pct : float
            单币种最大仓位占比。
        """
        self.broker = broker
        self.coins = coins
        self.symbols = [f"{c}/USDT" for c in coins]
        self.topk = topk
        self.lookback_days = lookback_days
        self.rebalance_interval_hours = rebalance_interval_hours
        self.min_trade_usdt = min_trade_usdt
        self.max_position_pct = max_position_pct

        # 当前持仓 {coin: amount_in_base}
        self.positions: Dict[str, float] = {}
        self.last_rebalance: Optional[datetime] = None

    def compute_signals(self) -> pd.DataFrame:
        """
        计算动量信号。

        对每个币种获取近期 OHLCV，计算动量因子并排名。

        Returns
        -------
        pd.DataFrame
            columns: coin, price, momentum_score, rank
        """
        rows = []
        print(f"[runner] 获取 {len(self.symbols)} 个币种行情数据...")
        for sym, coin in zip(self.symbols, self.coins):
            try:
                df = self.broker.fetch_ohlcv(sym, "1d", limit=self.lookback_days + 5)
                if len(df) < self.lookback_days:
                    print(f"  {coin}: 数据不足 ({len(df)} 天)")
                    continue

                close = df["close"]
                # 动量 = 近期收益率（多周期加权）
                momentum_7d = close.iloc[-1] / close.iloc[-7] - 1 if len(close) >= 7 else 0
                momentum_14d = close.iloc[-1] / close.iloc[-14] - 1 if len(close) >= 14 else 0
                momentum_30d = close.iloc[-1] / close.iloc[-30] - 1 if len(close) >= 30 else 0
                # 波动率调整
                vol = close.pct_change().tail(30).std()

                score = 0.4 * momentum_7d + 0.35 * momentum_14d + 0.25 * momentum_30d
                # 波动率惩罚
                if vol and vol > 0:
                    score = score / (vol * np.sqrt(365))

                rows.append({
                    "coin": coin,
                    "symbol": sym,
                    "price": float(close.iloc[-1]),
                    "momentum_7d": momentum_7d,
                    "momentum_30d": momentum_30d,
                    "score": score,
                })
            except Exception as e:
                print(f"  {coin}: 获取失败 - {e}")

        df = pd.DataFrame(rows)
        if not df.empty:
            df["rank"] = df["score"].rank(ascending=False)
            df = df.sort_values("score", ascending=False)
        return df

    def run_once(self, dry_run: bool = True) -> Dict:
        """
        执行一次调仓。

        Parameters
        ----------
        dry_run : bool
            True = 只分析不交易，False = 实际下单。

        Returns
        -------
        dict
            调仓结果摘要。
        """
        print(f"\n{'='*50}")
        print(f"🔄 调仓检查 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*50}")

        # 1. 获取当前持仓
        balances = self.broker.get_balances()
        usdt_balance = balances.get("USDT", 0.0)
        current_holdings = {
            c: balances.get(c, 0.0)
            for c in self.coins
            if c in balances
        }
        print(f"\n💰 USDT 余额: {usdt_balance:.2f}")
        print(f"📦 当前持仓: {len(current_holdings)} 个币种")
        for coin, amt in current_holdings.items():
            if amt > 0:
                sym = f"{coin}/USDT"
                try:
                    price = self.broker.get_current_prices([sym]).get(sym, 0)
                    val = amt * price
                    print(f"  {coin}: {amt:.4f} (≈${val:.2f})")
                except:
                    print(f"  {coin}: {amt:.4f}")

        # 2. 计算动量排名
        signals = self.compute_signals()
        if signals.empty:
            return {"status": "no_data"}

        print(f"\n📊 动量排名 (Top {self.topk}):")
        for _, row in signals.head(self.topk).iterrows():
            print(f"  {row['rank']:.0f}. {row['coin']:8s}  "
                  f"score={row['score']:.4f}  price=\${row['price']:.4f}")

        # 3. 决定买卖
        target_coins = set(signals.head(self.topk)["coin"])
        current_coins = set(current_holdings.keys())

        to_buy = target_coins - current_coins
        to_sell = current_coins - target_coins

        print(f"\n📋 调仓计划:")
        print(f"  目标持仓: {target_coins}")
        print(f"  买入: {to_buy if to_buy else '无'}")
        print(f"  卖出: {to_sell if to_sell else '无'}")

        trades = []
        if dry_run:
            print(f"\n⚠ DRY RUN — 仅分析，不实际下单")
        else:
            # 卖出
            for coin in to_sell:
                if coin in current_holdings:
                    amt = current_holdings[coin]
                    sym = f"{coin}/USDT"
                    result = self.broker.market_sell(sym, amt)
                    if result:
                        trades.append(("SELL", coin, amt))

            # 买入
            if to_buy:
                n_buy = len(to_buy)
                # 留 5% 作为 buffer
                budget_per_coin = (usdt_balance * 0.95) / n_buy
                budget_per_coin = min(budget_per_coin, usdt_balance * self.max_position_pct)

                for coin in to_buy:
                    if budget_per_coin > self.min_trade_usdt:
                        sym = f"{coin}/USDT"
                        result = self.broker.market_buy(sym, budget_per_coin)
                        if result:
                            trades.append(("BUY", coin, budget_per_coin))

        self.last_rebalance = datetime.now()
        return {
            "status": "ok",
            "dry_run": dry_run,
            "usdt_balance": usdt_balance,
            "target_coins": list(target_coins),
            "signals": signals,
            "trades": trades,
        }

    def run_loop(self, dry_run: bool = True):
        """
        持续运行调仓循环。

        Parameters
        ----------
        dry_run : bool
            True = 模拟运行，不下单。
        """
        print(f"\n🚀 自动交易系统启动")
        print(f"   环境: {'TESTNET (模拟)' if self.broker.testnet else '⚠ MAINNET (实盘)'}")
        print(f"   模式: {'DRY RUN (观察)' if dry_run else '⚠ LIVE (实盘交易)'}")
        print(f"   币种: {len(self.coins)} 个")
        print(f"   持仓数: {self.topk}")
        print(f"   调仓间隔: {self.rebalance_interval_hours}h")
        print(f"   按 Ctrl+C 停止\n")

        while True:
            try:
                result = self.run_once(dry_run=dry_run)
                if result["status"] == "ok":
                    pass
            except Exception as e:
                print(f"[runner] ❌ 调仓异常: {e}")

            # 等待下次调仓
            next_run = datetime.now() + timedelta(hours=self.rebalance_interval_hours)
            print(f"\n⏰ 下次调仓: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   等待 {self.rebalance_interval_hours}h...\n")
            time.sleep(self.rebalance_interval_hours * 3600)
