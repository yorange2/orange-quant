"""
回测运行器

封装 qlib 的 backtest_loop，提供：
- 交易成本配置（佣金、印花税、滑点）
- 基准比较
- 绩效指标输出（年化收益、夏普比率、最大回撤、IC 等）
"""

from typing import Dict, Optional

import pandas as pd
import numpy as np

from qlib.backtest import backtest, executor as ex
from qlib.utils import flatten_dict


class BacktestRunner:
    """
    回测运行器。

    配置交易成本、执行器、benchmark，运行回测并输出绩效指标。

    使用方式：

        runner = BacktestRunner(
            strategy=strategy,
            start_time="2017-01-01",
            end_time="2020-08-01",
            benchmark="SH000300",
        )
        results = runner.run()
        runner.print_summary(results)
    """

    # 默认交易成本（A股市场）
    DEFAULT_COST_CONFIG = {
        "open_cost": 0.0005,       # 开仓佣金 万五
        "close_cost": 0.0015,      # 平仓佣金 + 印花税 万十五
        "min_cost": 5.0,           # 最低佣金 5元
        "deal_price": "close",     # 以收盘价成交
        "limit_threshold": 0.095,  # 涨跌停限制 9.5%
    }

    def __init__(
        self,
        strategy,
        start_time: str,
        end_time: str,
        benchmark: str = "SH000300",
        codes: str = "all",
        cost_config: Optional[Dict] = None,
        freq: str = "day",
    ):
        """
        Parameters
        ----------
        strategy : BaseStrategy
            交易策略实例。
        start_time : str
            回测开始日期，如 "2017-01-01"。
        end_time : str
            回测结束日期，如 "2020-08-01"。
        benchmark : str
            基准指数代码，默认沪深300。
        codes : str
            交易池，默认 "all"（全市场）。也可用 "csi300"、"csi500"。
        cost_config : dict or None
            交易成本配置，None 使用默认 A 股成本。
        freq : str
            交易频率，默认 "day"（日频）。
        """
        self.strategy = strategy
        self.start_time = start_time
        self.end_time = end_time
        self.benchmark = benchmark
        self.codes = codes
        self.cost_config = cost_config or self.DEFAULT_COST_CONFIG
        self.freq = freq

    def run(self) -> dict:
        """
        运行回测，返回 portfolio_dict 和 indicator_dict。

        Returns
        -------
        dict
            包含 "portfolio" 和 "indicators" 两个 DataFrame 的字典。
        """
        print(f"[orange_quant] 回测: {self.start_time} → {self.end_time}")
        print(f"[orange_quant] 基准: {self.benchmark}, 频率: {self.freq}")

        # 回测配置
        executor_config = {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": self.freq,
                "generate_portfolio_metrics": True,
            },
        }

        # 交易成本配置
        from qlib.backtest.exchange import Exchange
        exchange = Exchange(
            freq=self.freq,
            codes=self.codes,
            **self.cost_config,
        )

        executor = ex.SimulatorExecutor(
            time_per_step=self.freq,
            exchange=exchange,
        )

        # 执行回测
        portfolio_dict, indicator_dict = backtest.backtest_loop(
            start_time=self.start_time,
            end_time=self.end_time,
            trade_strategy=self.strategy,
            trade_executor=executor,
            benchmark=self.benchmark,
        )

        print("[orange_quant] 回测完成！")
        return {
            "portfolio": pd.DataFrame(flatten_dict(portfolio_dict)),
            "indicators": pd.DataFrame(flatten_dict(indicator_dict)),
        }

    def print_summary(self, results: dict) -> None:
        """
        打印回测绩效摘要。

        Parameters
        ----------
        results : dict
            run() 方法返回的结果字典。
        """
        portfolio = results.get("portfolio", pd.DataFrame())
        indicators = results.get("indicators", pd.DataFrame())

        print("\n" + "=" * 60)
        print("📊 回测绩效摘要")
        print("=" * 60)

        if not portfolio.empty:
            # 提取核心指标
            metrics = portfolio.T
            # 尝试计算年化收益、夏普比率、最大回撤
            try:
                # 日收益率 → 年化
                if "return" in metrics.index:
                    daily_return = metrics.loc["return"]
                    annual_return = float(daily_return.mean()) * 252
                    annual_vol = float(daily_return.std()) * np.sqrt(252)
                    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
                    print(f"年化收益率:   {annual_return:.2%}")
                    print(f"年化波动率:   {annual_vol:.2%}")
                    print(f"夏普比率:     {sharpe:.2f}")

                    # 最大回撤
                    cum_return = (1 + daily_return).cumprod()
                    rolling_max = cum_return.expanding().max()
                    drawdown = (cum_return - rolling_max) / rolling_max
                    max_drawdown = float(drawdown.min())
                    print(f"最大回撤:     {max_drawdown:.2%}")
            except Exception as e:
                print(f"(指标计算异常: {e})")

        print("=" * 60 + "\n")
