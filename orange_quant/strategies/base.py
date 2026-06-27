"""
策略基类

定义量化交易策略的统一接口。
自定义策略只需继承此基类，实现 generate_trade_decision 方法即可。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import pandas as pd
from qlib.backtest.decision import BaseTradeDecision, Order, TradeDecisionWO
from qlib.strategy.base import BaseStrategy


class BaseQuantStrategy(BaseStrategy, ABC):
    """
    量化策略抽象基类。

    封装了 qlib BaseStrategy 的通用逻辑，子类只需关注信号生成和下单逻辑。

    使用方式：

        class MyStrategy(BaseQuantStrategy):
            def generate_trade_decision(self, execute_result=None):
                # 1. 获取当日预测信号
                # 2. 计算目标持仓
                # 3. 生成订单
                order_list = [
                    Order(stock_id="SH600000", amount=100, direction=Order.BUY),
                    Order(stock_id="SH600004", amount=50, direction=Order.SELL),
                ]
                return TradeDecisionWO(order_list, self)

    Attributes
    ----------
    topk : int
        持仓股票数量。
    n_drop : int
        每日换仓数量。
    hold_thresh : int
        最小持有天数（避免频繁交易）。
    commission : float
        交易佣金费率。
    slippage : float
        滑点比例。
    """

    def __init__(
        self,
        topk: int = 50,
        n_drop: int = 5,
        hold_thresh: int = 1,
        commission: float = 0.001,
        slippage: float = 0.001,
        **kwargs,
    ):
        """
        Parameters
        ----------
        topk : int
            持仓股票数量上限。
        n_drop : int
            每次调仓卖出/买入的股票数量。
        hold_thresh : int
            最短持有天数，持有不足此天数的股票不会被卖出。
        commission : float
            交易佣金费率（单向），默认千分之一。
        slippage : float
            滑点比例，默认千分之一。
        """
        super().__init__(**kwargs)
        self.topk = topk
        self.n_drop = n_drop
        self.hold_thresh = hold_thresh
        self.commission = commission
        self.slippage = slippage

    @abstractmethod
    def generate_trade_decision(
        self, execute_result: Optional[dict] = None
    ) -> Union[BaseTradeDecision, List[Order]]:
        """
        生成交易决策。

        每个交易日调用一次。根据当前持仓、信号和市场状态，返回需要执行的订单。

        Parameters
        ----------
        execute_result : dict or None
            上一个交易日的执行结果反馈。首日为 None。

        Returns
        -------
        TradeDecisionWO 或 Order 列表
            包含当日待执行订单的决策对象。
        """
        ...

    def get_risk_degree(self, execute_result, trade_step, trade_decision):
        """默认风险等级为中性。"""
        return 0.0
