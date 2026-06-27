"""
动量 TopK 策略

基于模型预测分数的 TopK 选股策略：
- 持有 topk 只得分最高的股票
- 每交易日卖出得分最低的 n_drop 只，买入得分最高的 n_drop 只
- 参考 qlib 的 TopkDropoutStrategy
"""

from typing import Dict, List, Optional

import pandas as pd
from qlib.backtest.decision import Order, TradeDecisionWO
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy

from .base import BaseQuantStrategy


class MomentumTopKStrategy(BaseQuantStrategy):
    """
    动量 TopK 策略。

    根据模型预测分数排序，持有 topk 只最看好的股票，
    每日淘汰 n_drop 只最低分股票，换入 n_drop 只最高分股票。

    这是最常用的量化选股策略模板，可用于：
    - 单因子选股（如动量、反转、价值）
    - 多因子打分选股
    - ML 模型预测分数选股

    使用方式：

        strategy = MomentumTopKStrategy(
            topk=50,
            n_drop=5,
            hold_thresh=1,
            signal=prediction_signal,  # 可选：外部信号
        )
    """

    def __init__(
        self,
        topk: int = 50,
        n_drop: int = 5,
        hold_thresh: int = 1,
        signal: Optional[pd.DataFrame] = None,
        risk_degree: float = 0.95,
        **kwargs,
    ):
        """
        Parameters
        ----------
        topk : int
            持仓股票数量。
        n_drop : int
            每次换仓数量。
        hold_thresh : int
            最小持有天数。
        signal : pd.DataFrame or None
            外部预测信号。如果为 None，则从 qlib 信号记录中读取。
            格式：MultiIndex (datetime, instrument)，单列分数。
        risk_degree : float
            仓位比例，1.0 表示满仓，默认 0.95。
        """
        super().__init__(topk=topk, n_drop=n_drop, hold_thresh=hold_thresh, **kwargs)
        self.external_signal = signal
        self.risk_degree = risk_degree

        # 延迟初始化 —— 实际回测时由 qlib 注入
        self._inner_strategy: Optional[TopkDropoutStrategy] = None

    def generate_trade_decision(self, execute_result=None):
        """
        生成交易决策。

        将外部配置映射到 qlib 的 TopkDropoutStrategy 上执行。
        """
        if self._inner_strategy is None:
            self._inner_strategy = TopkDropoutStrategy(
                signal=self.external_signal or getattr(self, "signal", "<PRED>"),
                topk=self.topk,
                n_drop=self.n_drop,
                hold_thresh=self.hold_thresh,
                risk_degree=self.risk_degree,
            )
        return self._inner_strategy.generate_trade_decision(execute_result)

    def get_risk_degree(self, execute_result, trade_step, trade_decision):
        """返回仓位比例。"""
        return self.risk_degree
