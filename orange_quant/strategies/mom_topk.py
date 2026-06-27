"""
动量 TopK 策略

基于模型预测分数的 TopK 选股策略：
- 持有 topk 只得分最高的股票
- 每交易日卖出得分最低的 n_drop 只，买入得分最高的 n_drop 只
- 直接继承 qlib 的 TopkDropoutStrategy，确保回测框架完全兼容
"""

from typing import Optional

import pandas as pd
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy


class MomentumTopKStrategy(TopkDropoutStrategy):
    """
    动量 TopK 策略。

    根据模型预测分数排序，持有 topk 只最看好的股票，
    每日淘汰 n_drop 只最低分股票，换入 n_drop 只最高分股票。

    直接继承 qlib 的 TopkDropoutStrategy，100% 兼容回测框架，
    同时作为 orange_quant 用户开发自定义策略的参考模板。

    使用方式：

        # 方式一：直接使用
        strategy = MomentumTopKStrategy(
            topk=50,
            n_drop=5,
            signal="<PRED>",   # 使用模型预测分数
        )

        # 方式二：从 YAML 配置驱动
        strategy = init_instance_by_config({
            "class": "MomentumTopKStrategy",
            "module_path": "orange_quant.strategies.mom_topk",
            "kwargs": {"signal": "<PRED>", "topk": 50, "n_drop": 5},
        })
    """

    def __init__(
        self,
        topk: int = 50,
        n_drop: int = 5,
        hold_thresh: int = 1,
        signal: str = "<PRED>",
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
        signal : str or pd.DataFrame
            预测信号。"<PRED>" 表示使用模型预测分数，也可传入 DataFrame。
        risk_degree : float
            仓位比例，1.0 表示满仓，默认 0.95。
        """
        super().__init__(
            signal=signal,
            topk=topk,
            n_drop=n_drop,
            hold_thresh=hold_thresh,
            risk_degree=risk_degree,
            **kwargs,
        )
