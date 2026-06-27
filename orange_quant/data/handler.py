"""
自定义 Alpha 因子处理器

在 qlib Alpha158 基础上扩展自定义因子。
参考：qlib/qlib/contrib/data/handler.py
"""

from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset.handler import DataHandlerLP


class Alpha158Custom(Alpha158):
    """
    扩展的 Alpha158 处理器，支持添加自定义因子。

    使用方式：

        handler = Alpha158Custom(
            instruments="csi300",
            start_time="2010-01-01",
            end_time="2020-08-01",
            fit_start_time="2010-01-01",
            fit_end_time="2014-12-31",
            extra_fields=["$pe", "$pb", "$market_cap"],  # 自定义额外字段
        )

        dataset = DatasetH(handler=handler, segments={...})
    """

    def __init__(
        self,
        instruments: str = "csi300",
        start_time: str = "2010-01-01",
        end_time: str = "2020-08-01",
        fit_start_time: str = None,
        fit_end_time: str = None,
        infer_processors: list = None,
        learn_processors: list = None,
        extra_fields: list = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        instruments : str
            股票池，如 "csi300", "csi500", "all"。
        start_time / end_time : str
            数据时间范围。
        fit_start_time / fit_end_time : str
            用于 fit（如标准化）的时间范围，默认等于 start/end。
        extra_fields : list
            额外的基本面/行情字段，如 ["$pe", "$pb", "$market_cap"]。
        """
        self._extra_fields = extra_fields or []

        # 构建传给父类的参数，避免传入 None 覆盖父类默认值
        parent_kwargs = {
            "instruments": instruments,
            "start_time": start_time,
            "end_time": end_time,
            "fit_start_time": fit_start_time,
            "fit_end_time": fit_end_time,
        }
        if infer_processors is not None:
            parent_kwargs["infer_processors"] = infer_processors
        if learn_processors is not None:
            parent_kwargs["learn_processors"] = learn_processors
        parent_kwargs.update(kwargs)

        super().__init__(**parent_kwargs)

    def get_feature_config(self):
        """可重写此方法添加自定义因子表达式。"""
        # 获取 Alpha158 默认的 158 个因子
        config = super().get_feature_config()

        # 示例：添加自定义因子
        # custom_factors = [
        #     # 5日均量比
        #     ("Volume_Ratio_5", "Ref($volume, 0) / Mean($volume, 5)"),
        #     # 10日振幅
        #     ("Amplitude_10", "Max($high, 10) / Min($low, 10) - 1"),
        # ]
        # config.extend(custom_factors)

        return config
