"""
完整实验流程

编排 qlib 的完整量化实验：
  数据加载 → 模型训练 → 信号生成 → 信号分析（IC）→ 回测 → 绩效分析

使用 qlib 的 QlibRecorder (R) 管理实验记录。
"""

import sys
from pathlib import Path
from typing import Optional

import yaml
import pandas as pd

import qlib
from qlib import init
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config

from ..data.handler import Alpha158Custom
from ..models.trainer import LightGBMTrainer
from ..strategies.mom_topk import MomentumTopKStrategy
from ..backtest.runner import BacktestRunner


class QuantExperiment:
    """
    量化实验管理器。

    一键运行完整实验流程，自动记录模型参数、预测信号、IC 分析、回测结果。

    使用方式：

        # 方式一：从 YAML 配置加载
        experiment = QuantExperiment.from_yaml("config/workflow_config.yaml")
        experiment.run()

        # 方式二：编程式构建
        experiment = QuantExperiment(
            provider_uri="~/.qlib/qlib_data/cn_data",
            instruments="csi300",
            train_start="2010-01-01",
            train_end="2014-12-31",
            valid_start="2015-01-01",
            valid_end="2016-12-31",
            test_start="2017-01-01",
            test_end="2020-08-01",
        )
        experiment.run()
    """

    def __init__(
        self,
        provider_uri: str = "~/.qlib/qlib_data/cn_data",
        region: str = "cn",
        instruments: str = "csi300",
        train_start: str = "2010-01-01",
        train_end: str = "2014-12-31",
        valid_start: str = "2015-01-01",
        valid_end: str = "2016-12-31",
        test_start: str = "2017-01-01",
        test_end: str = "2020-08-01",
        model_params: Optional[dict] = None,
        strategy_params: Optional[dict] = None,
        backtest_params: Optional[dict] = None,
    ):
        """
        Parameters
        ----------
        provider_uri : str
            qlib 数据路径。
        region : str
            市场区域。
        instruments : str
            股票池，如 "csi300"、"csi500"、"all"。
        train_start / train_end : str
            训练集时间范围。
        valid_start / valid_end : str
            验证集时间范围。
        test_start / test_end : str
            测试集（回测）时间范围。
        model_params : dict
            LightGBM 超参数，覆盖默认值。
        strategy_params : dict
            策略参数，覆盖默认值。
        backtest_params : dict
            回测参数，覆盖默认值。
        """
        self.provider_uri = str(Path(provider_uri).expanduser())
        self.region = region
        self.instruments = instruments
        self.train_start = train_start
        self.train_end = train_end
        self.valid_start = valid_start
        self.valid_end = valid_end
        self.test_start = test_start
        self.test_end = test_end

        self.model_params = model_params or {}
        self.strategy_params = strategy_params or {}
        self.backtest_params = backtest_params or {}

    @classmethod
    def from_yaml(cls, config_path: str) -> "QuantExperiment":
        """
        从 YAML 配置文件创建实验。

        Parameters
        ----------
        config_path : str
            YAML 配置文件路径。
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        qlib_config = config.get("qlib_init", {})
        data_config = config.get("data", {})
        train_config = config.get("train", {})
        valid_config = config.get("valid", {})
        test_config = config.get("test", {})
        model_cfg = config.get("model", {})
        strategy_cfg = config.get("strategy", {})
        backtest_cfg = config.get("backtest", {})

        return cls(
            provider_uri=qlib_config.get("provider_uri", "~/.qlib/qlib_data/cn_data"),
            region=qlib_config.get("region", "cn"),
            instruments=data_config.get("instruments", "csi300"),
            train_start=train_config.get("start", "2010-01-01"),
            train_end=train_config.get("end", "2014-12-31"),
            valid_start=valid_config.get("start", "2015-01-01"),
            valid_end=valid_config.get("end", "2016-12-31"),
            test_start=test_config.get("start", "2017-01-01"),
            test_end=test_config.get("end", "2020-08-01"),
            model_params=model_cfg.get("kwargs", {}),
            strategy_params=strategy_cfg.get("kwargs", {}),
            backtest_params=backtest_cfg,
        )

    def run(self) -> dict:
        """
        执行完整实验流程。

        Returns
        -------
        dict
            包含 model, predictions, ic_analysis, backtest_results 的字典。
        """
        print("\n" + "=" * 60)
        print("🚀 Orange Quant 实验开始")
        print("=" * 60 + "\n")

        # ── Step 1: 初始化 qlib ──
        print(f"[orange_quant] 初始化 qlib, 数据路径: {self.provider_uri}")
        qlib.init(provider_uri=self.provider_uri, region=self.region)

        # ── Step 2: 构建数据集 ──
        print(f"[orange_quant] 加载数据: {self.instruments}")
        handler = Alpha158Custom(
            instruments=self.instruments,
            start_time=self.train_start,
            end_time=self.test_end,
            fit_start_time=self.train_start,
            fit_end_time=self.train_end,
        )

        dataset = DatasetH(
            handler=handler,
            segments={
                "train": (self.train_start, self.train_end),
                "valid": (self.valid_start, self.valid_end),
                "test": (self.test_start, self.test_end),
            },
        )
        print(f"[orange_quant] 数据集构建完成: train={self.train_start}~{self.train_end}, "
              f"valid={self.valid_start}~{self.valid_end}, test={self.test_start}~{self.test_end}")

        # ── Step 3: 训练模型 ──
        trainer = LightGBMTrainer(dataset=dataset, **self.model_params)
        trainer.fit()
        predictions = trainer.predict(segment="test")

        # ── Step 4: 记录实验 ──
        with R.start(experiment_name="orange_quant_exp"):
            R.log_params(
                instruments=self.instruments,
                train_period=f"{self.train_start}_{self.train_end}",
                test_period=f"{self.test_start}_{self.test_end}",
                **self.model_params,
            )
            R.save_objects(**{"lgb_model.pkl": trainer.model})

            # 信号记录
            sr = SignalRecord(trainer.model, dataset, R.get_recorder())
            sr.generate()

            # 信号分析（IC、Rank IC、Long-Short 收益）
            sar = SigAnaRecord(R.get_recorder())
            sar.generate()

            # 回测
            strategy = MomentumTopKStrategy(
                signal="<PRED>",
                topk=50,
                n_drop=5,
                **self.strategy_params,
            )

            port_analysis_config = {
                "executor": {
                    "class": "SimulatorExecutor",
                    "module_path": "qlib.backtest.executor",
                    "kwargs": {
                        "time_per_step": "day",
                        "generate_portfolio_metrics": True,
                    },
                },
                "backtest": {
                    "start_time": self.test_start,
                    "end_time": self.test_end,
                    "account": 100000000,  # 初始资金 1亿
                    "benchmark": self.backtest_params.get("benchmark", "SH000300"),
                    "exchange_kwargs": {
                        "freq": "day",
                        "limit_threshold": 0.095,
                        "deal_price": "close",
                        "open_cost": 0.0005,
                        "close_cost": 0.0015,
                        "min_cost": 5,
                    },
                },
                "strategy": {
                    "class": "TopkDropoutStrategy",
                    "module_path": "qlib.contrib.strategy.signal_strategy",
                    "kwargs": {
                        "signal": "<PRED>",
                        "topk": 50,
                        "n_drop": 5,
                        "risk_degree": 0.95,
                        **self.strategy_params,
                    },
                },
            }

            par = PortAnaRecord(
                R.get_recorder(),
                port_analysis_config,
                "day",
            )
            par.generate()

        print("\n" + "=" * 60)
        print("✅ 实验完成！使用 `R.get_recorder()` 查看结果。")
        print("=" * 60 + "\n")

        return {
            "trainer": trainer,
            "predictions": predictions,
            "recorder": R.get_recorder(),
        }


def run_from_yaml(config_path: str = "config/workflow_config.yaml") -> dict:
    """
    从 YAML 配置运行实验的便捷函数。

    可直接在 notebook 或脚本中调用：
        from orange_quant.workflow.experiment import run_from_yaml
        results = run_from_yaml("config/workflow_config.yaml")
    """
    experiment = QuantExperiment.from_yaml(config_path)
    return experiment.run()
