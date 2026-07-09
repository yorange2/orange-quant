"""
完整实验流程

编排 qlib 的完整量化实验：
  数据加载 → 模型训练 → 信号生成 → 信号分析（IC）→ 回测 → 绩效分析
"""

from pathlib import Path
from typing import Optional

import yaml
import mlflow

import qlib
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.data.dataset import DatasetH
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.model.gbdt import LGBModel


class QuantExperiment:
    """
    量化实验管理器 — Hyperliquid 永续合约。
    """

    def __init__(
        self,
        provider_uri: str = "data/qlib_data/hyperliquid",
        region: str = "cn",
        instruments: str = "all",
        train_start: str = "2022-01-01",
        train_end: str = "2025-06-30",
        valid_start: str = "2025-07-01",
        valid_end: str = "2026-01-31",
        test_start: str = "2026-02-01",
        test_end: str = "2026-06-27",
        model_params: Optional[dict] = None,
        strategy_config: Optional[dict] = None,
        backtest_params: Optional[dict] = None,
    ):
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
        self.strategy_config = strategy_config or {}
        self.backtest_params = backtest_params or {}

    @classmethod
    def from_yaml(cls, config_path: str) -> "QuantExperiment":
        """从 YAML 配置文件创建实验"""
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
            provider_uri=qlib_config.get("provider_uri", "data/qlib_data/hyperliquid"),
            region=qlib_config.get("region", "cn"),
            instruments=data_config.get("instruments", "all"),
            train_start=train_config.get("start", "2022-01-01"),
            train_end=train_config.get("end", "2025-06-30"),
            valid_start=valid_config.get("start", "2025-07-01"),
            valid_end=valid_config.get("end", "2026-01-31"),
            test_start=test_config.get("start", "2026-02-01"),
            test_end=test_config.get("end", "2026-06-27"),
            model_params=model_cfg.get("kwargs", {}),
            strategy_config=strategy_cfg,
            backtest_params=backtest_cfg,
        )

    def run(self) -> dict:
        """执行完整实验流程"""
        print("\n" + "=" * 60)
        print("🚀 Orange Quant Hyperliquid 实验开始")
        print("=" * 60 + "\n")

        # Step 1: 初始化 qlib
        print(f"[hyperliquid_lgb_momtopk] 初始化 qlib, 数据路径: {self.provider_uri}")
        qlib.init(provider_uri=self.provider_uri, region=self.region)

        # Step 2: 构建数据集
        print(f"[hyperliquid_lgb_momtopk] 加载数据: {self.instruments}")
        handler = Alpha158(
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
        print(f"[hyperliquid_lgb_momtopk] 数据集构建完成: "
              f"train={self.train_start}~{self.train_end}, "
              f"valid={self.valid_start}~{self.valid_end}, "
              f"test={self.test_start}~{self.test_end}")

        # Step 3: 训练模型
        model = LGBModel(**self.model_params)
        model.fit(dataset)
        predictions = model.predict(dataset, segment="test")

        # Step 4: 记录实验
        if mlflow.active_run():
            mlflow.end_run()

        with R.start(experiment_name="hyperliquid_lgb_momtopk_exp"):
            recorder = R.get_recorder()
            R.log_params(
                instruments=self.instruments,
                train_period=f"{self.train_start}_{self.train_end}",
                test_period=f"{self.test_start}_{self.test_end}",
                **self.model_params,
            )

            sr = SignalRecord(model, dataset, recorder)
            sr.generate()

            sar = SigAnaRecord(recorder)
            sar.generate()

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
                    "account": 1000000,
                    "benchmark": self.backtest_params.get("benchmark", "BTC"),
                    "exchange_kwargs": self.backtest_params.get("exchange_kwargs", {
                        "freq": "day",
                        "limit_threshold": 1.0,
                        "deal_price": "close",
                        "open_cost": 0.0005,
                        "close_cost": 0.0005,
                        "min_cost": 0,
                    }),
                },
                "strategy": self.strategy_config,
            }

            par = PortAnaRecord(recorder, port_analysis_config, "day")
            par.generate()

        print("\n" + "=" * 60)
        print("✅ 实验完成！使用 `R.get_recorder()` 查看结果。")
        print("=" * 60 + "\n")

        return {
            "model": model,
            "predictions": predictions,
            "recorder": recorder,
        }


def run_from_yaml(config_path: str = "config/hyperliquid-lgb-momtopk.yaml") -> dict:
    """从 YAML 配置运行实验并导出模型"""
    import pickle

    experiment = QuantExperiment.from_yaml(config_path)
    results = experiment.run()

    model_path = Path("models")
    model_path.mkdir(parents=True, exist_ok=True)
    config_name = Path(config_path).stem
    output_path = model_path / f"{config_name}.pkl"
    pickle.dump(results["model"], open(output_path, "wb"))
    print(f"💾 模型已导出至 {output_path}")

    return results
