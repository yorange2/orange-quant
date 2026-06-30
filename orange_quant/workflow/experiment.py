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
import mlflow

import qlib
from qlib import init
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config

from qlib.contrib.data.handler import Alpha158
from qlib.contrib.model.gbdt import LGBModel
# 已知的 qlib PyTorch 模型 module 路径
KNOWN_DL_MODULES = {
    "LSTM": "qlib.contrib.model.pytorch_lstm_ts",
    "GRU": "qlib.contrib.model.pytorch_gru_ts",
    "Transformer": "qlib.contrib.model.pytorch_transformer_ts",
    "ALSTM": "qlib.contrib.model.pytorch_alstm_ts",
    "GATs": "qlib.contrib.model.pytorch_gats_ts",
    "TCN": "qlib.contrib.model.pytorch_tcn_ts",
    "TRA": "qlib.contrib.model.pytorch_tra",
    "Localformer": "qlib.contrib.model.pytorch_localformer_ts",
    "SFM": "qlib.contrib.model.pytorch_sfm",
    "KRNN": "qlib.contrib.model.pytorch_krnn",
    "HIST": "qlib.contrib.model.pytorch_hist",
    "IGMTF": "qlib.contrib.model.pytorch_igmtf",
    "TCTS": "qlib.contrib.model.pytorch_tcts",
    "ADARNN": "qlib.contrib.model.pytorch_adarnn",
    "ADD": "qlib.contrib.model.pytorch_add",
    "Sandwich": "qlib.contrib.model.pytorch_sandwich",
}
from ..strategies.mom_topk import MomentumTopKStrategy


class QuantExperiment:
    """
    量化实验管理器。

    一键运行完整实验流程，自动记录模型参数、预测信号、IC 分析、回测结果。
    模型和策略均使用 orange_quant 自己的类（通过 YAML 配置驱动）。

    使用方式：

        # 方式一：从 YAML 配置加载
        experiment = QuantExperiment.from_yaml("config/csi300-lgb-momtopk.yaml")
        experiment.run()

        # 方式二：编程式构建
        experiment = QuantExperiment(
            provider_uri="data/qlib_data/cn_data",
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
        provider_uri: str = "data/qlib_data/cn_data",
        region: str = "cn",
        instruments: str = "csi300",
        train_start: str = "2010-01-01",
        train_end: str = "2014-12-31",
        valid_start: str = "2015-01-01",
        valid_end: str = "2016-12-31",
        test_start: str = "2017-01-01",
        test_end: str = "2020-08-01",
        model_params: Optional[dict] = None,
        strategy_config: Optional[dict] = None,
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
        strategy_config : dict
            策略完整配置（含 class, module_path, kwargs），用于 PortAnaRecord。
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
        self.strategy_config = strategy_config or {}
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
            provider_uri=qlib_config.get("provider_uri", "data/qlib_data/cn_data"),
            region=qlib_config.get("region", "cn"),
            instruments=data_config.get("instruments", "csi300"),
            train_start=train_config.get("start", "2010-01-01"),
            train_end=train_config.get("end", "2014-12-31"),
            valid_start=valid_config.get("start", "2015-01-01"),
            valid_end=valid_config.get("end", "2016-12-31"),
            test_start=test_config.get("start", "2017-01-01"),
            test_end=test_config.get("end", "2020-08-01"),
            model_params=model_cfg.get("kwargs", {}),
            strategy_config=strategy_cfg,  # 完整的策略配置（含 class, module_path, kwargs）
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
        print(f"[orange_quant] 数据集构建完成: train={self.train_start}~{self.train_end}, "
              f"valid={self.valid_start}~{self.valid_end}, test={self.test_start}~{self.test_end}")

        # ── Step 3: 训练模型 ──
        model = LGBModel(**self.model_params)
        model.fit(dataset)
        predictions = model.predict(dataset, segment="test")

        # ── Step 4: 记录实验 ──
        # 结束数据加载阶段可能启动的 mlflow run，避免嵌套冲突
        if mlflow.active_run():
            mlflow.end_run()

        with R.start(experiment_name="orange_quant_exp"):
            recorder = R.get_recorder()  # 立即保存 recorder 引用
            R.log_params(
                instruments=self.instruments,
                train_period=f"{self.train_start}_{self.train_end}",
                test_period=f"{self.test_start}_{self.test_end}",
                **self.model_params,
            )

            # 信号记录
            sr = SignalRecord(model, dataset, recorder)
            sr.generate()

            # 信号分析（IC、Rank IC、Long-Short 收益）
            sar = SigAnaRecord(recorder)
            sar.generate()

            # 回测 — 使用 orange_quant 策略配置
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
                    "exchange_kwargs": self.backtest_params.get("exchange_kwargs", {
                        "freq": "day",
                        "limit_threshold": 0.095,
                        "deal_price": "close",
                        "open_cost": 0.0005,
                        "close_cost": 0.0015,
                        "min_cost": 5,
                    }),
                },
                "strategy": self.strategy_config,
            }

            par = PortAnaRecord(
                recorder,
                port_analysis_config,
                "day",
            )
            par.generate()

        print("\n" + "=" * 60)
        print("✅ 实验完成！使用 `R.get_recorder()` 查看结果。")
        print("=" * 60 + "\n")

        return {
            "model": model,
            "predictions": predictions,
            "recorder": recorder,
        }


def run_from_yaml(config_path: str = "config/csi300-lgb-momtopk.yaml") -> dict:
    """
    从 YAML 配置运行实验的便捷函数。

    可直接在 notebook 或脚本中调用：
        from orange_quant.workflow.experiment import run_from_yaml
        results = run_from_yaml("config/csi300-lgb-momtopk.yaml")

    训练完成后自动将模型导出到 models/{config_name}.pkl。
    """
    import pickle

    experiment = QuantExperiment.from_yaml(config_path)
    results = experiment.run()

    # 自动导出模型到 models/
    model_path = Path("models")
    model_path.mkdir(parents=True, exist_ok=True)
    config_name = Path(config_path).stem  # e.g. "csi300-lgb-momtopk"
    output_path = model_path / f"{config_name}.pkl"
    pickle.dump(results["model"], open(output_path, "wb"))
    print(f"💾 模型已导出至 {output_path}")

    return results


def run_dl_from_yaml(config_path: str = "config/csi300-lstm-momtopk.yaml") -> dict:
    """
    从 YAML 配置运行深度学习实验的便捷函数。

    支持所有 qlib PyTorch 模型：
        LSTM, GRU, Transformer, ALSTM, TRA, Localformer,
        SFM, TCN, KRNN, GATs, HIST, IGMTF, TCTS 等

    使用方式：
        from orange_quant.workflow.experiment import run_dl_from_yaml
        results = run_dl_from_yaml("config/csi300-lstm-momtopk.yaml")

    Parameters
    ----------
    config_path : str
        深度学习实验 YAML 配置文件路径。
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    qlib_cfg = config.get("qlib_init", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    valid_cfg = config.get("valid", {})
    test_cfg = config.get("test", {})
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})
    strategy_cfg = config.get("strategy", {})
    backtest_cfg = config.get("backtest", {})

    provider_uri = str(Path(qlib_cfg.get("provider_uri", "data/qlib_data/cn_data")).expanduser())
    region = qlib_cfg.get("region", "cn")
    instruments = data_cfg.get("instruments", "csi300")
    train_start = train_cfg.get("start", "2008-01-01")
    train_end = train_cfg.get("end", "2014-12-31")
    valid_start = valid_cfg.get("start", "2015-01-01")
    valid_end = valid_cfg.get("end", "2016-12-31")
    test_start = test_cfg.get("start", "2017-01-01")
    test_end = test_cfg.get("end", "2020-08-01")

    model_name = model_cfg.get("name", "LSTM")
    model_kwargs = model_cfg.get("kwargs", {})
    step_len = dataset_cfg.get("step_len", 20)
    benchmark = backtest_cfg.get("benchmark", "SH000300")

    print("\n" + "=" * 60)
    print(f"🚀 Orange Quant 深度学习实验 — {model_name}")
    print("=" * 60 + "\n")

    # ── Step 1: 初始化 qlib ──
    print(f"[orange_quant] 初始化 qlib, 数据路径: {provider_uri}")
    qlib.init(provider_uri=provider_uri, region=region)

    # ── Step 2: 构建时序数据集 (TSDatasetH) ──
    print(f"[orange_quant] 加载数据: {instruments}, step_len={step_len}")

    # DL 模型使用 TSDatasetH + 特殊预处理
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset import TSDatasetH

    handler = Alpha158(
        instruments=instruments,
        start_time=train_start,
        end_time=test_end,
        fit_start_time=train_start,
        fit_end_time=train_end,
        infer_processors=[
            {
                "class": "FilterCol",
                "kwargs": {
                    "fields_group": "feature",
                    "col_list": [
                        "RESI5", "WVMA5", "RSQR5", "KLEN", "RSQR10", "CORR5",
                        "CORD5", "CORR10", "ROC60", "RESI10", "VSTD5", "RSQR60",
                        "CORR60", "WVMA60", "STD5", "RSQR20", "CORD60", "CORD10",
                        "CORR20", "KLOW",
                    ],
                },
            },
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ],
        learn_processors=[
            {"class": "DropnaLabel"},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
        ],
        label=["Ref($close, -2) / Ref($close, -1) - 1"],
    )

    dataset = TSDatasetH(
        handler=handler,
        segments={
            "train": (train_start, train_end),
            "valid": (valid_start, valid_end),
            "test": (test_start, test_end),
        },
        step_len=step_len,
    )
    print(f"[orange_quant] TSDatasetH 构建完成, step_len={step_len}")

    # ── Step 3: 训练模型 ──
    module_path = KNOWN_DL_MODULES.get(model_name)
    if module_path is None:
        raise ValueError(
            f"未知模型 '{model_name}'。已知模型: {list(KNOWN_DL_MODULES.keys())}"
        )
    import importlib
    module = importlib.import_module(module_path)
    model_cls = getattr(module, model_name)
    model = model_cls(**model_kwargs)

    print(f"[orange_quant] 开始训练 {model_name} 模型...")
    model.fit(dataset)
    print(f"[orange_quant] {model_name} 训练完成！")

    predictions = model.predict(dataset, segment="test")

    # ── Step 4: 记录实验 ──
    if mlflow.active_run():
        mlflow.end_run()

    with R.start(experiment_name=f"orange_quant_dl_{model_name.lower()}"):
        recorder = R.get_recorder()
        R.log_params(
            model=model_name,
            instruments=instruments,
            step_len=step_len,
            train_period=f"{train_start}_{train_end}",
            test_period=f"{test_start}_{test_end}",
            **model_kwargs,
        )
        # 信号记录
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # 信号分析
        sar = SigAnaRecord(recorder)
        sar.generate()

        # 回测
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
                "start_time": test_start,
                "end_time": test_end,
                "account": 100000000,
                "benchmark": benchmark,
                "exchange_kwargs": backtest_cfg.get("exchange_kwargs", {
                    "freq": "day",
                    "limit_threshold": 0.095,
                    "deal_price": "close",
                    "open_cost": 0.0005,
                    "close_cost": 0.0015,
                    "min_cost": 5,
                }),
            },
            "strategy": strategy_cfg,
        }

        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()

    print("\n" + "=" * 60)
    print(f"✅ {model_name} 实验完成！")
    print("=" * 60 + "\n")

    results = {
        "model": model,
        "predictions": predictions,
        "recorder": recorder,
    }

    # 自动导出模型到 models/
    import pickle
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    config_name = Path(config_path).stem  # e.g. "csi300-lstm-momtopk"
    output_path = model_dir / f"{config_name}.pkl"
    pickle.dump(model, open(output_path, "wb"))
    print(f"💾 模型已导出至 {output_path}")

    return results
