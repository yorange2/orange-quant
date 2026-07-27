"""
Full experiment pipeline

Orchestrates the complete qlib quantitative experiment:
  data loading -> model training -> signal generation -> signal analysis (IC) -> backtest -> performance analysis

Uses qlib's QlibRecorder (R) to manage experiment records.
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
from qlib.utils.paral import AsyncCaller


def _flush_async_metrics(recorder) -> None:
    """Flush qlib's background metric logger before metrics are read back.

    SigAnaRecord logs IC/Rank IC on a background thread (qlib wraps log_metrics
    with AsyncCaller). PortAnaRecord.check() then reads every metric file back
    synchronously, so without a flush it can hit a half-written 'Rank IC' file
    and raise "Metric 'Rank IC' is malformed. No data found." We flush the queue,
    then reopen a fresh caller so later metric logging (and end_run) still works
    — this mirrors qlib's own start_run pattern.
    """
    if getattr(recorder, "async_log", None) is not None:
        recorder.async_log.wait()
        recorder.async_log = AsyncCaller()

from qlib.contrib.data.handler import Alpha158
from qlib.contrib.model.gbdt import LGBModel
# Known qlib PyTorch model module paths
KNOWN_DL_MODULES = {
    "LSTM": "qlib.contrib.model.pytorch_lstm_ts",
    "GRU": "qlib.contrib.model.pytorch_gru_ts",
    "TransformerModel": "qlib.contrib.model.pytorch_transformer_ts",
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


class QuantExperiment:
    """
    Quantitative experiment manager.

    Runs the full experiment pipeline in one call, automatically recording model
    parameters, predicted signals, IC analysis, and backtest results.
    Both the model and strategy use biance_lgb_momtopk's own classes (driven by YAML config).

    Usage:

        # Option 1: load from a YAML config
        experiment = QuantExperiment.from_yaml("config/csi300-lgb-momtopk.yaml")
        experiment.run()

        # Option 2: build programmatically
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
            qlib data path.
        region : str
            Market region.
        instruments : str
            Stock universe, e.g. "csi300", "csi500", "all".
        train_start / train_end : str
            Training set time range.
        valid_start / valid_end : str
            Validation set time range.
        test_start / test_end : str
            Test (backtest) set time range.
        model_params : dict
            LightGBM hyperparameters, overriding the defaults.
        strategy_config : dict
            Full strategy config (including class, module_path, kwargs), used for PortAnaRecord.
        backtest_params : dict
            Backtest parameters, overriding the defaults.
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
        Create an experiment from a YAML config file.

        Parameters
        ----------
        config_path : str
            Path to the YAML config file.
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
            strategy_config=strategy_cfg,  # Full strategy config (including class, module_path, kwargs)
            backtest_params=backtest_cfg,
        )

    def run(self) -> dict:
        """
        Execute the full experiment pipeline.

        Returns
        -------
        dict
            Dictionary containing model, predictions, ic_analysis, backtest_results.
        """
        print("\n" + "=" * 60)
        print("🚀 Orange Quant experiment starting")
        print("=" * 60 + "\n")

        # -- Step 1: initialize qlib --
        print(f"[biance_lgb_momtopk] Initializing qlib, data path: {self.provider_uri}")
        qlib.init(provider_uri=self.provider_uri, region=self.region)

        # -- Step 2: build dataset --
        print(f"[biance_lgb_momtopk] Loading data: {self.instruments}")
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
        print(f"[biance_lgb_momtopk] Dataset built: train={self.train_start}~{self.train_end}, "
              f"valid={self.valid_start}~{self.valid_end}, test={self.test_start}~{self.test_end}")

        # -- Step 3: train model --
        model = LGBModel(**self.model_params)
        model.fit(dataset)
        predictions = model.predict(dataset, segment="test")

        # -- Step 4: record experiment --
        # End any mlflow run started during the data loading stage to avoid nested-run conflicts
        if mlflow.active_run():
            mlflow.end_run()

        with R.start(experiment_name="biance_lgb_momtopk_exp"):
            recorder = R.get_recorder()  # Save the recorder reference immediately
            # The signal-analysis and backtest records below are experiment
            # tracking only — callers just need the trained model. A transient
            # mlflow hiccup here must never take down the pipeline, so keep them
            # non-fatal and let run() still return the model.
            try:
                R.log_params(
                    instruments=self.instruments,
                    train_period=f"{self.train_start}_{self.train_end}",
                    test_period=f"{self.test_start}_{self.test_end}",
                    **self.model_params,
                )

                # Signal record
                sr = SignalRecord(model, dataset, recorder)
                sr.generate()

                # Signal analysis (IC, Rank IC, Long-Short returns)
                sar = SigAnaRecord(recorder)
                sar.generate()

                # Flush async metric writes before PortAnaRecord reads them back
                _flush_async_metrics(recorder)

                # Backtest -- uses biance_lgb_momtopk strategy config
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
                        "account": 100000000,  # Initial capital: 100 million
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
            except Exception as e:  # noqa: BLE001 - analysis is best-effort
                print(f"⚠️  Analysis/backtest recording failed "
                      f"(model is still valid and will be exported): {e}")

        print("\n" + "=" * 60)
        print("✅ Experiment complete! Use `R.get_recorder()` to view results.")
        print("=" * 60 + "\n")

        return {
            "model": model,
            "predictions": predictions,
            "recorder": recorder,
        }


def run_from_yaml(config_path: str = "config/csi300-lgb-momtopk.yaml") -> dict:
    """
    Convenience function to run an experiment from a YAML config.

    Can be called directly from a notebook or script:
        from biance_lgb_momtopk.workflow.experiment import run_from_yaml
        results = run_from_yaml("config/csi300-lgb-momtopk.yaml")

    After training completes, the model is automatically exported to models/{config_name}.pkl.
    """
    import pickle

    experiment = QuantExperiment.from_yaml(config_path)
    results = experiment.run()

    # Automatically export the model to models/
    model_path = Path("models")
    model_path.mkdir(parents=True, exist_ok=True)
    config_name = Path(config_path).stem  # e.g. "csi300-lgb-momtopk"
    output_path = model_path / f"{config_name}.pkl"
    pickle.dump(results["model"], open(output_path, "wb"))
    print(f"💾 Model exported to {output_path}")

    return results


def run_dl_from_yaml(config_path: str = "config/csi300-lstm-momtopk.yaml") -> dict:
    """
    Convenience function to run a deep learning experiment from a YAML config.

    Supports all qlib PyTorch models:
        LSTM, GRU, Transformer, ALSTM, TRA, Localformer,
        SFM, TCN, KRNN, GATs, HIST, IGMTF, TCTS, etc.

    Usage:
        from biance_lgb_momtopk.workflow.experiment import run_dl_from_yaml
        results = run_dl_from_yaml("config/csi300-lstm-momtopk.yaml")

    Parameters
    ----------
    config_path : str
        Path to the deep learning experiment YAML config file.
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
    print(f"🚀 Orange Quant deep learning experiment — {model_name}")
    print("=" * 60 + "\n")

    # -- Step 1: initialize qlib --
    print(f"[biance_lgb_momtopk] Initializing qlib, data path: {provider_uri}")
    qlib.init(provider_uri=provider_uri, region=region)

    # -- Step 2: build time-series dataset (TSDatasetH) --
    print(f"[biance_lgb_momtopk] Loading data: {instruments}, step_len={step_len}")

    # DL models use TSDatasetH + special preprocessing
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
    print(f"[biance_lgb_momtopk] TSDatasetH built, step_len={step_len}")

    # -- Step 3: train model --
    module_path = KNOWN_DL_MODULES.get(model_name)
    if module_path is None:
        raise ValueError(
            f"Unknown model '{model_name}'. Known models: {list(KNOWN_DL_MODULES.keys())}"
        )
    import importlib
    module = importlib.import_module(module_path)
    model_cls = getattr(module, model_name)
    model = model_cls(**model_kwargs)

    print(f"[biance_lgb_momtopk] Starting training of {model_name} model...")
    model.fit(dataset)
    print(f"[biance_lgb_momtopk] {model_name} training complete!")

    predictions = model.predict(dataset, segment="test")

    # -- Step 4: record experiment --
    if mlflow.active_run():
        mlflow.end_run()

    with R.start(experiment_name=f"biance_lgb_momtopk_dl_{model_name.lower()}"):
        recorder = R.get_recorder()
        R.log_params(
            model=model_name,
            instruments=instruments,
            step_len=step_len,
            train_period=f"{train_start}_{train_end}",
            test_period=f"{test_start}_{test_end}",
            **model_kwargs,
        )
        # Signal record
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # Signal analysis
        sar = SigAnaRecord(recorder)
        sar.generate()

        # Flush async metric writes before PortAnaRecord reads them back
        _flush_async_metrics(recorder)

        # Backtest
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
    print(f"✅ {model_name} experiment complete!")
    print("=" * 60 + "\n")

    results = {
        "model": model,
        "predictions": predictions,
        "recorder": recorder,
    }

    # Automatically export the model to models/
    import pickle
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    config_name = Path(config_path).stem  # e.g. "csi300-lstm-momtopk"
    output_path = model_dir / f"{config_name}.pkl"
    pickle.dump(model, open(output_path, "wb"))
    print(f"💾 Model exported to {output_path}")

    return results
