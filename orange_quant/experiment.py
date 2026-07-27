"""
Full experiment pipeline (exchange-agnostic).

Orchestrates the complete qlib quantitative experiment:
  data loading -> model training -> signal generation -> signal analysis (IC)
  -> backtest -> performance analysis

Uses qlib's QlibRecorder (R) to manage experiment records. Everything is driven
by the YAML config, so the same pipeline serves A-shares, Binance, and Hyperliquid.
"""

from pathlib import Path
from typing import Optional

import yaml
import mlflow

import qlib
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.data.dataset import DatasetH
from qlib.utils.paral import AsyncCaller

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

_DEFAULT_ACCOUNT = 1000000
_DEFAULT_EXCHANGE_KWARGS = {
    "freq": "day",
    "limit_threshold": 0.095,
    "deal_price": "close",
    "open_cost": 0.0005,
    "close_cost": 0.0015,
    "min_cost": 5,
}


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


def _port_analysis_config(strategy_config, backtest_params, test_start, test_end):
    """Build the qlib PortAnaRecord config from the YAML backtest section."""
    return {
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
            "account": backtest_params.get("account", _DEFAULT_ACCOUNT),
            "benchmark": backtest_params.get("benchmark", "SH000300"),
            "exchange_kwargs": backtest_params.get("exchange_kwargs", dict(_DEFAULT_EXCHANGE_KWARGS)),
        },
        "strategy": strategy_config,
    }


class QuantExperiment:
    """
    Quantitative experiment manager. Runs the full pipeline in one call, recording
    model parameters, predicted signals, IC analysis, and backtest results. Model
    and strategy are driven entirely by the YAML config.
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
        experiment_name: str = "orange_quant_exp",
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
        self.experiment_name = experiment_name

    @classmethod
    def from_yaml(cls, config_path: str, experiment_name: Optional[str] = None) -> "QuantExperiment":
        """Create an experiment from a YAML config file."""
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
            strategy_config=strategy_cfg,
            backtest_params=backtest_cfg,
            experiment_name=experiment_name or f"{Path(config_path).stem}_exp",
        )

    def run(self) -> dict:
        """Execute the full experiment pipeline."""
        print("\n" + "=" * 60)
        print("🚀 Orange Quant experiment starting")
        print("=" * 60 + "\n")

        print(f"[experiment] Initializing qlib, data path: {self.provider_uri}")
        qlib.init(provider_uri=self.provider_uri, region=self.region)

        print(f"[experiment] Loading data: {self.instruments}")
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
        print(f"[experiment] Dataset built: train={self.train_start}~{self.train_end}, "
              f"valid={self.valid_start}~{self.valid_end}, test={self.test_start}~{self.test_end}")

        model = LGBModel(**self.model_params)
        model.fit(dataset)
        predictions = model.predict(dataset, segment="test")

        # End any mlflow run started during data loading to avoid nested-run conflicts
        if mlflow.active_run():
            mlflow.end_run()

        with R.start(experiment_name=self.experiment_name):
            recorder = R.get_recorder()
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

                sr = SignalRecord(model, dataset, recorder)
                sr.generate()

                sar = SigAnaRecord(recorder)
                sar.generate()

                # Flush async metric writes before PortAnaRecord reads them back
                _flush_async_metrics(recorder)

                par = PortAnaRecord(
                    recorder,
                    _port_analysis_config(self.strategy_config, self.backtest_params,
                                          self.test_start, self.test_end),
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
    """Run a LightGBM experiment from a YAML config and export the model to models/."""
    import pickle

    experiment = QuantExperiment.from_yaml(config_path)
    results = experiment.run()

    model_path = Path("models")
    model_path.mkdir(parents=True, exist_ok=True)
    config_name = Path(config_path).stem
    output_path = model_path / f"{config_name}.pkl"
    pickle.dump(results["model"], open(output_path, "wb"))
    print(f"💾 Model exported to {output_path}")

    return results


def run_dl_from_yaml(config_path: str = "config/csi300-lstm-momtopk.yaml") -> dict:
    """
    Run a deep-learning experiment from a YAML config, exporting the model to models/.

    Supports all qlib PyTorch models: LSTM, GRU, Transformer, ALSTM, TRA,
    Localformer, SFM, TCN, KRNN, GATs, HIST, IGMTF, TCTS, etc.
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

    print("\n" + "=" * 60)
    print(f"🚀 Orange Quant deep learning experiment — {model_name}")
    print("=" * 60 + "\n")

    print(f"[experiment] Initializing qlib, data path: {provider_uri}")
    qlib.init(provider_uri=provider_uri, region=region)

    print(f"[experiment] Loading data: {instruments}, step_len={step_len}")

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
    print(f"[experiment] TSDatasetH built, step_len={step_len}")

    module_path = KNOWN_DL_MODULES.get(model_name)
    if module_path is None:
        raise ValueError(
            f"Unknown model '{model_name}'. Known models: {list(KNOWN_DL_MODULES.keys())}"
        )
    import importlib
    module = importlib.import_module(module_path)
    model_cls = getattr(module, model_name)
    model = model_cls(**model_kwargs)

    print(f"[experiment] Starting training of {model_name} model...")
    model.fit(dataset)
    print(f"[experiment] {model_name} training complete!")

    predictions = model.predict(dataset, segment="test")

    if mlflow.active_run():
        mlflow.end_run()

    with R.start(experiment_name=f"{Path(config_path).stem}_dl_{model_name.lower()}"):
        recorder = R.get_recorder()
        try:
            R.log_params(
                model=model_name,
                instruments=instruments,
                step_len=step_len,
                train_period=f"{train_start}_{train_end}",
                test_period=f"{test_start}_{test_end}",
                **model_kwargs,
            )
            sr = SignalRecord(model, dataset, recorder)
            sr.generate()

            sar = SigAnaRecord(recorder)
            sar.generate()

            # Flush async metric writes before PortAnaRecord reads them back
            _flush_async_metrics(recorder)

            par = PortAnaRecord(
                recorder,
                _port_analysis_config(strategy_cfg, backtest_cfg, test_start, test_end),
                "day",
            )
            par.generate()
        except Exception as e:  # noqa: BLE001 - analysis is best-effort
            print(f"⚠️  Analysis/backtest recording failed "
                  f"(model is still valid and will be exported): {e}")

    print("\n" + "=" * 60)
    print(f"✅ {model_name} experiment complete!")
    print("=" * 60 + "\n")

    results = {
        "model": model,
        "predictions": predictions,
        "recorder": recorder,
    }

    import pickle
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    config_name = Path(config_path).stem
    output_path = model_dir / f"{config_name}.pkl"
    pickle.dump(model, open(output_path, "wb"))
    print(f"💾 Model exported to {output_path}")

    return results
