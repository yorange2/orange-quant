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

# qlib's "Alpha158 for NN" 20-column FilterCol recipe (the well-behaved subset).
_NN20_COLS = [
    "RESI5", "WVMA5", "RSQR5", "KLEN", "RSQR10", "CORR5", "CORD5", "CORR10",
    "ROC60", "RESI10", "VSTD5", "RSQR60", "CORR60", "WVMA60", "STD5", "RSQR20",
    "CORD60", "CORD10", "CORR20", "KLOW",
]

_DEFAULT_ACCOUNT = 1000000
_DEFAULT_EXCHANGE_KWARGS = {
    "freq": "day",
    "limit_threshold": 0.095,
    "deal_price": "close",
    "open_cost": 0.0005,
    "close_cost": 0.0015,
    "min_cost": 5,
}


def _prefer_mps(model):
    """Move a qlib PyTorch model to the Apple MPS (Metal) GPU when available.

    qlib's PyTorch models only auto-select CUDA-or-CPU (their ``GPU`` kwarg is
    CUDA-only), so on Apple Silicon they train on CPU. Overriding ``model.device``
    and moving the underlying ``nn.Module`` to ``mps`` gives a large speedup. Any
    registered buffers (e.g. a transformer's positional encoding) move with ``.to``.

    The inner module attribute is named differently across qlib models
    (``model`` for TransformerModel, ``ALSTM_model``/``GRU_model``/``LSTM_model``
    for the recurrent ones), so we can't hard-code ``model.model``: we scan the
    model's attributes and move every ``nn.Module`` we find. Missing that is not
    cosmetic — qlib moves the *input* batch to ``self.device`` during fit/predict,
    so a device set to ``mps`` with weights left on CPU raises a hard
    "weight is on cpu but expected on mps" error.

    Safe no-op when MPS is unavailable or the model has no ``nn.Module``/``device``.
    Set PYTORCH_ENABLE_MPS_FALLBACK=1 so any op MPS lacks falls back to CPU.
    """
    import os
    if os.environ.get("ORANGE_DISABLE_MPS"):
        # Escape hatch: MPS' CPU-fallback for ops it lacks (e.g. some LSTM/
        # attention kernels) can thrash device transfers and end up slower than
        # native CPU for these small recurrent models. Set ORANGE_DISABLE_MPS=1
        # to keep training on CPU.
        print("[experiment] ORANGE_DISABLE_MPS set; training on CPU")
        return model
    try:
        import torch
    except Exception:
        return model
    mps = getattr(torch.backends, "mps", None)
    if not (mps and mps.is_available()):
        return model
    try:
        dev = torch.device("mps")
        moved = []
        for attr in vars(model):
            val = getattr(model, attr)
            if isinstance(val, torch.nn.Module):
                val.to(dev)
                moved.append(attr)
        if hasattr(model, "device"):
            model.device = dev
        if moved:
            print(f"[experiment] Using Apple MPS (Metal GPU) for "
                  f"{type(model).__name__} (moved: {', '.join(moved)})")
        else:
            print(f"[experiment] No nn.Module found on {type(model).__name__}; "
                  f"set device=mps only")
    except Exception as e:
        print(f"[experiment] MPS enable failed ({e}); staying on the default device")
    return model


def _cast_handler_float32(handler) -> None:
    """Downcast a qlib handler's prepared frames to float32 (for MPS).

    qlib feeds the transformer the raw batch dtype (float64) and MPS rejects
    float64, so cast the processed learn/infer/raw frames the samplers read from.
    """
    import numpy as np
    for attr in ("_learn", "_infer", "_data"):
        df = getattr(handler, attr, None)
        if df is not None and hasattr(df, "astype"):
            try:
                setattr(handler, attr, df.astype(np.float32))
            except Exception as e:
                print(f"[experiment] float32 cast of handler.{attr} failed: {e}")


def _float32_reweighter():
    """A Reweighter that yields float32 sample weights (for MPS).

    qlib's recurrent models (ALSTM/GRU/LSTM) otherwise build the training weight
    as ``np.ones(len(dl))`` — float64 — and do ``weight.to(self.device)`` before
    the loss. MPS has no float64 dtype at all, so that move raises outright. Their
    ``fit`` accepts a ``reweighter`` whose weights are used verbatim, so returning
    float32 ones keeps the loss identical (all-ones) while staying MPS-safe.
    Returns ``None`` if qlib's Reweighter base can't be imported.
    """
    try:
        from qlib.data.dataset.weight import Reweighter
    except Exception:
        return None
    import numpy as np

    class _Float32Ones(Reweighter):
        def __init__(self):  # base __init__ intentionally raises; override it
            pass

        def reweight(self, data):
            return np.ones(len(data), dtype=np.float32)

    return _Float32Ones()


def _end_active_experiment() -> None:
    """Close any active mlflow run / qlib experiment so qlib can be re-initialized.

    qlib's RecorderWrapper raises RecorderInitializationError when ``qlib.init``
    is called while ``exp_manager.active_experiment`` is set. What sets it is
    training: LGBModel.fit logs its eval metrics through ``R.log_metrics``, which
    implicitly starts an experiment. (Building a handler does not — a fit is
    required to reproduce the failure.) Both mlflow and qlib state are cleared
    here; each step is best-effort because neither may exist on a first call.
    """
    try:
        if mlflow.active_run():
            mlflow.end_run()
    except Exception:  # noqa: BLE001 - cleanup must never mask the real work
        pass
    try:
        R.end_exp()
    except Exception:  # noqa: BLE001 - no active experiment is the normal case
        pass


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
        ensemble_config: Optional[dict] = None,
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
        self.ensemble_config = ensemble_config or {}

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
            ensemble_config=model_cfg.get("ensemble", {}),
        )

    def fit_predict(self) -> dict:
        """Init qlib, build the dataset, train, and predict the test segment.

        Split out of ``run`` so callers that only need a trained model and its
        test-segment signal — e.g. a strategy sweep, where topk/risk_degree
        affect only the backtest and not the model — can reuse one training run
        across many backtests instead of retraining per parameter set.

        Returns {"model", "dataset", "predictions"}; no experiment recording.
        """
        # qlib refuses to re-init while an experiment is active, and training
        # leaves one active. A caller fitting several datasets in one process
        # (the strategy sweep) would hit RecorderInitializationError on the
        # call after its first fit, so clear any leftover experiment first.
        _end_active_experiment()

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

        n_seeds = int(self.ensemble_config.get("n_seeds", 1) or 1)
        if n_seeds > 1:
            # Seed-bagged ensemble: train n_seeds LGBModels with different seeds
            # and average their signals. Reduces cross-sectional signal variance,
            # which robustly lifts Rank IC out-of-sample (see orange_quant.ensemble).
            from orange_quant.ensemble import EnsembleLGB
            base_seed = int(self.model_params.get("seed", 0))
            submodels = []
            for i in range(n_seeds):
                params = dict(self.model_params)
                params["seed"] = base_seed + i
                print(f"[experiment] Training ensemble member {i + 1}/{n_seeds} "
                      f"(seed={params['seed']})")
                m = LGBModel(**params)
                m.fit(dataset)
                submodels.append(m)
            model = EnsembleLGB(submodels)
        else:
            model = LGBModel(**self.model_params)
            model.fit(dataset)

        return {"model": model, "dataset": dataset,
                "predictions": model.predict(dataset, segment="test")}

    def run(self) -> dict:
        """Execute the full experiment pipeline."""
        print("\n" + "=" * 60)
        print("🚀 Orange Quant experiment starting")
        print("=" * 60 + "\n")

        fitted = self.fit_predict()
        model, dataset = fitted["model"], fitted["dataset"]
        predictions = fitted["predictions"]

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

    # Feature selection for the NN. The qlib "Alpha158 for NN" recipe filters the
    # 158 raw features down to 20 well-behaved ones (default). That handicaps the
    # DL model relative to LGB, which trains on all 158, so make it configurable:
    #   dataset.feature_cols: "nn20" (default) -> the 20-col FilterCol below
    #                         "all"            -> no FilterCol, all 158 features
    #                         [list of names]  -> use exactly those columns
    # Whatever is chosen must match model.kwargs.d_feat.
    feature_cols = dataset_cfg.get("feature_cols", "nn20")
    infer_processors = []
    if feature_cols not in ("all", None):
        col_list = _NN20_COLS if feature_cols == "nn20" else list(feature_cols)
        infer_processors.append(
            {"class": "FilterCol",
             "kwargs": {"fields_group": "feature", "col_list": col_list}}
        )
        n_feat = len(col_list)
    else:
        n_feat = 158  # full Alpha158 feature set
    infer_processors += [
        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
    ]
    d_feat = model_kwargs.get("d_feat")
    if d_feat is not None and d_feat != n_feat:
        print(f"[experiment] WARNING: model d_feat={d_feat} != selected "
              f"feature count {n_feat}; overriding d_feat={n_feat}")
    model_kwargs["d_feat"] = n_feat

    print(f"[experiment] Loading data: {instruments}, step_len={step_len}, "
          f"features={feature_cols} (d_feat={n_feat})")

    from qlib.data.dataset import TSDatasetH

    handler = Alpha158(
        instruments=instruments,
        start_time=train_start,
        end_time=test_end,
        fit_start_time=train_start,
        fit_end_time=train_end,
        infer_processors=infer_processors,
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
    _prefer_mps(model)
    fit_kwargs = {}
    if str(getattr(model, "device", "")) == "mps":
        # MPS is float32-only, but qlib moves the raw float64 batch to the device
        # before casting inside forward(), so downcast the prepared data first.
        _cast_handler_float32(handler)
        # ...and, for models whose fit takes a reweighter (ALSTM/GRU/LSTM), swap
        # the default float64 all-ones weight for a float32 one (MPS has no
        # float64, so their `weight.to(device)` would otherwise raise).
        import inspect
        if "reweighter" in inspect.signature(model.fit).parameters:
            rw = _float32_reweighter()
            if rw is not None:
                fit_kwargs["reweighter"] = rw

    print(f"[experiment] Starting training of {model_name} model...")
    model.fit(dataset, **fit_kwargs)
    print(f"[experiment] {model_name} training complete!")

    # qlib PyTorch models' predict() takes no `segment` (it uses the test segment
    # internally) — unlike LGBModel.predict(dataset, segment="test").
    predictions = model.predict(dataset)

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
