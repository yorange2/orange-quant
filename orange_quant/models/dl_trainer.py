"""
深度学习模型训练封装

封装 qlib 的 PyTorch 模型（LSTM, GRU, Transformer 等），
提供统一的训练 / 预测接口。

支持所有 qlib 的 PyTorch 模型：
    LSTM, GRU, Transformer, ALSTM, TRA, Localformer,
    SFM, TCN, KRNN, GATs, HIST, IGMTF, TCTS 等
"""

import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
from qlib.model.base import Model
from qlib.data.dataset import TSDatasetH


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


class DLTrainer:
    """
    深度学习模型训练器。

    封装 qlib 任意 PyTorch 模型的训练和预测，
    自动处理 TSDatasetH（时序数据集）。

    使用方式：

        trainer = DLTrainer(
            model_name="LSTM",
            dataset=ts_dataset,
            d_feat=20, hidden_size=64, num_layers=2,
            n_epochs=100, lr=1e-3, GPU=0,
        )
        trainer.fit()
        predictions = trainer.predict()
    """

    def __init__(
        self,
        model_name: str,
        dataset: TSDatasetH,
        module_path: Optional[str] = None,
        GPU: int = 0,
        **model_kwargs,
    ):
        """
        Parameters
        ----------
        model_name : str
            模型名称，如 "LSTM", "GRU", "Transformer"。
            会从 KNOWN_DL_MODULES 自动查找 module_path。
        dataset : TSDatasetH
            qlib TSDatasetH 时序数据集，需含 "train", "valid", "test" 分段。
        module_path : str or None
            模型 module 的完整路径。为 None 时从 model_name 自动查找。
        GPU : int
            GPU 设备号，-1 表示使用 CPU。
        **model_kwargs : dict
            传给模型的参数，因模型而异：
            - 通用: d_feat, hidden_size, num_layers, dropout, n_epochs, lr,
                    early_stop, batch_size, loss, metric, n_jobs
            - Transformer 额外: n_heads, attn_dropout 等
        """
        self.model_name = model_name
        self.dataset = dataset
        self.model_kwargs = {"GPU": GPU, **model_kwargs}

        # 自动查找 module_path
        if module_path is None:
            module_path = KNOWN_DL_MODULES.get(model_name)
            if module_path is None:
                raise ValueError(
                    f"未知模型 '{model_name}'。已知模型: {list(KNOWN_DL_MODULES.keys())}\n"
                    f"如需自定义模型，请指定 module_path 参数。"
                )
        self.module_path = module_path

        self.model: Optional[Model] = None

    def fit(self) -> "DLTrainer":
        """训练模型。"""
        # 动态导入模型类
        import importlib
        module = importlib.import_module(self.module_path)
        model_cls = getattr(module, self.model_name)

        self.model = model_cls(**self.model_kwargs)
        print(f"[orange_quant] 开始训练 {self.model_name} 模型...")
        print(f"[orange_quant] module: {self.module_path}")
        print(f"[orange_quant] 参数: {self.model_kwargs}")
        self.model.fit(self.dataset)
        print(f"[orange_quant] {self.model_name} 训练完成！")
        return self

    def predict(self, segment: str = "test") -> pd.Series:
        """在指定分段上预测。"""
        if self.model is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()。")
        print(f"[orange_quant] {self.model_name} 在 {segment} 分段上预测...")
        predictions = self.model.predict(self.dataset, segment=segment)
        print(f"[orange_quant] 预测完成，共 {len(predictions)} 条记录。")
        return predictions

    def save(self, path: str) -> None:
        """保存模型。"""
        if self.model is None:
            raise RuntimeError("没有可保存的模型。")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"[orange_quant] 模型已保存到: {path}")

    @classmethod
    def load(cls, path: str, model_name: str = "LSTM") -> "DLTrainer":
        """从文件加载模型。"""
        trainer = cls.__new__(cls)
        trainer.model_name = model_name
        trainer.dataset = None
        trainer.model_kwargs = {}
        trainer.module_path = KNOWN_DL_MODULES.get(model_name, "")
        with open(path, "rb") as f:
            trainer.model = pickle.load(f)
        print(f"[orange_quant] {model_name} 模型已从 {path} 加载。")
        return trainer
