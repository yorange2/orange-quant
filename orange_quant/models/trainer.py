"""
LightGBM 模型训练封装

提供统一的训练 / 预测 / 保存 / 加载接口。
"""

import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
from qlib.model.base import Model
from qlib.contrib.model.gbdt import LGBModel
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


class LightGBMTrainer:
    """
    LightGBM 模型训练器。

    使用方式：

        trainer = LightGBMTrainer(
            dataset=dataset,
            loss="mse",
            num_boost_round=500,
            early_stopping_rounds=50,
        )
        trainer.fit()
        predictions = trainer.predict()
        trainer.save("models/lgb_model.pkl")
    """

    def __init__(
        self,
        dataset: DatasetH,
        loss: str = "mse",
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50,
        learning_rate: float = 0.05,
        num_leaves: int = 64,
        feature_fraction: float = 0.8,
        bagging_fraction: float = 0.8,
        bagging_freq: int = 5,
        verbose_eval: int = 50,
        **kwargs,
    ):
        """
        Parameters
        ----------
        dataset : DatasetH
            qlib DatasetH 实例，需包含 "train", "valid", "test" 分段。
        loss : str
            损失函数，回归任务默认 "mse"。
        num_boost_round : int
            迭代轮数。
        early_stopping_rounds : int
            早停轮数。
        learning_rate : float
            学习率。
        num_leaves : int
            叶子节点数。
        """
        self.dataset = dataset
        self.model_params = {
            "loss": loss,
            "num_boost_round": num_boost_round,
            "early_stopping_rounds": early_stopping_rounds,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "feature_fraction": feature_fraction,
            "bagging_fraction": bagging_fraction,
            "bagging_freq": bagging_freq,
            "verbose_eval": verbose_eval,
            **kwargs,
        }
        self.model: Optional[LGBModel] = None

    def fit(self) -> "LightGBMTrainer":
        """训练模型。使用 dataset 的 'train' 分段训练，'valid' 分段早停。"""
        self.model = LGBModel(**self.model_params)
        print("[orange_quant] 开始训练 LightGBM 模型...")
        print(f"[orange_quant] 参数: {self.model_params}")
        self.model.fit(self.dataset)
        print("[orange_quant] 训练完成！")
        return self

    def predict(self, segment: str = "test") -> pd.Series:
        """在指定分段上预测，返回预测分数 Series。"""
        if self.model is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()。")
        print(f"[orange_quant] 在 {segment} 分段上预测...")
        predictions = self.model.predict(self.dataset, segment=segment)
        print(f"[orange_quant] 预测完成，共 {len(predictions)} 条记录。")
        return predictions

    def save(self, path: str) -> None:
        """保存模型到文件。"""
        if self.model is None:
            raise RuntimeError("没有可保存的模型。")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"[orange_quant] 模型已保存到: {path}")

    @classmethod
    def load(cls, path: str) -> "LightGBMTrainer":
        """从文件加载模型。"""
        trainer = cls.__new__(cls)
        trainer.dataset = None
        trainer.model_params = {}
        with open(path, "rb") as f:
            trainer.model = pickle.load(f)
        print(f"[orange_quant] 模型已从 {path} 加载。")
        return trainer
