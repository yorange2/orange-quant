"""
LightGBM 模型预测器

加载训练好的 qlib LGBModel，使用 qlib Alpha158 特征引擎，
从 Binance 实时 OHLCV 数据生成预测排名。

使用方式：
    predictor = ModelPredictor(model_path="models/binance-lgb-momtopk.pkl")
    scores = predictor.predict(broker, coins=["BTC", "ETH", ...])
"""

import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from .broker import BinanceBroker


class ModelPredictor:
    """
    LightGBM 模型预测器。

    用 qlib 的 Alpha158 特征引擎从 OHLCV 计算 158 个因子，
    加载训练好的 LGBModel，通过 qlib DatasetH 接口进行预测。
    """

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self._load_model()

    def _load_model(self):
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        print(f"[predictor] ✅ 模型已加载: {self.model_path.name}")

    def predict(
        self,
        broker: BinanceBroker,
        coins: List[str],
        lookback_days: int = 160,
    ) -> pd.DataFrame:
        """
        使用模型预测，返回币种排名。

        Parameters
        ----------
        broker : BinanceBroker
        coins : list[str]
            币种列表（不含 USDT）。
        lookback_days : int
            回看天数（Alpha158 至少需要 90 天，推荐 ≥ 160）。

        Returns
        -------
        pd.DataFrame
            columns: coin, score, rank
        """
        if self.model is None:
            raise RuntimeError("模型未加载")

        # 1. 从 Binance 获取 OHLCV
        print(f"[predictor] 获取 {len(coins)} 个币种 {lookback_days} 天数据...")
        records = []
        latest_prices = {}  # coin → 最新收盘价
        for coin in coins:
            sym = f"{coin}/USDT"
            try:
                df = broker.fetch_ohlcv(sym, "1d", limit=lookback_days)
                if len(df) < 60:
                    continue
                latest_prices[coin] = float(df["close"].iloc[-1])
                df["instrument"] = coin
                df = df.reset_index()
                records.append(df)
            except Exception as e:
                print(f"  {coin}: {e}")

        if len(records) < 3:
            print("[predictor] ⚠ 有效数据不足")
            return pd.DataFrame()

        raw_df = pd.concat(records, ignore_index=True)
        raw_df = raw_df.rename(columns={"datetime": "date"})

        # 2. 构建 qlib DatasetH（Alpha158 特征 + 数据接口）
        print(f"[predictor] 计算 Alpha158 + 模型预测...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset, latest_date = self._create_dataset(raw_df, coins)

        # 3. 用 qlib LGBModel.predict() 标准接口推理
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = self.model.predict(dataset, segment="pred")

        # scores 的 index 是 MultiIndex (datetime, instrument)，提取 instrument
        coins_pred = [idx[1] for idx in scores.index]
        result = pd.DataFrame({
            "coin": coins_pred,
            "score": scores.values,
            "price": [latest_prices.get(c, 0) for c in coins_pred],
        })
        result["rank"] = result["score"].rank(ascending=False)
        result = result.sort_values("score", ascending=False)

        print(f"[predictor] ✅ Top 5: {result.head(5)['coin'].tolist()}")
        return result

    def _create_dataset(self, raw_df, coins):
        """
        用 qlib Alpha158 handler 构建 DatasetH。

        Returns
        -------
        dataset : DatasetH
            qlib 数据集，segment "pred" 对应最新一天。
        latest_date : str
            最新日期字符串。
        """
        import qlib
        from qlib.data.dataset import DatasetH
        from qlib.contrib.data.handler import Alpha158

        start = str(raw_df["date"].min().strftime("%Y-%m-%d"))
        end = str(raw_df["date"].max().strftime("%Y-%m-%d"))

        # 初始化 qlib，指向本地 binance 数据目录
        try:
            qlib.init(provider_uri="data/qlib_data/binance", region="cn", auto_mount=False)
        except Exception:
            pass

        handler = Alpha158(
            instruments=list(coins),
            start_time=start,
            end_time=end,
            fit_start_time=start,
            fit_end_time=end,
        )

        # 获取全部特征，找到最新日期
        features = handler.fetch(col_set="feature")
        latest_date = str(features.index.get_level_values("datetime").max().strftime("%Y-%m-%d"))

        # 用 handler 构建 DatasetH，segment "pred" 精确对应最新日期
        dataset = DatasetH(
            handler=handler,
            segments={"pred": (latest_date, latest_date)},
        )

        return dataset, latest_date
