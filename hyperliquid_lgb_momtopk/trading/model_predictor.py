"""
LightGBM 模型预测器

加载训练好的 qlib LGBModel，使用 qlib Alpha158 特征引擎，
从 Hyperliquid OHLCV 数据生成预测排名。
"""

import pickle
import warnings
from pathlib import Path
from typing import List, Union

import pandas as pd
import numpy as np

from .broker import HyperliquidBroker, PaperBroker


class ModelPredictor:
    """
    LightGBM 模型预测器。
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
        broker: Union[HyperliquidBroker, PaperBroker],
        coins: List[str],
        lookback_days: int = 160,
    ) -> pd.DataFrame:
        """
        使用模型预测，返回币种排名。

        Parameters
        ----------
        broker : HyperliquidBroker or PaperBroker
        coins : list[str]
            币种列表（如 ["BTC", "ETH"]）。
        lookback_days : int
            回看天数。

        Returns
        -------
        pd.DataFrame
            columns: coin, score, rank
        """
        if self.model is None:
            raise RuntimeError("模型未加载")

        print(f"[predictor] 获取 {len(coins)} 个币种 {lookback_days} 天数据...")
        records = []
        latest_prices = {}
        for coin in coins:
            try:
                df = broker.fetch_ohlcv(coin, "1d", limit=lookback_days)
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

        print(f"[predictor] 计算 Alpha158 + 模型预测...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset, latest_date = self._create_dataset(raw_df, coins)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = self.model.predict(dataset, segment="pred")

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
        """用 qlib Alpha158 handler 构建 DatasetH"""
        import qlib
        from qlib.data.dataset import DatasetH
        from qlib.contrib.data.handler import Alpha158

        start = str(raw_df["date"].min().strftime("%Y-%m-%d"))
        end = str(raw_df["date"].max().strftime("%Y-%m-%d"))

        try:
            qlib.init(provider_uri="data/qlib_data/hyperliquid", region="cn", auto_mount=False)
        except Exception:
            pass

        handler = Alpha158(
            instruments=list(coins),
            start_time=start,
            end_time=end,
            fit_start_time=start,
            fit_end_time=end,
        )

        features = handler.fetch(col_set="feature")
        latest_date = str(features.index.get_level_values("datetime").max().strftime("%Y-%m-%d"))

        dataset = DatasetH(
            handler=handler,
            segments={"pred": (latest_date, latest_date)},
        )

        return dataset, latest_date
