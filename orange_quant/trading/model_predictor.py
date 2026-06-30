"""
LightGBM 模型预测器

加载训练好的 qlib LGBModel，使用 qlib Alpha158 特征引擎，
从 Binance 实时 OHLCV 数据生成预测排名。

使用方式：
    predictor = ModelPredictor(model_path="models/binance20_lgb.pkl")
    scores = predictor.predict(broker, coins=["BTC", "ETH", ...])
"""

import pickle
import json
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
    加载训练好的 LGBModel 进行预测。
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
        lookback_days: int = 90,
    ) -> pd.DataFrame:
        """
        使用模型预测，返回币种排名。

        Parameters
        ----------
        broker : BinanceBroker
        coins : list[str]
            币种列表（不含 USDT）。
        lookback_days : int
            回看天数（Alpha158 至少需要 60 天）。

        Returns
        -------
        pd.DataFrame
            columns: coin, score, rank
        """
        if self.model is None:
            raise RuntimeError("模型未加载")

        # 1. 获取 OHLCV
        print(f"[predictor] 获取 {len(coins)} 个币种 {lookback_days} 天数据...")
        records = []
        for coin in coins:
            sym = f"{coin}/USDT"
            try:
                df = broker.fetch_ohlcv(sym, "1d", limit=lookback_days)
                if len(df) < 60:
                    continue
                df["instrument"] = coin
                df = df.reset_index()
                records.append(df)
            except Exception as e:
                print(f"  {coin}: {e}")

        if len(records) < 3:
            print("[predictor] ⚠ 有效数据不足")
            return pd.DataFrame()

        # 2. 构建 qlib 特征
        print(f"[predictor] 计算 Alpha158 + 模型预测...")
        raw_df = pd.concat(records, ignore_index=True)
        raw_df = raw_df.rename(columns={"datetime": "date"})

        # 用 qlib 的 D.features 计算特征
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features_df = self._compute_alpha158(raw_df, coins)

        # 3. 取最新一天的预测
        latest = features_df.index.get_level_values("datetime").max()
        X = features_df.xs(latest, level="datetime")

        # 4. 预测
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = self.model.predict(X)

        result = pd.DataFrame({
            "coin": X.index,
            "score": scores.values.flatten() if hasattr(scores, "values") else scores,
        })
        result["rank"] = result["score"].rank(ascending=False)
        result = result.sort_values("score", ascending=False)

        print(f"[predictor] ✅ Top 5: {result.head(5)['coin'].tolist()}")
        return result

    def _compute_alpha158(self, raw_df, coins):
        """用 qlib 的 Alpha158 处理器计算特征"""
        import qlib
        from qlib.data import D
        from qlib.contrib.data.handler import Alpha158

        # 计算日期范围
        start = str(raw_df["date"].min().strftime("%Y-%m-%d"))
        end = str(raw_df["date"].max().strftime("%Y-%m-%d"))

        # 用 qlib 初始化已有数据目录（模型训练时的 provider），
        # 这样 Alpha158 的 fit 参数（标准化用的均值和标准差）能复用
        try:
            qlib.init(provider_uri="data/qlib_data/binance", region="cn", auto_mount=False)
        except Exception:
            # 如果本地没有 qlib 数据，用空配置
            pass

        # 创建 handler 计算特征
        handler = Alpha158(
            instruments=list(coins),
            start_time=start,
            end_time=end,
            fit_start_time=start,
            fit_end_time=end,
        )

        # 获取特征 DataFrame
        # handler 内部会用 D.features() 计算
        features = handler.fetch(col_set="feature")
        return features
