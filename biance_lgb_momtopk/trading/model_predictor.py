"""
LightGBM model predictor

Loads a trained qlib LGBModel, uses the qlib Alpha158 feature engine,
and generates predicted rankings from live Binance OHLCV data.

Usage:
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
    LightGBM model predictor.

    Uses qlib's Alpha158 feature engine to compute 158 factors from OHLCV,
    loads the trained LGBModel, and predicts via the qlib DatasetH interface.
    """

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self._load_model()

    def _load_model(self):
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        print(f"[predictor] ✅ Model loaded: {self.model_path.name}")

    def predict(
        self,
        broker: BinanceBroker,
        coins: List[str],
        lookback_days: int = 160,
    ) -> pd.DataFrame:
        """
        Predict using the model, returning a coin ranking.

        Parameters
        ----------
        broker : BinanceBroker
        coins : list[str]
            Coin list (without USDT).
        lookback_days : int
            Lookback window in days (Alpha158 needs at least 90 days, 160+ recommended).

        Returns
        -------
        pd.DataFrame
            columns: coin, score, rank
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # 1. Fetch OHLCV from Binance
        print(f"[predictor] Fetching {lookback_days} days of data for {len(coins)} coins...")
        records = []
        latest_prices = {}  # coin -> latest close price
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
            print("[predictor] ⚠ Not enough valid data")
            return pd.DataFrame()

        raw_df = pd.concat(records, ignore_index=True)
        raw_df = raw_df.rename(columns={"datetime": "date"})

        # 2. Build a qlib DatasetH (Alpha158 features + data interface)
        print(f"[predictor] Computing Alpha158 + model predictions...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset, latest_date = self._create_dataset(raw_df, coins)

        # 3. Run inference through the standard qlib LGBModel.predict() interface
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = self.model.predict(dataset, segment="pred")

        # scores' index is a MultiIndex (datetime, instrument); extract instrument
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
        Build a DatasetH using the qlib Alpha158 handler.

        Returns
        -------
        dataset : DatasetH
            qlib dataset, the "pred" segment corresponds to the latest day.
        latest_date : str
            Latest date string.
        """
        import qlib
        from qlib.data.dataset import DatasetH
        from qlib.contrib.data.handler import Alpha158

        start = str(raw_df["date"].min().strftime("%Y-%m-%d"))
        end = str(raw_df["date"].max().strftime("%Y-%m-%d"))

        # Initialize qlib, pointing at the local binance data directory
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

        # Fetch all features, find the latest date
        features = handler.fetch(col_set="feature")
        latest_date = str(features.index.get_level_values("datetime").max().strftime("%Y-%m-%d"))

        # Build a DatasetH with the handler; the "pred" segment maps exactly to the latest date
        dataset = DatasetH(
            handler=handler,
            segments={"pred": (latest_date, latest_date)},
        )

        return dataset, latest_date
