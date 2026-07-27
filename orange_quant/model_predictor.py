"""
LightGBM model predictor (exchange-agnostic).

Loads a trained qlib LGBModel, uses the qlib Alpha158 feature engine, and
generates predicted rankings. Market data comes from the coin-based broker
(``broker.fetch_ohlcv(coin)``); Alpha158 features are read from the qlib binary
store at ``provider_uri`` (rebuilt daily before retrain), so both the broker and
the data directory are supplied by the caller rather than hardcoded per exchange.
"""

import pickle
import warnings
from pathlib import Path
from typing import List

import pandas as pd


class ModelPredictor:
    def __init__(self, model_path: str, provider_uri: str):
        self.model_path = Path(model_path)
        self.provider_uri = provider_uri
        self.model = None
        self._load_model()

    def _load_model(self):
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        print(f"[predictor] ✅ Model loaded: {self.model_path.name}")

    def predict(self, broker, coins: List[str], lookback_days: int = 160) -> pd.DataFrame:
        """
        Predict using the model, returning a coin ranking.

        Parameters
        ----------
        broker : coin-based broker (live or paper)
        coins : list[str]
            Coin list (e.g. ["BTC", "ETH"]).
        lookback_days : int
            Lookback window in days.

        Returns
        -------
        pd.DataFrame with columns: coin, score, price, rank
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        print(f"[predictor] Fetching {lookback_days} days of data for {len(coins)} coins...")
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
            print("[predictor] ⚠ Not enough valid data")
            return pd.DataFrame()

        raw_df = pd.concat(records, ignore_index=True)
        raw_df = raw_df.rename(columns={"datetime": "date"})

        print(f"[predictor] Computing Alpha158 + model predictions...")
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
        """Build a DatasetH using the qlib Alpha158 handler."""
        import qlib
        from qlib.data.dataset import DatasetH
        from qlib.contrib.data.handler import Alpha158

        start = str(raw_df["date"].min().strftime("%Y-%m-%d"))
        end = str(raw_df["date"].max().strftime("%Y-%m-%d"))

        try:
            qlib.init(provider_uri=self.provider_uri, region="cn", auto_mount=False)
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
