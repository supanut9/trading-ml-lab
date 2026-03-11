import pandas as pd
import duckdb


class FeatureEngineer:
    """
    Transforms raw OHLCV data into features for ML training.
    """

    @staticmethod
    def load_data(db_path: str, symbol: str, timeframe: str) -> pd.DataFrame:
        conn = duckdb.connect(db_path, read_only=True)
        query = "SELECT * FROM ohlcv WHERE symbol = ? AND timeframe = ? ORDER BY timestamp ASC"
        df = conn.execute(query, [symbol, timeframe]).df()
        conn.close()
        return df

    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Returns
        df["returns"] = df["close"].pct_change()

        # Moving Averages
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()

        # Volatility
        df["volatility_20"] = df["returns"].rolling(window=20).std()

        # RSI (Vectorized version)
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # Price relative to SMA
        df["close_to_sma_20"] = df["close"] / df["sma_20"] - 1

        return df

    @staticmethod
    def add_targets(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
        """
        Add the target variable we want to predict.
        Example: Price change in next 'horizon' candles.
        """
        df = df.copy()
        # Predicting direction: 1 if price goes up, 0 if down
        df["target"] = (df["close"].shift(-horizon) > df["close"]).astype(int)
        return df

    @classmethod
    def prepare_dataset(cls, db_path: str, symbol: str, timeframe: str):
        df = cls.load_data(db_path, symbol, timeframe)
        df = cls.add_technical_indicators(df)
        df = cls.add_targets(df)

        # Drop NaNs created by rolling windows
        df.dropna(inplace=True)

        return df
