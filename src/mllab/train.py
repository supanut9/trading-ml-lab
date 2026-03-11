import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from .features import FeatureEngineer


class Trainer:
    """
    Trains ML models for trading.
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def train_rf(self, df: pd.DataFrame, target_col: str = "target"):
        features = ["returns", "sma_20", "sma_50", "volatility_20", "rsi_14", "close_to_sma_20"]
        
        X = df[features]
        y = df[target_col]
        
        # Chronological split for time-series data
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluation
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"RandomForest Training Complete.")
        print(f"Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        model_path = os.path.join(self.model_dir, "rf_latest.pkl")
        joblib.dump(model, model_path)
        print(f"Model saved to: {model_path}")
        
        return model, model_path


if __name__ == "__main__":
    # Example usage
    fe = FeatureEngineer()
    dataset = fe.prepare_dataset(
        db_path="../trading-data-pipeline/data/trading.db",
        symbol="BTC/USDT",
        timeframe="1h"
    )
    
    trainer = Trainer()
    trainer.train_rf(dataset)
