# trading-ml-lab

ML experimentation for trading — feature engineering, model training, walk-forward evaluation.

## Tech Stack
- **Python 3.12+** with `PyTorch`, `XGBoost`, `MLflow`, `Polars`
- Uses `uv` for package management

## Workflow
```bash
# Train a model
uv run mllab train --model xgboost --data ../trading-data-pipeline/data/btc_1h.parquet

# Evaluate with walk-forward validation
uv run mllab evaluate --model models/xgb_latest.pkl

# Export model as a strategy
uv run mllab export --model models/xgb_latest.pkl --to ../trading-strategies/
```
