"""
Central configuration: paths and constants shared across the project.
Keeping these in one place means main.py, the notebook, and the CI/CD
scripts all agree on where things live.
"""
from pathlib import Path

# ---- Paths -----------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT_DIR / "datasets"
RAW_DATA_PATH = DATASETS_DIR / "Chocolate_Sales.csv"

BATCH_DIR = DATASETS_DIR / "batch_prediction_dataset"
ON_DEMAND_INPUT_PATH = BATCH_DIR / "on_demand_dataset.csv"
ON_DEMAND_OUTPUT_PATH = BATCH_DIR / "on_demand_predictions.csv"

MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "sales_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

MLRUNS_DIR = ROOT_DIR / "mlruns"

# ---- Columns ----------------------------------------------------------------
TARGET_COL = "Amount"

CATEGORICAL_FEATURES = ["Product", "Country", "Channel"]
NUMERIC_FEATURES = [
    "Discount_Pct",
    "Price_per_Box",
    "Marketing_Spend",
    "Boxes_Shipped",
    "Year",
    "Month",
    "Quarter",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# ---- Modelling ---------------------------------------------------------------
RANDOM_STATE = 42
MLFLOW_EXPERIMENT_NAME = "chocolate-sales-forecasting"

# Time-based split: hold out the last 2 calendar months of real data
# as a stand-in "future" test set, because the business question is a
# forecasting question (predict 2024), not an interpolation question.
HOLDOUT_MONTHS = 2

# 2024 simulation: how we extrapolate from 2023 behaviour when no
# real 2024 orders exist yet. See src/forecast.py.
FORECAST_YEAR = 2024
