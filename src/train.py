import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src import config
from src.evaluate import compute_metrics

try:
    import mlflow
    import mlflow.sklearn

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                config.CATEGORICAL_FEATURES,
            ),
        ],
        remainder="passthrough",
    )


def time_based_split(df: pd.DataFrame):
    cutoff = df["Order_Date"].max() - pd.DateOffset(months=config.HOLDOUT_MONTHS)
    train_df = df[df["Order_Date"] <= cutoff]
    test_df = df[df["Order_Date"] > cutoff]
    return train_df, test_df


def get_candidate_models() -> dict:
    return {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=config.RANDOM_STATE),
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=200, max_depth=8, random_state=config.RANDOM_STATE
        ),
    }


def train_and_select_best(df: pd.DataFrame) -> dict:
    train_df, test_df = time_based_split(df)
    X_train, y_train = train_df[config.FEATURE_COLUMNS], train_df[config.TARGET_COL]
    X_test, y_test = test_df[config.FEATURE_COLUMNS], test_df[config.TARGET_COL]

    if MLFLOW_AVAILABLE:
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    leaderboard = []
    best = {"name": None, "pipeline": None, "metrics": None, "rmse": np.inf}

    for name, estimator in get_candidate_models().items():
        pipeline = Pipeline(
            steps=[("preprocess", build_preprocessor()), ("model", estimator)]
        )

        run_ctx = (
            mlflow.start_run(run_name=name) if MLFLOW_AVAILABLE else _NullContext()
        )
        with run_ctx:
            start = time.time()
            pipeline.fit(X_train, y_train)
            train_seconds = time.time() - start

            preds = pipeline.predict(X_test)
            metrics = compute_metrics(y_test, preds)
            cv_sample = X_train.sample(
                n=min(25_000, len(X_train)), random_state=config.RANDOM_STATE
            )
            cv_target = y_train.loc[cv_sample.index]
            cv_scores = cross_val_score(
                pipeline,
                cv_sample,
                cv_target,
                cv=KFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE),
                scoring="neg_root_mean_squared_error",
                n_jobs=1,
            )
            metrics["CV_RMSE_mean"] = float(-cv_scores.mean())
            metrics["CV_RMSE_std"] = float(cv_scores.std())
            metrics["train_seconds"] = round(train_seconds, 2)

            if MLFLOW_AVAILABLE:
                mlflow.log_param("model_type", name)
                mlflow.log_params(
                    {
                        k: v
                        for k, v in estimator.get_params().items()
                        if isinstance(v, (int, float, str, bool)) or v is None
                    }
                )
                mlflow.log_metrics(
                    {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                )
                mlflow.sklearn.log_model(pipeline, artifact_path="model")

            leaderboard.append({"model": name, **metrics})

            if metrics["RMSE"] < best["rmse"]:
                best = {
                    "name": name,
                    "pipeline": pipeline,
                    "metrics": metrics,
                    "rmse": metrics["RMSE"],
                }

    return {"best": best, "leaderboard": leaderboard}


def save_model(pipeline, name: str, metrics: dict) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.MODEL_PATH)
    metadata = {
        "model_name": name,
        "metrics": metrics,
        "feature_columns": config.FEATURE_COLUMNS,
        "target_column": config.TARGET_COL,
        "trained_at": pd.Timestamp.utcnow().isoformat(),
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2))


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
