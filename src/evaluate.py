"""
Evaluation metrics shared between training, the notebook, and the
on-demand prediction sanity checks.

Multiple metrics are reported because each tells a different story:
  - MAE   : average absolute dollar error, easy to explain to stakeholders.
  - RMSE  : penalises large misses harder than MAE (relevant for a sales
            forecast feeding budget decisions, where a single huge miss
            is worse than several small ones).
  - MAPE  : a scale-free percentage error, useful for comparing accuracy
            across products with very different price points.
  - R2    : how much of the variance in order value the model explains.
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mean_absolute_percentage_error(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }
