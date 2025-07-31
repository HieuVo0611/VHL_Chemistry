"""
src/evaluate.py

Đánh giá mô hình trên tập kiểm thử, in MAE/MSE/R2 cho Pb và Cd.
"""

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(
    model: Any,
    X_test,  # pd.DataFrame of shape (n_samples, n_features)
    y_test   # pd.DataFrame with columns ['pb','cd'] of shape (n_samples, 2)
) -> None:
    """
    Predict on X_test và so sánh với y_test, in ra MAE/MSE/R2 cho từng chất.

    Args:
        model: một TransformedTargetRegressor hoặc Pipeline đã được fit.
        X_test: DataFrame kích thước (n_samples, n_features).
        y_test: DataFrame với hai cột 'pb','cd' kích thước (n_samples, 2).
    """
    # Chuyển về mảng numpy
    X = X_test.values
    y_true = y_test[['pb', 'cd']].values  # shape (n_samples, 2)

    # Dự đoán (đã được expm1 ngược log tự động bởi TransformedTargetRegressor)
    y_pred = model.predict(X)            # shape (n_samples, 2)

    # Metrics cho Pb (cột 0) và Cd (cột 1)
    mae_pb = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    mae_cd = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    mse_pb = mean_squared_error(y_true[:, 0], y_pred[:, 0])
    mse_cd = mean_squared_error(y_true[:, 1], y_pred[:, 1])
    r2_pb  = r2_score(y_true[:, 0], y_pred[:, 0])
    r2_cd  = r2_score(y_true[:, 1], y_pred[:, 1])

    logger.info("=== Evaluation on Test Set ===")
    logger.info("Pb:  MAE = %.4f µM,  MSE = %.4f,  R2 = %.4f", mae_pb, mse_pb, r2_pb)
    logger.info("Cd:  MAE = %.4f µM,  MSE = %.4f,  R2 = %.4f", mae_cd, mse_cd, r2_cd)
