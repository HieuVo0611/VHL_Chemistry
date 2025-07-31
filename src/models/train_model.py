"""
src/models/train_model.py

Train a multi-output regression pipeline with log-target transformation.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.compose import TransformedTargetRegressor

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

from joblib import dump

# Configure module‐level logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "RandomForest",
    params: Optional[Dict[str, Any]] = None,
    model_save_path: Optional[Path] = None
) -> TransformedTargetRegressor:
    """
    Train a multi-output regression pipeline (feature scaling + regressor)
    wrapped in a log1p/expm1 target transformer.

    Args:
        X: array of shape (n_samples, n_features)
        y: array of shape (n_samples, n_targets)
        model_type: "RandomForest", "ElasticNet", or "XGBoost"
        params: hyperparameters for the chosen base regressor
        model_save_path: if provided, path to save the fitted pipeline (joblib .pkl)

    Returns:
        A fitted TransformedTargetRegressor that applies log1p to targets before fitting
        and expm1 to predictions.
    """
    params = params or {}
    mt = model_type.lower()
    logger.info("Initializing model type: %s", mt)

    # Select base regressor (multi-output if needed)
    if mt == "randomforest":
        base = RandomForestRegressor(**params)
    elif mt == "elasticnet":
        lin = ElasticNet(**params)
        base = MultiOutputRegressor(lin)
    elif mt in ("xgboost", "xgb", "xgbregressor"):
        if XGBRegressor is None:
            raise ImportError("XGBoost is not installed; install xgboost to use this model type.")
        base = MultiOutputRegressor(XGBRegressor(**params))
    else:
        raise ValueError(f"Unsupported model_type '{model_type}'. "
                         "Choose from RandomForest, ElasticNet, XGBoost.")

    # Build pipeline: scale features then regress
    feat_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", base)
    ])

    # Wrap with log1p/expm1 on targets
    model = TransformedTargetRegressor(
        regressor=feat_pipe,
        func=np.log1p,
        inverse_func=np.expm1
    )

    # Train
    logger.info("Training pipeline on %d samples and %d features", X.shape[0], X.shape[1])
    model.fit(X, y)
    logger.info("Training complete")

    # Save if requested
    if model_save_path:
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        dump(model, model_save_path)
        logger.info("Saved trained pipeline to %s", model_save_path)

    return model
