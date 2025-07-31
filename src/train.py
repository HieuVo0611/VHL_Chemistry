from pathlib import Path
import logging
from typing import Tuple, Any

import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_SAVE_DIR
from src.data.load_data import load_dataset
from src.data.preprocess import pad_dataset
from src.data.feature_engineering import extract_features_dataset
from src.models.train_model import train_model

# Configure module-level logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_pipeline(
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Any, pd.DataFrame, pd.DataFrame]:
    """
    Load data, preprocess, extract features, train multi-output model pipeline,
    and save the trained pipeline to disk.

    Args:
        test_size: Proportion of the dataset to include in the test split.
        random_state: Random seed for reproducibility.

    Returns:
        pipeline: Trained sklearn Pipeline (scaler + regressor).
        X_test: DataFrame of test features.
        y_test: DataFrame of test labels (pb & cd).
    """
    # 1. Load raw data
    logger.info("Loading raw dataset from %s", RAW_DATA_DIR)
    df_raw = load_dataset(RAW_DATA_DIR)
    logger.info("Loaded raw data: %d rows", len(df_raw))

    # 2. Preprocess (padding/cropping)
    logger.info("Padding all samples to uniform length")
    df_padded = pad_dataset(df_raw)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = PROCESSED_DATA_DIR / "padded_data.pkl"
    df_padded.to_pickle(processed_path)
    logger.info("Saved padded dataset to %s", processed_path)

    # 3. Feature extraction
    logger.info("Extracting features and labels")
    X_df, y_df = extract_features_dataset(df_padded)
    logger.info(
        "Features shape: %s, Labels shape: %s",
        X_df.shape, y_df.shape
    )

    # 4. Train/test split
    logger.info(
        "Splitting data: test_size=%.2f, random_state=%d",
        test_size, random_state
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df,
        test_size=test_size,
        random_state=random_state
    )
    logger.info(
        "Split complete: %d train / %d test samples",
        len(X_train), len(X_test)
    )

    # 5. Train model pipeline
    logger.info("Training model pipeline (RandomForestRegressor)")
    pipeline = train_model(
        X_train.values,
        y_train.values,
        model_type="RandomForest",
        params={"n_estimators": 200, "random_state": random_state},
        model_save_path=MODEL_SAVE_DIR / "model_pipeline.pkl"
    )

    # In case train_model did not auto-save the pipeline
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    if not (MODEL_SAVE_DIR / "model_pipeline.pkl").exists():
        dump(pipeline, MODEL_SAVE_DIR / "model_pipeline.pkl")
        logger.info(
            "Saved pipeline manually to %s",
            MODEL_SAVE_DIR / "model_pipeline.pkl"
        )

    logger.info("Training complete")
    return pipeline, X_test, y_test


if __name__ == "__main__":
    train_pipeline()
