import logging
import numpy as np
from pathlib import Path
from joblib import load

from src.config import MODEL_SAVE_PATH
from src.data.preprocess import pad_sample
from src.data.feature_engineering import extract_features_from_sample

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def predict_sample(
    df_sample
) -> float:
    """
    Given a raw voltammogram sample DataFrame with ['X','Y'], predict concentration.
    """
    # Pad and extract features
    df_padded = pad_sample(df_sample, target_length=len(df_sample))
    features = extract_features_from_sample(df_padded)
    X = np.array(list(features.values())).reshape(1, -1)

    # Load pipeline
    pipeline_path = MODEL_SAVE_PATH / 'model_pipeline.pkl'
    pipeline = load(pipeline_path)

    # Predict and inverse log1p
    pred_log = pipeline.predict(X)
    pred = np.expm1(pred_log)[0]
    logger.info(f"Predicted concentration: {pred:.4f}")
    return pred
