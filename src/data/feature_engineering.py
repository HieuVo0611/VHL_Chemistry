"""
src/data/feature_engineering.py

Module for extracting features from each voltammogram sample,
and for batching that over a full dataset.
"""

import logging
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy.fft import rfft, rfftfreq
import pywt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def extract_features_from_sample(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract a set of features from one padded voltammogram.
    Input df must have columns ['X', 'Y'].
    Returns a dict mapping feature names to values.
    """
    x = df['X'].values
    y = df['Y'].values

    # 1. Smooth Y
    y_smooth = savgol_filter(y, window_length=7, polyorder=3)

    # 2. Peak features
    peaks, _ = find_peaks(y_smooth)
    peak_vals = y_smooth[peaks] if peaks.size else np.array([0.0])
    features: Dict[str, float] = {
        'peak_count': float(peaks.size),
        'peak_mean': float(np.mean(peak_vals)),
        'peak_max': float(np.max(peak_vals)),
    }

    # 3. Integral feature
    features['integral'] = float(np.trapz(y_smooth, x))

    # 4. Derivative features
    dy = np.gradient(y_smooth, x)
    features['deriv_mean'] = float(np.mean(dy))
    features['deriv_max'] = float(np.max(dy))

    # 5. FFT features (first 5 coefficients)
    yf = rfft(y_smooth)
    xf = rfftfreq(len(y_smooth), (x[1] - x[0]) if len(x) > 1 else 1.0)
    for i in range(min(5, len(yf))):
        features[f'fft_{i}'] = float(np.abs(yf[i]))

    # 6. Wavelet energy (approximation coeffs)
    coeffs = pywt.wavedec(y_smooth, 'db1', level=2)
    features['wavelet_energy'] = float(sum(np.sum(c**2) for c in coeffs))

    return features


def extract_features_dataset(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    From a combined padded DataFrame with columns
    ['X','Y','pb','cd','sample_id'], extract features per sample.

    Returns:
      - X_df: DataFrame where each row is the feature dict of one sample.
      - y_df: DataFrame with columns ['pb','cd'] of true labels.
    """
    if 'sample_id' not in df.columns:
        raise KeyError("DataFrame must include 'sample_id' for dataset extraction.")

    feature_dicts = []
    label_dicts = []

    for sid, group in df.groupby('sample_id'):
        try:
            feat = extract_features_from_sample(group[['X', 'Y']])
        except Exception as e:
            logger.error(f"Failed to extract features for sample {sid}: {e}")
            continue

        feature_dicts.append(feat)
        label_dicts.append({
            'pb': float(group['pb'].iloc[0]),
            'cd': float(group['cd'].iloc[0])
        })

    if not feature_dicts:
        raise ValueError("No features extracted from any sample.")

    X_df = pd.DataFrame(feature_dicts)
    y_df = pd.DataFrame(label_dicts)

    logger.info(
        "Extracted features for %d samples: X_df shape=%s, y_df shape=%s",
        len(X_df), X_df.shape, y_df.shape
    )
    return X_df, y_df
