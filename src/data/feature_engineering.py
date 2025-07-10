import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.fft import fft

def extract_features_from_sample(df):
    features = {}

    for col in ['X', 'Y']:
        series = df[col].values

        # Thống kê cơ bản
        features[f'{col}_mean'] = np.mean(series)
        features[f'{col}_std'] = np.std(series)
        features[f'{col}_min'] = np.min(series)
        features[f'{col}_max'] = np.max(series)

        # Entropy
        hist, _ = np.histogram(series, bins=10, density=True)
        features[f'{col}_entropy'] = entropy(hist + 1e-10)  # tránh log(0)

        # FFT power
        fft_vals = np.abs(fft(series))
        features[f'{col}_fft_power'] = np.sum(fft_vals ** 2) / len(fft_vals)

        # Trend (slope)
        x_idx = np.arange(len(series))
        slope = np.polyfit(x_idx, series, 1)[0]
        features[f'{col}_slope'] = slope

    return features

def extract_features_dataset(padded_list):
    features_list = []
    for df in padded_list:
        feats = extract_features_from_sample(df)
        features_list.append(feats)
    return pd.DataFrame(features_list)
