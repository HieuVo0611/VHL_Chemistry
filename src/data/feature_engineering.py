import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.fft import fft
from scipy.signal import find_peaks
from scipy.integrate import trapz
import pywt

def extract_peak_features(df):
    y = df['Y'].values
    peaks, properties = find_peaks(y, height=0, prominence=0.1)
    return {
        'num_peaks': len(peaks),
        'max_peak_height': np.max(y[peaks]) if len(peaks) > 0 else 0,
        'mean_peak_prominence': np.mean(properties['prominences']) if len(peaks) > 0 else 0,
        'peak_positions': np.mean(df['X'].iloc[peaks]) if len(peaks) > 0 else 0
    }

def extract_integral_features(df):
    x, y = df['X'].values, df['Y'].values
    bins = np.linspace(min(x), max(x), 11)
    features = {}
    for i in range(len(bins)-1):
        mask = (x >= bins[i]) & (x < bins[i+1])
        integral = trapz(y[mask], x[mask]) if np.sum(mask) > 1 else 0
        features[f'integral_bin_{i}'] = integral
    return features

def extract_derivative_features(df):
    x, y = df['X'].values, df['Y'].values
    # Kiểm tra giá trị X có hợp lệ không
    if len(np.unique(x)) < 2:
        return {
            'dy_dx_mean': 0,
            'dy_dx_std': 0,
            'd2y_dx2_mean': 0,
            'd2y_dx2_std': 0
        }
    # Làm mượt Y để giảm nhiễu
    from scipy.signal import savgol_filter
    y_smooth = savgol_filter(y, window_length=11, polyorder=2)
    # Tính đạo hàm
    dy_dx = np.gradient(y_smooth, x)
    d2y_dx2 = np.gradient(dy_dx, x)
    # Kiểm tra NaN/Inf
    if np.any(np.isnan(dy_dx)) or np.any(np.isinf(dy_dx)):
        dy_dx = np.zeros_like(dy_dx)
    if np.any(np.isnan(d2y_dx2)) or np.any(np.isinf(d2y_dx2)):
        d2y_dx2 = np.zeros_like(d2y_dx2)
    return {
        'dy_dx_mean': np.mean(dy_dx),
        'dy_dx_std': np.std(dy_dx),
        'd2y_dx2_mean': np.mean(d2y_dx2),
        'd2y_dx2_std': np.std(d2y_dx2)
    }

def extract_fft_features(df):
    y = df['Y'].values
    fft_vals = np.abs(fft(y))
    fft_vals = np.clip(fft_vals, 0, 1e6)  # Giới hạn giá trị lớn
    top_k = 5
    top_indices = np.argsort(fft_vals)[-top_k:]
    return {f'fft_freq_{i}': fft_vals[idx] for i, idx in enumerate(top_indices)}

def extract_wavelet_features(df):
    y = df['Y'].values
    coeffs = pywt.wavedec(y, 'db1', level=3)
    features = {}
    for i, c in enumerate(coeffs):
        c = np.nan_to_num(c, nan=0, posinf=0, neginf=0)  # Thay NaN/Inf bằng 0
        features[f'wavelet_level_{i}_energy'] = np.sum(c**2)
    return features

def extract_features_from_sample(df):
    features = {}
    # Kiểm tra dữ liệu đầu vào
    if df['X'].isna().any() or df['Y'].isna().any():
        df = df.fillna(0)
    if df['X'].isin([np.inf, -np.inf]).any() or df['Y'].isin([np.inf, -np.inf]).any():
        df = df.clip(lower=-1e6, upper=1e6)
    # Gọi các hàm trích xuất đặc trưng
    features.update(extract_peak_features(df))
    features.update(extract_integral_features(df))
    features.update(extract_derivative_features(df))
    features.update(extract_fft_features(df))
    features.update(extract_wavelet_features(df))
    for col in ['X', 'Y']:
        series = df[col].values
        features[f'{col}_mean'] = np.mean(series)
        features[f'{col}_std'] = np.std(series)
        features[f'{col}_min'] = np.min(series)
        features[f'{col}_max'] = np.max(series)
        hist, _ = np.histogram(series, bins=10, density=True)
        features[f'{col}_entropy'] = entropy(hist + 1e-10)
        fft_vals = np.abs(fft(series))
        fft_vals = np.clip(fft_vals, 0, 1e6)
        features[f'{col}_fft_power'] = np.sum(fft_vals ** 2) / len(fft_vals)
        x_idx = np.arange(len(series))
        slope = np.polyfit(x_idx, series, 1)[0]
        features[f'{col}_slope'] = slope
    return features

def extract_features_dataset(padded_list):
    features_list = []
    for df in padded_list:
        feats = extract_features_from_sample(df)
        features_list.append(feats)
    features_df = pd.DataFrame(features_list)
    features_df = features_df.fillna(0).clip(lower=-1e6, upper=1e6)
    return features_df