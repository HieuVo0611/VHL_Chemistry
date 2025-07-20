import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

def pad_sample(df, target_length):
    # Kiểm tra và xử lý NaN/Inf
    if df['X'].isna().any() or df['Y'].isna().any():
        print(f"Warning: NaN found in sample, filling with mean")
        df = df.fillna({'X': df['X'].mean(), 'Y': df['Y'].mean()})
    if df['X'].isin([np.inf, -np.inf]).any() or df['Y'].isin([np.inf, -np.inf]).any():
        print(f"Warning: Inf found in sample, clipping values")
        df = df.clip(lower=-1e6, upper=1e6)
    
    # Đảm bảo X tăng đơn điệu
    if not np.all(np.diff(df['X']) > 0):
        print("Warning: X is not strictly increasing, sorting")
        df = df.sort_values('X').reset_index(drop=True)
    
    # Làm mượt Y
    if len(df) > 10:
        df['Y'] = savgol_filter(df['Y'], window_length=11, polyorder=2)
    
    current_length = len(df)
    if current_length >= target_length:
        return df.iloc[:target_length].reset_index(drop=True)
    else:
        x_step = (df['X'].iloc[-1] - df['X'].iloc[0]) / (current_length - 1) if current_length > 1 else 0.01
        x_last = df['X'].iloc[-1]
        pad_length = target_length - current_length
        x_pad = x_last + x_step * np.arange(1, pad_length + 1)
        y_last = df['Y'].iloc[-1]
        y_pad = np.full(pad_length, y_last)
        
        padding_rows = pd.DataFrame({'X': x_pad, 'Y': y_pad})
        padded_df = pd.concat([df, padding_rows], ignore_index=True)
        
        padded_df = padded_df.fillna({'X': padded_df['X'].mean(), 'Y': padded_df['Y'].mean()})
        padded_df = padded_df.clip(lower=-1e6, upper=1e6)
        return padded_df

def pad_dataset(data_list):
    max_length = max([len(df) for df in data_list])
    padded_list = [pad_sample(df, max_length) for df in data_list]
    
    for i, sample_df in enumerate(padded_list[:3]):
        print(f"Sample {i}:")
        print("NaN in sample:\n", sample_df.isna().sum())
        print("Inf in sample:\n", sample_df.isin([np.inf, -np.inf]).sum())
    
    return padded_list