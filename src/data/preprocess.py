"""
src/data/preprocess.py

Module for uniform-length padding of voltammogram samples.
"""

from typing import Literal, Tuple
import logging

import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def pad_sample(
    x: np.ndarray,
    y: np.ndarray,
    target_length: int,
    method: Literal['constant', 'edge', 'linear'] = 'linear',
    fill_value: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pad or crop one (x, y) voltammogram to exactly target_length points.

    Args:
        x: 1D array of X values.
        y: 1D array of Y values.
        target_length: Desired output length.
        method: Padding method ('constant', 'edge', 'linear').
        fill_value: Value used for constant padding of Y.

    Returns:
        Tuple of (x_padded, y_padded), each length = target_length.
    """
    n = len(x)
    if n == target_length:
        return x, y
    if n > target_length:
        # Crop evenly from both ends
        start = (n - target_length) // 2
        return x[start:start+target_length], y[start:start+target_length]
    # Need to pad
    pad_len = target_length - n
    if method == 'constant':
        x_pad = np.full(pad_len, x[-1])
        y_pad = np.full(pad_len, fill_value)
    elif method == 'edge':
        # extend X by constant step, Y by last value
        step = (x[-1] - x[-2]) if n > 1 else 1.0
        x_pad = x[-1] + np.arange(1, pad_len+1) * step
        y_pad = np.full(pad_len, y[-1])
    else:  # 'linear'
        # linear extrapolation based on last two points
        if n > 1:
            dx = x[-1] - x[-2]
            dy = y[-1] - y[-2]
        else:
            dx, dy = 1.0, 0.0
        x_pad = x[-1] + np.arange(1, pad_len+1) * dx
        y_pad = y[-1] + np.arange(1, pad_len+1) * dy
    x_padded = np.concatenate([x, x_pad])
    y_padded = np.concatenate([y, y_pad])
    return x_padded, y_padded


def pad_dataset(
    df: pd.DataFrame,
    method: Literal['constant', 'edge', 'linear'] = 'linear',
    fill_value: float = 0.0
) -> pd.DataFrame:
    """
    Pad all samples in a combined DataFrame to the same length.
    Expects columns ['X', 'Y', 'pb', 'cd', 'sample_id'].

    Args:
        df: Combined DataFrame with one row per measurement point and a 'sample_id' column.
        method: Padding method passed to pad_sample.
        fill_value: Value used for constant padding of Y.

    Returns:
        A DataFrame where each sample (grouped by sample_id) has exactly
        the same number of rows = maximum original length.
        Columns preserved: ['X', 'Y', 'pb', 'cd', 'sample_id'].
    """
    if 'sample_id' not in df.columns:
        raise KeyError("DataFrame must include 'sample_id' column before padding.")

    # Determine maximum sample length
    lengths = df.groupby('sample_id').size()
    max_len = int(lengths.max())
    logger.info("Padding dataset to uniform length: %d (max of samples)", max_len)

    padded_frames = []
    for sid, group in df.groupby('sample_id'):
        x_arr = group['X'].to_numpy()
        y_arr = group['Y'].to_numpy()
        x_pad, y_pad = pad_sample(x_arr, y_arr, max_len, method, fill_value)

        # Reconstruct DataFrame for this sample
        df_pad = pd.DataFrame({
            'X': x_pad,
            'Y': y_pad,
            'pb': group['pb'].iloc[0],
            'cd': group['cd'].iloc[0],
            'sample_id': sid
        })
        padded_frames.append(df_pad)

    # Concatenate all padded samples
    df_padded = pd.concat(padded_frames, ignore_index=True)
    logger.info("Completed padding all %d samples", len(padded_frames))
    return df_padded
