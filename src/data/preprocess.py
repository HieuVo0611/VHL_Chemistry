import pandas as pd

def pad_sample(df, target_length):
    current_length = len(df)
    if current_length >= target_length:
        return df.iloc[:target_length].reset_index(drop=True)
    else:
        # Padding bằng cách lặp lại dòng cuối
        last_row = df.iloc[-1]
        padding_rows = pd.DataFrame([last_row] * (target_length - current_length))
        padded_df = pd.concat([df, padding_rows], ignore_index=True)
        return padded_df

def pad_dataset(data_list):
    # Tìm sample dài nhất
    max_length = max([len(df) for df in data_list])
    padded_list = [pad_sample(df, max_length) for df in data_list]
    return padded_list
