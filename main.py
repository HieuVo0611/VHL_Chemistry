from src.data.load_data import load_dataset
from src.data.preprocess import pad_dataset
from src.data.feature_engineering import extract_features_dataset
from src.models.train_model import train_model
from src.config import *
import pandas as pd
import os
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
import numpy as np


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, r2


def compute_sample_weights(y):
    # y là 1D array (Pb hoặc Cd)
    hist, bin_edges = np.histogram(y, bins=10)
    bin_idx = np.digitize(y, bin_edges[:-1], right=True)
    freq = np.array([hist[i-1] for i in bin_idx])
    weights = 1 / (freq + 1e-6)
    return weights


def main():
    retrain = True  # Đặt True nếu muốn train lại toàn bộ, False để bỏ qua model đã train

    folder_path = os.path.join(PROCESSED_DATA_DIR, 'metadata_pb_cd.csv')
    df = pd.read_csv(folder_path)
    # Group theo Filenames để lấy từng sample
    data_list = []
    labels_list = []
    for sample, group in df.groupby('Filenames'):
        sample_df = group[['X', 'Y']]
        data_list.append(sample_df)
        pb_label = group['Pb'].iloc[0]
        cd_label = group['Cd'].iloc[0]
        labels_list.append([pb_label, cd_label])

    padded_list = pad_dataset(data_list)
    features_df = extract_features_dataset(padded_list)
    labels_df = pd.DataFrame(labels_list, columns=['Pb', 'Cd'])

    # Chia train/val/test 6:2:2
    X_temp, X_test, y_temp, y_test = train_test_split(features_df, labels_df, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    # Gộp train+val để cross-validation
    X_trainval = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_trainval = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    # Các model và thông số
    model_configs = [
        ('random_forest', dict(n_estimators=500, max_depth=20, min_samples_split=4, min_samples_leaf=2, max_features='sqrt', bootstrap=True)),
        ('xgboost', dict(n_estimators=500, max_depth=8, learning_rate=0.05)),
        ('decision_tree', dict(max_depth=20)),
        ('extra_trees', dict(n_estimators=500, max_depth=20)),
        ('bagging', dict(n_estimators=200)),
        ('hist_gbr', dict(n_estimators=500, max_depth=10)),
        ('bayesian_ridge', dict()),
        ('huber', dict()),
        ('elasticnet', dict()),
        ('passive_aggressive', dict()),
        ('sgd', dict()),
        ('theil_sen', dict()),
        ('linear', dict()),
        ('ridge', dict()),
        ('lasso', dict()),
        ('svr', dict()),
        ('knn', dict()),
        ('gbr', dict(n_estimators=500, max_depth=5)),
        ('adaboost', dict(n_estimators=200)),
    ]

    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for model_type, params in model_configs:
        for target in ['Pb', 'Cd']:
            model_filename = f'{model_type}_{target}_{timestamp}.pkl'
            model_save_path = os.path.join(MODEL_SAVE_PATH, model_filename)
            # Kiểm tra nếu model đã tồn tại và retrain=False thì bỏ qua
            if not retrain:
                # Kiểm tra bất kỳ file model nào cùng tên (bỏ timestamp)
                prefix = f'{model_type}_{target}_'
                existed = any(f.startswith(prefix) and f.endswith('.pkl') for f in os.listdir(MODEL_SAVE_PATH))
                if existed:
                    print(f"Skip {model_type} for {target} (already trained)")
                    continue

            print(f"Training {model_type} for {target}...")
            y_target = y_trainval[target]
            # KFold CV
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_metrics = []
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_trainval)):
                X_tr, X_val = X_trainval.iloc[train_idx], X_trainval.iloc[val_idx]
                y_tr, y_val = y_target.iloc[train_idx], y_target.iloc[val_idx]
                sample_weight = compute_sample_weights(y_tr.values)
                model = train_model(
                    X=X_tr, y=y_tr, model_save_path=None, model_type=model_type, **params
                )
                try:
                    model.fit(X_tr, y_tr, sample_weight=sample_weight)
                except TypeError:
                    model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)
                mae = mean_absolute_error(y_val, y_pred)
                mse = mean_squared_error(y_val, y_pred)
                r2 = r2_score(y_val, y_pred)
                fold_metrics.append((mae, mse, r2))
            # Train lại trên toàn bộ train+val để lưu model cuối cùng
            model_save_path = os.path.join(MODEL_SAVE_PATH, f'{model_type}_{target}_{timestamp}.pkl')
            final_model = train_model(
                X=X_trainval, y=y_target, model_save_path=model_save_path, model_type=model_type, **params
            )
            # Đánh giá trên test set
            y_test_target = y_test[target]
            y_test_pred = final_model.predict(X_test)
            test_mae = mean_absolute_error(y_test_target, y_test_pred)
            test_mse = mean_squared_error(y_test_target, y_test_pred)
            test_r2 = r2_score(y_test_target, y_test_pred)
            mean_mae, mean_mse, mean_r2 = np.mean(fold_metrics, axis=0)
            results.append([
                f"{model_type}_{target}_{timestamp}", 'metadata_pb_cd.csv',
                mean_mae, mean_mse, mean_r2, test_mae, test_mse, test_r2
            ])

    # Ghi kết quả ra file csv
    results_df = pd.DataFrame(results, columns=['Model', 'Data', 'CV_MAE', 'CV_MSE', 'CV_R2', 'Test_MAE', 'Test_MSE', 'Test_R2'])
    results_path = os.path.join(PROCESSED_DATA_DIR, 'train_results.csv')
    results_df.to_csv(results_path, index=False)
    print("Training & evaluation completed. Results saved to", results_path)


if __name__ == "__main__":
    main()
    print("Training completed successfully!")