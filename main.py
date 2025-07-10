from src.data.load_data import load_dataset, load_dataset_V250710
from src.data.preprocess import pad_dataset
from src.data.feature_engineering import extract_features_dataset
from src.models.train_model import train_model
from src.config import *
import pandas as pd
import os
from datetime import datetime


def main():
    use_new_data = False  # Đổi thành False nếu muốn dùng data cũ
    label = 'PB'  # hoặc 'ENR' nếu dùng data cũ

    if use_new_data:
        folder_path = os.path.join(RAW_DATA_DIR, '1053 so lieu Cam bien dien hoa(HP4)', '1053 so lieu phan tich Cd, Pb')
        data_list, labels_list = load_dataset_V250710(folder_path)
    else:
        folder_path = os.path.join(RAW_DATA_DIR, 'pb_cd')
        data_list, labels_list = load_dataset(folder_path=folder_path, label=label)

    # Step 2: Padding samples
    padded_list = pad_dataset(data_list)

    # Step 3: Extract features
    features_df = extract_features_dataset(padded_list)
    labels_df = pd.DataFrame(labels_list, columns=['Pb', 'Cd'])

    # Step 4: Train model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_new_data" if use_new_data else ""
    model_save_path = os.path.join(MODEL_SAVE_PATH, f'random_forest_model{suffix}_{timestamp}.pkl')
    model = train_model(
        X=features_df,
        y=labels_df,
        model_save_path=model_save_path,
        model_type='random_forest',
        n_estimators=1000,
    )

    # Step 5: Predict new data
    # model = joblib.load(model_save_path)


if __name__ == "__main__":
    main()
    print("Training completed successfully!")