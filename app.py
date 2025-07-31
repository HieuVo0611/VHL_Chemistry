import shutil
from pathlib import Path
import logging

import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

from src.data.load_data import parse_swp_file
from src.data.preprocess import pad_sample
from src.data.feature_engineering import extract_features_from_sample

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_pipeline(pipeline_path: Path):
    """
    Load the trained sklearn Pipeline (with TransformedTargetRegressor) from disk.
    """
    if not pipeline_path.exists():
        logger.error(f"Pipeline file not found: {pipeline_path}")
        raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")
    pipeline = load(pipeline_path)
    logger.info(f"Loaded pipeline from {pipeline_path}")
    return pipeline

def main() -> None:
    """
    Streamlit app for predicting Pb & Cd concentrations from a voltammogram.
    """
    st.title("Dự đoán nồng độ Pb & Cd từ Voltammogram")

    uploaded_file = st.file_uploader(
        "Upload file dữ liệu (.swp, .xlsx, .csv)",
        type=["swp", "xlsx", "xls", "csv"]
    )
    if not uploaded_file:
        return

    # Save uploaded file to temp folder
    tmp_dir = Path("temp_upload")
    tmp_dir.mkdir(exist_ok=True)
    file_path = tmp_dir / uploaded_file.name
    try:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logger.info(f"Saved upload to {file_path}")

        # Read raw X, Y
        suffix = file_path.suffix.lower()
        if suffix == ".swp":
            df = parse_swp_file(file_path)
        elif suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(file_path, engine="openpyxl")
        else:
            df = pd.read_csv(file_path)
        st.subheader("Xem trước dữ liệu thô")
        st.dataframe(df.head())
        logger.info("Raw data loaded successfully")

        # Load trained pipeline
        pipeline_path = Path("outputs/models/model_pipeline.pkl")
        pipeline = load_pipeline(pipeline_path)

        # Separate X and Y arrays
        x = df['X'].to_numpy()
        y = df['Y'].to_numpy()

        # Determine padding length (from metadata or use current length)
        target_length = getattr(pipeline, "target_length", len(x))
        st.write(f"Padding dữ liệu về độ dài: {target_length}")
        x_pad, y_pad = pad_sample(x, y, target_length=target_length, method="linear")
        df_padded = pd.DataFrame({'X': x_pad, 'Y': y_pad})
        st.write(f"Độ dài sau padding: {len(df_padded)}")
        logger.info("Data padding complete")

        # Extract features
        features = extract_features_from_sample(df_padded)
        st.subheader("Feature đã trích xuất")
        st.json(features)
        logger.info("Feature extraction complete")

        # Predict
        X_vec = np.array(list(features.values())).reshape(1, -1)
        pred = pipeline.predict(X_vec)[0]  # already inverse-log
        pb_pred, cd_pred = pred[0], pred[1]

        st.subheader("Kết quả dự đoán")
        st.write(f"• Pb: **{pb_pred:.4f}** µM")
        st.write(f"• Cd: **{cd_pred:.4f}** µM")
        logger.info(f"Predicted Pb={pb_pred:.4f}, Cd={cd_pred:.4f}")

    except Exception as e:
        logger.exception("Inference error")
        st.error(f"Đã xảy ra lỗi khi dự đoán: {e}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
