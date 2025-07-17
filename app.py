import streamlit as st
import pandas as pd
import numpy as np
import os
from src.models.predict import load_model, predict_sample
from src.data.feature_engineering import extract_features_from_sample
from src.data.preprocess import pad_sample
from src.config import MODEL_SAVE_PATH

def read_swp_file(uploaded_file):
    lines = uploaded_file.read().decode('utf-8').splitlines()
    data_start = False
    data = []
    for line in lines:
        if 'Horizonal Axis' in line:
            data_start = True
            continue
        if data_start and line.strip() == '':
            continue
        if data_start:
            values = line.strip().split()
            if len(values) == 2:
                try:
                    data.append([float(values[0]), float(values[1])])
                except Exception:
                    continue
    df = pd.DataFrame(data, columns=['X', 'Y'])
    return df

def main():
    st.set_page_config(page_title="Chemistry Model Demo", layout="wide")
    st.title("Chemistry Model Demo")
    st.markdown("Upload a data file (.swp/.csv/.xlsx), select model, and predict.")

    model_files = [f for f in os.listdir(MODEL_SAVE_PATH) if f.endswith('.pkl')]
    model_file = st.selectbox("Select model file", model_files)

    uploaded_file = st.file_uploader("Upload sample file (.swp/.xlsx/.csv)", type=["swp", "xlsx", "csv"])
    if uploaded_file:
        if uploaded_file.name.endswith('.swp'):
            df = read_swp_file(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        st.write("Sample Data:", df.head())

        df_padded = pad_sample(df, target_length=len(df))
        features = extract_features_from_sample(df_padded)
        st.write("Extracted Features:", features)

        model_path = os.path.join(MODEL_SAVE_PATH, model_file)
        model = load_model(model_path)
        pred = predict_sample(model, list(features.values()))
        st.success(f"Prediction: {pred:.4f}")

if __name__ == "__main__":
    main()