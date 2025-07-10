
import joblib
import numpy as np

def load_model(model_path):
    model = joblib.load(model_path)
    return model

def predict_sample(model, feature_vector):
    # feature_vector: 1D np.array hoặc list
    feature_vector = np.array(feature_vector).reshape(1, -1)
    prediction = model.predict(feature_vector)
    return prediction[0]  # [ENR, CIP]
