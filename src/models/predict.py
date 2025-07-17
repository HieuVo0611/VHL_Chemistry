import joblib
import numpy as np

def load_model(model_path):
    model = joblib.load(model_path)
    return model

def predict_sample(model, feature_vector):
    feature_vector = np.array(feature_vector).reshape(1, -1)
    prediction_log = model.predict(feature_vector)
    prediction = np.expm1(prediction_log)  # Chuyển về giá trị thực
    return prediction[0]  # Pb hoặc Cd
