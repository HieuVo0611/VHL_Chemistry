import os
import re
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# Function to extract labels from filenames
def extract_labels(filename):
    match = re.search(r'ENR ([\d\.]+)uM - CIP ([\d\.]+)uM', filename)
    if match:
        return float(match.group(1)), float(match.group(2)), f"ENR {match.group(1)}uM - CIP {match.group(2)}uM"
    return None, None, None

# Load all .xlsx files and extract data
path = "./data/cip_enr/"  # Folder containing extracted .xlsx files
files = [f for f in os.listdir(path) if f.endswith(".xlsx")]

data_list = []
labels = []
file_names = []

for file in files:
    file_path = os.path.join(path, file)
    Enr, Cip, title = extract_labels(file)
    
    if Enr is not None and Cip is not None:
        df = pd.read_excel(file_path, sheet_name="Sheet1")
        if {'X', 'Y'}.issubset(df.columns):
            data_list.append(df[['X', 'Y']].values)
            labels.append([Enr, Cip])
            file_names.append(title)

# Convert to NumPy arrays
X = np.array(data_list, dtype=object)  # Variable-length feature vectors
y = np.array(labels)

# Pad sequences to the same length (for ML compatibility)
max_len = max(len(arr) for arr in X)
X_padded = np.array([np.pad(arr, ((0, max_len - len(arr)), (0, 0)), mode='constant') for arr in X])

# Split dataset (60% train, 40% validation)
X_train, X_val, y_train, y_val, train_files, val_files = train_test_split(
    X_padded, y, file_names, test_size=0.4, random_state=42)


# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(len(X_train), -1)).reshape(X_train.shape)
X_val = scaler.transform(X_val.reshape(len(X_val), -1)).reshape(X_val.shape)

# Hyperparameter tuning with GridSearchCV
# param_grid_rfr = {
#     'n_estimators': [200, 500, 1000],
#     'max_depth': [None, 10, 30, 50],
#     'min_samples_split': [2, 5, 10, 15],
#     'min_samples_leaf': [1, 2, 4, 6],
#     'max_features': [1,'sqrt', 'log2'],
#     'bootstrap': [True, False],
#     'criterion' :['squared_error', 'absolute_error', 'poisson', 'friedman_mse'],
# }

# param_grid_xgb = {
#     'n_estimators': [200, 500, 1000], 
#     'learning_rate': [0.001,0.005, 0.01, 0.05, 0.1], 
#     'max_depth': [None, 10, 30, 50],
#     'min_child_weight': [1, 3, 5, 7],
#     'subsample': [0.5, 0.7, 1],
#     'colsample_bytree': [0.5, 0.7, 1],
#     'objective': ['reg:squarederror', 'reg:squaredlogerror', 'reg:logistic'],
# }

param_grid_xgb = {
    'colsample_bytree': [0.5], 
    'learning_rate': [0.01], 
    'max_depth': [None], 
    'min_child_weight': [1], 
    'n_estimators': [1000], 
    'objective': ['reg:squarederror'], 
    'subsample': [0.7]}

# param_grid_rfr = {
#     'bootstrap': [False], 
#     'criterion': ['friedman_mse'], 
#     'max_depth': [None], 
#     'max_features': ['log2'], 
#     'min_samples_leaf': [2], 
#     'min_samples_split': [2], 
#     'n_estimators': [200],
# }

start_time = time.time()
# grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rfr, cv=3, scoring='neg_mean_absolute_error', n_jobs=-7)
grid_search = GridSearchCV(XGBRegressor(random_state=42, verbosity=0), param_grid_xgb, cv=3, scoring='neg_mean_absolute_error', n_jobs=-7)
grid_search.fit(X_train.reshape(len(X_train), -1), y_train)

# Train Random Forest Regressor with best parameters
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# model = RandomForestRegressor(**best_params, random_state=42)
model = XGBRegressor(**best_params, random_state=42, verbosity=0, n_jobs=-7)
model.fit(X_train.reshape(len(X_train), -1), y_train)

# Predict on validation set
y_pred = model.predict(X_val.reshape(len(X_val), -1))

# Evaluate model
mae = mean_absolute_error(y_val, y_pred)
print(f"Mean Absolute Error: {np.around(mae,2)}")

end_time = time.time()

# # Visualize a random sample from the validation set
# random_idx = np.random.randint(0, len(X_val))
# sample_data = X_val[random_idx]
# sample_title = val_files[random_idx]

# plt.figure(figsize=(8, 6))
# plt.plot(sample_data[:, 0], sample_data[:, 1], marker='o', linestyle='-', label=sample_title)
# plt.title(sample_title)
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.legend()
# plt.show()

# Function to predict Enr and Cip from a new .xlsx file
def predict_labels(file_path):
    df = pd.read_excel(file_path, sheet_name="Sheet1")
    if {'X', 'Y'}.issubset(df.columns):
        features = df[['X', 'Y']].values
        features_padded = np.pad(features, ((0, max_len - len(features)), (0, 0)), mode='constant')
        features_scaled = scaler.transform([features_padded.flatten()])
        prediction = model.predict(features_scaled)[0]
        print(f"Predicted Enr: {np.around(prediction[0],2)}, Predicted Cip: {np.around(prediction[1],2)}")
        return prediction
    return None

# Example usage: predict on a new file
new_file = "./data/cip_enr/ENR 0.5uM - CIP 0.5uM.xlsx" # Path to new .xlsx file to predict (Example: ENR 0.5uM - CIP 0.5uM.xlsx)
predict_labels(new_file)
print(f"Execution Time: {end_time - start_time:.2f} seconds")