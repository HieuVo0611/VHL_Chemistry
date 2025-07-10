import os
from typing import Union, Literal, Optional
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def train_model(
        X,
        y,
        model_save_path,
        model_type: Literal[
            'random_forest', 'linear', 'ridge', 'lasso', 'svr', 'knn', 'gbr', 'adaboost'
        ] = 'random_forest',
        n_estimators: int = 200,
        max_depth: int = 20,
        min_samples_split: float = 4,
        min_samples_leaf: float = 2,
        max_features: Union[float, Literal['sqrt', 'log2']] = 'sqrt',
        bootstrap: bool = True,
        random_state=42,
        n_jobs: Optional[int] = None,
        **kwargs
    ):
    # Tách train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=random_state
    )

    # Chọn model
    if model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state,
            n_jobs=n_jobs
        )
    elif model_type == 'linear':
        model = LinearRegression(n_jobs=n_jobs)
    elif model_type == 'ridge':
        model = Ridge(random_state=random_state)
    elif model_type == 'lasso':
        model = Lasso(random_state=random_state)
    elif model_type == 'svr':
        model = SVR()
    elif model_type == 'knn':
        model = KNeighborsRegressor()
    elif model_type == 'gbr':
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
    elif model_type == 'adaboost':
        model = AdaBoostRegressor(
            n_estimators=n_estimators,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_train, y_train)

    # Predict & Eval
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{model_type} trained! MSE: {mse:.4f} | R2: {r2:.4f}")

    # Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    print(f"Model saved at: {model_save_path}")

    return model
