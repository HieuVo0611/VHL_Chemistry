import os
from typing import Union, Literal, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor, BaggingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, HuberRegressor, ElasticNet, PassiveAggressiveRegressor, SGDRegressor, TheilSenRegressor

# Thêm import XGBoost
from xgboost import XGBRegressor

def train_model(
        X,
        y,
        model_save_path,
        model_type: Literal[
            'random_forest', 'linear', 'ridge', 'lasso', 'svr', 'knn', 'gbr', 'adaboost', 'xgboost'
        ] = 'random_forest',
        n_estimators: int = 500,
        max_depth: int = 20,
        min_samples_split: int = 4,
        min_samples_leaf: int = 2,
        max_features: Union[str, float] = 'sqrt',
        bootstrap: bool = True,
        alpha: float = 1.0,
        lasso_alpha: float = 0.01,
        C: float = 1.0,
        kernel: str = 'rbf',
        n_neighbors: int = 5,
        random_state=42,
        n_jobs: Optional[int] = None,
        learning_rate: float = 0.05,
        **kwargs
    ):
    # Tách train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=random_state
    )

    # Chọn model với thông số phù hợp
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
    elif model_type == 'xgboost':
        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=n_jobs,
            tree_method='hist'
        )
    elif model_type == 'decision_tree':
        model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    elif model_type == 'extra_trees':
        model = ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=n_jobs)
    elif model_type == 'bagging':
        model = BaggingRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs)
    elif model_type == 'hist_gbr':
        model = HistGradientBoostingRegressor(max_iter=n_estimators, max_depth=max_depth, random_state=random_state)
    elif model_type == 'bayesian_ridge':
        model = BayesianRidge()
    elif model_type == 'huber':
        model = HuberRegressor()
    elif model_type == 'elasticnet':
        model = ElasticNet(random_state=random_state)
    elif model_type == 'passive_aggressive':
        model = PassiveAggressiveRegressor(random_state=random_state, max_iter=1000)
    elif model_type == 'sgd':
        model = SGDRegressor(random_state=random_state, max_iter=1000)
    elif model_type == 'theil_sen':
        model = TheilSenRegressor(random_state=random_state)
    elif model_type == 'linear':
        model = LinearRegression(n_jobs=n_jobs)
    elif model_type == 'ridge':
        model = Ridge(alpha=alpha, random_state=random_state)
    elif model_type == 'lasso':
        model = Lasso(alpha=lasso_alpha, random_state=random_state)
    elif model_type == 'svr':
        model = SVR(C=C, kernel=kernel)
    elif model_type == 'knn':
        model = KNeighborsRegressor(n_neighbors=n_neighbors)
    elif model_type == 'gbr':
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=5,
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
    if model_save_path:
        joblib.dump(model, model_save_path)
    return model
