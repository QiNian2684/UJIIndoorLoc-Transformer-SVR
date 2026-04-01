from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

def build_svr_params(best_params: Dict[str, object]) -> Dict[str, object]:
    return {"kernel": best_params["svr_kernel"], "C": float(best_params["svr_C"]), "epsilon": float(best_params["svr_epsilon"]), "gamma": best_params["svr_gamma"], "degree": int(best_params["svr_degree"]), "coef0": float(best_params["svr_coef0"]), "cache_size": 256, "shrinking": True}

def compute_error_distances(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.linalg.norm(y_true - y_pred, axis=1)

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
    coord_mse = mean_squared_error(y_true, y_pred)
    coord_mae = mean_absolute_error(y_true, y_pred)
    coord_r2 = r2_score(y_true, y_pred)
    lon_mae = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    lat_mae = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    lon_mse = mean_squared_error(y_true[:, 0], y_pred[:, 0])
    lat_mse = mean_squared_error(y_true[:, 1], y_pred[:, 1])
    lon_rmse = math.sqrt(float(lon_mse))
    lat_rmse = math.sqrt(float(lat_mse))
    lon_bias = float(np.mean(y_pred[:, 0] - y_true[:, 0]))
    lat_bias = float(np.mean(y_pred[:, 1] - y_true[:, 1]))
    dist = compute_error_distances(y_true, y_pred)
    distance_rmse = math.sqrt(float(np.mean(dist ** 2)))
    distance_mean = float(np.mean(dist))
    distance_median = float(np.median(dist))
    metrics = {
        "mse": float(coord_mse), "mae": float(coord_mae), "rmse": float(distance_rmse), "mean": float(distance_mean), "middle": float(distance_median),
        "coord_mse": float(coord_mse), "coord_mae": float(coord_mae), "coord_r2": float(coord_r2),
        "longitude_mae": float(lon_mae), "latitude_mae": float(lat_mae), "longitude_mse": float(lon_mse), "latitude_mse": float(lat_mse),
        "longitude_rmse": float(lon_rmse), "latitude_rmse": float(lat_rmse), "longitude_bias": lon_bias, "latitude_bias": lat_bias,
        "distance_rmse": float(distance_rmse), "distance_mean": float(distance_mean), "distance_median": float(distance_median), "distance_std": float(np.std(dist)),
        "distance_var": float(np.var(dist)), "distance_q10": float(np.quantile(dist, 0.10)), "distance_q25": float(np.quantile(dist, 0.25)),
        "distance_q50": float(np.quantile(dist, 0.50)), "distance_q75": float(np.quantile(dist, 0.75)), "distance_q90": float(np.quantile(dist, 0.90)),
        "distance_q95": float(np.quantile(dist, 0.95)), "distance_q99": float(np.quantile(dist, 0.99)), "distance_iqr": float(np.quantile(dist, 0.75) - np.quantile(dist, 0.25)),
        "distance_max": float(np.max(dist)), "distance_min": float(np.min(dist)), "sample_count": int(len(dist)),
    }
    return metrics, dist

def fit_svr(X_train: np.ndarray, y_train: np.ndarray, svr_params: Dict[str, object]) -> MultiOutputRegressor:
    regressor = MultiOutputRegressor(SVR(**svr_params))
    regressor.fit(X_train, y_train)
    return regressor
