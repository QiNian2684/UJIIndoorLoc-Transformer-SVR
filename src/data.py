from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .utils import print_section


@dataclass
class RawDataBundle:
    train_df: pd.DataFrame
    eval_df: pd.DataFrame


@dataclass
class DataBundle:
    X_train: np.ndarray
    X_val: np.ndarray
    X_eval: np.ndarray
    y_train_scaled: np.ndarray
    y_val_scaled: np.ndarray
    y_eval_scaled: np.ndarray
    y_train_true: np.ndarray
    y_val_true: np.ndarray
    y_eval_true: np.ndarray
    scaler_X: MinMaxScaler
    scaler_y: Dict[str, StandardScaler]
    feature_names: List[str]
    feature_mask: np.ndarray
    train_df_split: pd.DataFrame
    val_df_split: pd.DataFrame
    eval_df_processed: pd.DataFrame
    eval_indices: np.ndarray
    preprocess_summary: Dict[str, float | int | str]


def inverse_coords(y_scaled: np.ndarray, scaler_y: Dict[str, StandardScaler]) -> np.ndarray:
    longitude = scaler_y["scaler_y_longitude"].inverse_transform(y_scaled[:, 0].reshape(-1, 1))
    latitude = scaler_y["scaler_y_latitude"].inverse_transform(y_scaled[:, 1].reshape(-1, 1))
    return np.hstack((longitude, latitude))


def load_raw_data(train_path: str, eval_path: str) -> RawDataBundle:
    print_section("读取原始数据文件")
    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)
    print(f"[原始数据] train_df.shape={train_df.shape}")
    print(f"[原始数据] eval_df.shape={eval_df.shape}")
    return RawDataBundle(train_df=train_df, eval_df=eval_df)


def load_and_preprocess_data(
    train_path: str,
    eval_path: str,
    feature_missing_threshold: float = 0.97,
    building_id: int = -1,
    val_size: float = 0.1,
    random_state: int = 42,
) -> DataBundle:
    raw = load_raw_data(train_path, eval_path)
    return preprocess_loaded_data(
        raw_data=raw,
        feature_missing_threshold=feature_missing_threshold,
        building_id=building_id,
        val_size=val_size,
        random_state=random_state,
    )


def preprocess_loaded_data(
    raw_data: RawDataBundle,
    feature_missing_threshold: float = 0.97,
    building_id: int = -1,
    val_size: float = 0.1,
    random_state: int = 42,
) -> DataBundle:
    """
    预处理策略：
    1. 使用前 520 列 WAP 特征；
    2. 将 100 替换为 -105；
    3. 可按 BUILDINGID 过滤；
    4. 官方 evaluation 集不再按样本缺失率过滤，始终保留全部 evaluation 样本；
    5. 按训练集缺失率删除特征；
    6. X 使用 MinMaxScaler；
    7. y 的 LONGITUDE / LATITUDE 分别使用 StandardScaler。

    说明：
    - trainingData.csv -> 继续切分为 train/internal validation，用于训练与 Optuna objective。
    - validationData.csv -> 作为官方 held-out evaluation，不参与调参。
    """
    print_section("执行预处理")

    train_df = raw_data.train_df.copy()
    eval_df = raw_data.eval_df.copy()

    raw_train_rows = int(len(train_df))
    raw_eval_rows = int(len(eval_df))

    if building_id not in (-1, 0, 1, 2):
        raise ValueError(f"building_id 只允许为 -1 / 0 / 1 / 2，当前为 {building_id}")

    if building_id != -1:
        if "BUILDINGID" not in train_df.columns or "BUILDINGID" not in eval_df.columns:
            raise KeyError("训练集或 evaluation 集中缺少 BUILDINGID 列，无法按 building_id 过滤。")
        train_df = train_df[train_df["BUILDINGID"] == building_id].copy()
        eval_df = eval_df[eval_df["BUILDINGID"] == building_id].copy()
        print(f"[筛选] building_id={building_id}，仅保留对应 building 的样本")
    else:
        print("[筛选] building_id=-1，使用全部数据")

    if len(train_df) == 0 or len(eval_df) == 0:
        raise ValueError("按当前 building_id 过滤后，训练集或 evaluation 集为空。")

    feature_cols = list(range(520))
    train_features = train_df.iloc[:, feature_cols].replace(100, -105).fillna(-105)
    eval_features = eval_df.iloc[:, feature_cols].replace(100, -105).fillna(-105)

    total_features = int(train_features.shape[1])
    eval_indices = eval_df.index.to_numpy()
    eval_df_processed = eval_df.reset_index(drop=True)
    eval_features = eval_features.reset_index(drop=True)

    train_missing_ratio_per_feature = (train_features == -105).mean(axis=0)
    feature_mask = (train_missing_ratio_per_feature < feature_missing_threshold).to_numpy()

    train_features = train_features.loc[:, feature_mask]
    eval_features = eval_features.loc[:, feature_mask]
    feature_names = train_features.columns.astype(str).tolist()

    kept_feature_count = int(train_features.shape[1])
    if kept_feature_count == 0:
        raise ValueError("按当前 feature_missing_threshold 过滤后，没有保留任何特征。")

    y_train_coords = train_df[["LONGITUDE", "LATITUDE"]].to_numpy()
    y_eval_coords = eval_df_processed[["LONGITUDE", "LATITUDE"]].to_numpy()

    stratify_labels = None
    if {"BUILDINGID", "FLOOR"}.issubset(train_df.columns):
        bf_labels = (
            train_df[["BUILDINGID", "FLOOR"]]
            .astype(int)
            .astype(str)
            .agg("_".join, axis=1)
            .to_numpy()
        )
        unique_labels, counts = np.unique(bf_labels, return_counts=True)
        if len(unique_labels) >= 2 and counts.min() >= 2:
            stratify_labels = bf_labels

    split = train_test_split(
        train_features.to_numpy(),
        y_train_coords,
        train_df.reset_index(drop=False),
        test_size=val_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify_labels,
    )
    X_train_raw, X_val_raw, y_train_coords_split, y_val_coords_split, train_df_split, val_df_split = split

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler_X.fit_transform(X_train_raw)
    X_val = scaler_X.transform(X_val_raw)
    X_eval = scaler_X.transform(eval_features.to_numpy())

    scaler_y_longitude = StandardScaler()
    scaler_y_latitude = StandardScaler()

    y_train_lon = scaler_y_longitude.fit_transform(y_train_coords_split[:, 0].reshape(-1, 1))
    y_val_lon = scaler_y_longitude.transform(y_val_coords_split[:, 0].reshape(-1, 1))
    y_eval_lon = scaler_y_longitude.transform(y_eval_coords[:, 0].reshape(-1, 1))

    y_train_lat = scaler_y_latitude.fit_transform(y_train_coords_split[:, 1].reshape(-1, 1))
    y_val_lat = scaler_y_latitude.transform(y_val_coords_split[:, 1].reshape(-1, 1))
    y_eval_lat = scaler_y_latitude.transform(y_eval_coords[:, 1].reshape(-1, 1))

    y_train_scaled = np.hstack((y_train_lon, y_train_lat)).astype(np.float32)
    y_val_scaled = np.hstack((y_val_lon, y_val_lat)).astype(np.float32)
    y_eval_scaled = np.hstack((y_eval_lon, y_eval_lat)).astype(np.float32)

    for name, array in {
        "X_train": X_train,
        "X_val": X_val,
        "X_eval": X_eval,
        "y_train_scaled": y_train_scaled,
        "y_val_scaled": y_val_scaled,
        "y_eval_scaled": y_eval_scaled,
    }.items():
        if np.isnan(array).any() or np.isinf(array).any():
            raise ValueError(f"{name} 包含 NaN 或 Inf。")

    scaler_y = {
        "scaler_y_longitude": scaler_y_longitude,
        "scaler_y_latitude": scaler_y_latitude,
    }

    y_train_true = inverse_coords(y_train_scaled, scaler_y)
    y_val_true = inverse_coords(y_val_scaled, scaler_y)
    y_eval_true = inverse_coords(y_eval_scaled, scaler_y)

    preprocess_summary = {
        "raw_train_rows": raw_train_rows,
        "raw_eval_rows": raw_eval_rows,
        "filtered_train_rows": int(len(train_df)),
        "filtered_eval_rows": int(len(eval_df_processed)),
        "total_features_before": total_features,
        "total_features_after": kept_feature_count,
        "feature_keep_ratio": float(kept_feature_count / max(total_features, 1)),
        "feature_missing_threshold": float(feature_missing_threshold),
        "building_id": int(building_id),
        "val_size": float(val_size),
        "train_split_rows": int(X_train.shape[0]),
        "val_split_rows": int(X_val.shape[0]),
        "eval_rows": int(X_eval.shape[0]),
        "evaluation_sample_filtering": "disabled",
        "split_policy": "trainingData -> train/internal_validation ; validationData -> official_evaluation",
    }

    print(f"[清洗] feature_missing_threshold={feature_missing_threshold}")
    print(f"[清洗] 原始特征数: {total_features}")
    print(f"[清洗] 保留特征数: {kept_feature_count}")
    print(f"[清洗] evaluation 样本数: {preprocess_summary['eval_rows']}")
    print(f"[清洗] 特征保留率: {preprocess_summary['feature_keep_ratio']:.4f}")
    print(f"[结果] X_train shape: {X_train.shape}")
    print(f"[结果] X_val shape: {X_val.shape}")
    print(f"[结果] X_eval shape: {X_eval.shape}")

    return DataBundle(
        X_train=X_train.astype(np.float32),
        X_val=X_val.astype(np.float32),
        X_eval=X_eval.astype(np.float32),
        y_train_scaled=y_train_scaled,
        y_val_scaled=y_val_scaled,
        y_eval_scaled=y_eval_scaled,
        y_train_true=y_train_true.astype(np.float32),
        y_val_true=y_val_true.astype(np.float32),
        y_eval_true=y_eval_true.astype(np.float32),
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        feature_names=feature_names,
        feature_mask=feature_mask,
        train_df_split=train_df_split.reset_index(drop=True),
        val_df_split=val_df_split.reset_index(drop=True),
        eval_df_processed=eval_df_processed,
        eval_indices=eval_indices,
        preprocess_summary=preprocess_summary,
    )
