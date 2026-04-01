from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


CALIBRATION_OOF_FOLDS = 5
CORRECTION_CLIP_QUANTILE = 0.995
CATBOOST_DEFAULT_PARAMS: dict[str, Any] = {
    "iterations": 700,
    "depth": 8,
    "learning_rate": 0.05,
    "l2_leaf_reg": 5.0,
    "random_strength": 1.0,
    "bagging_temperature": 1.0,
    "border_count": 128,
    "min_data_in_leaf": 8,
    "grow_policy": "SymmetricTree",
    "leaf_estimation_iterations": 3,
}


@dataclass
class ResidualCalibrationBundle:
    enabled: bool
    feature_mode: str | None = None
    pca_components: int | None = None
    scaler: StandardScaler | None = None
    pca: PCA | None = None
    global_model: CatBoostRegressor | None = None
    route_classifier: CatBoostClassifier | None = None
    route_clusterer: KMeans | None = None
    route_mode: str | None = None
    route_feature_dim: int = 0
    route_label_vocab: list[str] | None = None
    expert_models: dict[str, CatBoostRegressor] = field(default_factory=dict)
    expert_sample_count: dict[str, int] = field(default_factory=dict)
    train_feature_bank: np.ndarray | None = None
    train_residual_bank: np.ndarray | None = None
    neighbor_k: int = 8
    blend_alpha: float = 0.5
    route_min_samples: int = 64
    max_correction_norm: float | None = None
    fit_sample_count: int = 0
    target_mode: str | None = None
    grouping_summary: Dict[str, Any] | None = None
    train_metrics: Dict[str, float] | None = None
    config: Dict[str, Any] | None = None


def compute_reconstruction_error(
    signal_values: np.ndarray | None,
    reconstructed_signals: np.ndarray | None,
) -> np.ndarray | None:
    if signal_values is None or reconstructed_signals is None:
        return None
    return np.asarray(signal_values, dtype=np.float64) - np.asarray(reconstructed_signals, dtype=np.float64)


def _require_array(name: str, value: np.ndarray | None) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} is required for the selected stage-2 feature mode.")
    return np.asarray(value, dtype=np.float64)


def build_residual_features(
    feature_mode: str,
    latent_features: np.ndarray,
    base_predictions: np.ndarray,
    reconstruction_error: np.ndarray | None = None,
    signal_values: np.ndarray | None = None,
) -> np.ndarray:
    latent_features = np.asarray(latent_features, dtype=np.float64)
    base_predictions = np.asarray(base_predictions, dtype=np.float64)
    if feature_mode == "pred_only":
        return base_predictions
    if feature_mode == "latent_only":
        return latent_features
    if feature_mode == "latent_pred":
        return np.hstack([latent_features, base_predictions])
    if feature_mode == "recon_error":
        return _require_array("reconstruction_error", reconstruction_error)
    if feature_mode == "recon_error_pred":
        return np.hstack([_require_array("reconstruction_error", reconstruction_error), base_predictions])
    if feature_mode == "full_stack":
        return np.hstack([_require_array("reconstruction_error", reconstruction_error), latent_features, base_predictions])
    if feature_mode in {"raw_rssi", "paper_residual"}:
        return _require_array("signal_values", signal_values)
    if feature_mode in {"raw_rssi_pred", "paper_residual_pred", "rssi_pred"}:
        return np.hstack([_require_array("signal_values", signal_values), base_predictions])
    if feature_mode in {"raw_rssi_recon_error", "rssi_recon"}:
        return np.hstack([_require_array("signal_values", signal_values), _require_array("reconstruction_error", reconstruction_error)])
    if feature_mode in {"raw_rssi_latent_pred", "rssi_latent_pred"}:
        return np.hstack([_require_array("signal_values", signal_values), latent_features, base_predictions])
    if feature_mode == "full_stage2":
        return np.hstack([
            _require_array("signal_values", signal_values),
            latent_features,
            base_predictions,
            _require_array("reconstruction_error", reconstruction_error),
        ])
    raise ValueError(f"Unknown feature_mode: {feature_mode}")


def _resolve_residual_targets(
    *,
    y_true: np.ndarray,
    base_predictions: np.ndarray,
    group_labels: np.ndarray | None,
    target_mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    residual = np.asarray(y_true, dtype=np.float64) - np.asarray(base_predictions, dtype=np.float64)
    if group_labels is None or target_mode == "pointwise":
        return residual, {
            "target_mode": str(target_mode),
            "group_count": 0 if group_labels is None else int(len(np.unique(group_labels))),
            "uses_group_aggregation": False,
        }
    labels = np.asarray(group_labels).astype(str)
    aggregated = residual.copy()
    unique_labels = np.unique(labels)
    for label in unique_labels:
        idx = np.where(labels == label)[0]
        if len(idx) == 0:
            continue
        value = np.median(residual[idx], axis=0, keepdims=True) if target_mode == "group_median" else np.mean(residual[idx], axis=0, keepdims=True)
        aggregated[idx] = value
    return aggregated, {
        "target_mode": str(target_mode),
        "group_count": int(len(unique_labels)),
        "uses_group_aggregation": True,
    }


def _sample_indices(n_samples: int, max_samples: int | None, random_state: int) -> np.ndarray:
    if max_samples is None or max_samples <= 0 or n_samples <= max_samples:
        return np.arange(n_samples)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n_samples, size=max_samples, replace=False))


def _effective_oof_splits(n_samples: int, requested_splits: int) -> int:
    if n_samples < 2:
        return 1
    return max(2, min(int(requested_splits), int(n_samples)))


def _can_use_stratified_kfold(labels: np.ndarray | None, n_splits: int) -> bool:
    if labels is None:
        return False
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) == 0:
        return False
    unique_labels, counts = np.unique(labels, return_counts=True)
    return len(unique_labels) >= 2 and int(counts.min()) >= int(n_splits)


def _build_catboost_params(model_params: dict[str, Any] | None, random_state: int, loss_function: str) -> dict[str, Any]:
    params = dict(CATBOOST_DEFAULT_PARAMS)
    if model_params:
        params.update(model_params)
    params.update({
        "loss_function": loss_function,
        "eval_metric": loss_function,
        "random_seed": int(random_state),
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": -1,
    })
    return params


def _route_base_features(base_predictions: np.ndarray) -> np.ndarray:
    return np.asarray(base_predictions, dtype=np.float64)


def _compute_neighbor_summary(X_query: np.ndarray, X_ref: np.ndarray, residual_ref: np.ndarray | None, k: int) -> tuple[np.ndarray, np.ndarray | None]:
    if len(X_ref) == 0 or len(X_query) == 0:
        dist_feat = np.zeros((len(X_query), 3), dtype=np.float64)
        return dist_feat, None if residual_ref is None else np.zeros((len(X_query), residual_ref.shape[1]), dtype=np.float64)
    effective_k = max(1, min(int(k), len(X_ref)))
    nn = NearestNeighbors(n_neighbors=effective_k, metric="euclidean")
    nn.fit(X_ref)
    distances, indices = nn.kneighbors(X_query, return_distance=True)
    dist_feat = np.column_stack([
        distances[:, 0],
        np.mean(distances, axis=1),
        np.std(distances, axis=1),
    ])
    if residual_ref is None:
        return dist_feat, None
    neigh_residual = residual_ref[indices]
    residual_std = np.std(neigh_residual, axis=1)
    return dist_feat, residual_std


def clip_corrections(correction: np.ndarray, max_norm: float | None) -> np.ndarray:
    if max_norm is None or max_norm <= 0:
        return np.asarray(correction, dtype=np.float64)
    clipped = np.asarray(correction, dtype=np.float64).copy()
    norms = np.linalg.norm(clipped, axis=1)
    mask = norms > max_norm
    if np.any(mask):
        clipped[mask] *= (max_norm / np.maximum(norms[mask], 1e-12)).reshape(-1, 1)
    return clipped


def _transform_core_features(
    *,
    feature_mode: str,
    latent_features: np.ndarray,
    base_predictions: np.ndarray,
    reconstruction_error: np.ndarray | None,
    signal_values: np.ndarray | None,
    scaler: StandardScaler | None,
    pca: PCA | None,
) -> np.ndarray:
    X = build_residual_features(feature_mode, latent_features, base_predictions, reconstruction_error, signal_values)
    if scaler is not None:
        X = scaler.transform(X)
    if pca is not None:
        X = pca.transform(X)
    return np.asarray(X, dtype=np.float64)


def _build_route_labels(
    *,
    route_mode: str,
    route_labels: np.ndarray | None,
    route_clusterer: KMeans | None,
    base_predictions: np.ndarray,
) -> np.ndarray | None:
    if route_mode == "global_only":
        return None
    if route_mode == "pred_cluster":
        if route_clusterer is None:
            return None
        cluster_id = route_clusterer.predict(_route_base_features(base_predictions))
        return np.asarray([f"cluster_{int(v)}" for v in cluster_id], dtype=object)
    if route_mode == "region_cluster":
        if route_clusterer is None:
            return None
        cluster_id = route_clusterer.predict(_route_base_features(base_predictions))
        return np.asarray([f"region_{int(v)}" for v in cluster_id], dtype=object)
    if route_mode == "building_floor":
        if route_labels is None:
            return None
        return np.asarray(route_labels).astype(str)
    if route_mode == "building_floor_region":
        if route_labels is None or route_clusterer is None:
            return None
        cluster_id = route_clusterer.predict(_route_base_features(base_predictions))
        return np.asarray([f"{str(bf)}|region_{int(c)}" for bf, c in zip(route_labels, cluster_id)], dtype=object)
    raise ValueError(f"Unknown route_mode: {route_mode}")


def _fit_route_classifier(X_model: np.ndarray, labels: np.ndarray, random_state: int, model_params: dict[str, Any] | None) -> tuple[CatBoostClassifier, list[str]]:
    vocab = sorted(np.unique(labels).astype(str).tolist())
    vocab_to_int = {label: idx for idx, label in enumerate(vocab)}
    y_int = np.asarray([vocab_to_int[str(v)] for v in labels], dtype=np.int64)
    params = _build_catboost_params(model_params, random_state, "MultiClass")
    params.pop("eval_metric", None)
    clf = CatBoostClassifier(**params)
    clf.fit(X_model, y_int)
    return clf, vocab


def _predict_route_distribution(bundle: ResidualCalibrationBundle, X_model: np.ndarray, base_predictions: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    if bundle.route_mode in {None, "global_only"}:
        return None, None
    if bundle.route_mode in {"pred_cluster", "region_cluster"} and bundle.route_clusterer is not None:
        cluster_id = bundle.route_clusterer.predict(_route_base_features(base_predictions))
        prefix = "cluster" if bundle.route_mode == "pred_cluster" else "region"
        labels = np.asarray([f"{prefix}_{int(v)}" for v in cluster_id], dtype=object)
        confidence = np.ones(len(labels), dtype=np.float64)
        return labels, confidence
    if bundle.route_classifier is not None and bundle.route_label_vocab:
        proba = np.asarray(bundle.route_classifier.predict_proba(X_model), dtype=np.float64)
        idx = np.argmax(proba, axis=1)
        labels = np.asarray([bundle.route_label_vocab[int(i)] for i in idx], dtype=object)
        confidence = np.max(proba, axis=1)
        return labels, confidence
    return None, None


def _ensure_2d_prediction(pred: np.ndarray, n_samples: int) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    if pred.ndim == 1:
        pred = pred.reshape(n_samples, -1)
    return pred


def _build_model_features(bundle: ResidualCalibrationBundle, core_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    neighbor_feat, neighbor_residual_std = _compute_neighbor_summary(
        core_features,
        np.asarray(bundle.train_feature_bank, dtype=np.float64),
        bundle.train_residual_bank,
        bundle.neighbor_k,
    )
    if neighbor_residual_std is None:
        neighbor_residual_std = np.zeros((len(core_features), 2), dtype=np.float64)
    X_model = np.hstack([core_features, neighbor_feat, neighbor_residual_std])
    return X_model.astype(np.float64), neighbor_residual_std.astype(np.float64)


def fit_residual_calibrator(
    *,
    latent_features: np.ndarray,
    base_predictions: np.ndarray,
    y_true: np.ndarray,
    reconstruction_error: np.ndarray | None = None,
    signal_values: np.ndarray | None = None,
    feature_mode: str,
    random_state: int,
    max_train_samples: int | None = None,
    pca_components: int | None = None,
    correction_clip_quantile: float | None = CORRECTION_CLIP_QUANTILE,
    group_labels: np.ndarray | None = None,
    target_mode: str = "pointwise",
    model_params: dict[str, Any] | None = None,
    route_mode: str = "region_cluster",
    route_labels: np.ndarray | None = None,
    route_cluster_count: int = 8,
    route_min_samples: int = 64,
    neighbor_k: int = 8,
    blend_alpha: float = 0.5,
) -> ResidualCalibrationBundle:
    residual_target, grouping_summary = _resolve_residual_targets(
        y_true=np.asarray(y_true, dtype=np.float64),
        base_predictions=np.asarray(base_predictions, dtype=np.float64),
        group_labels=None if group_labels is None else np.asarray(group_labels),
        target_mode=str(target_mode),
    )
    raw_features = build_residual_features(
        feature_mode,
        latent_features,
        base_predictions,
        reconstruction_error,
        signal_values,
    )
    sample_idx = _sample_indices(raw_features.shape[0], max_train_samples, random_state)
    X_fit_raw = np.asarray(raw_features[sample_idx], dtype=np.float64)
    y_fit = np.asarray(residual_target[sample_idx], dtype=np.float64)
    base_fit = np.asarray(base_predictions[sample_idx], dtype=np.float64)
    route_fit_labels = None if route_labels is None else np.asarray(route_labels)[sample_idx]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_fit_raw)
    pca = None
    max_valid_components = int(min(X_scaled.shape[0], X_scaled.shape[1]))
    if pca_components is not None and pca_components > 0 and max_valid_components >= 2:
        effective_pca_components = min(int(pca_components), max_valid_components)
        if effective_pca_components < X_scaled.shape[1]:
            pca = PCA(n_components=effective_pca_components, random_state=int(random_state))
            X_core = pca.fit_transform(X_scaled)
        else:
            X_core = X_scaled
    else:
        X_core = X_scaled

    train_neighbor_feat, train_neighbor_residual_std = _compute_neighbor_summary(X_core, X_core, y_fit, neighbor_k)
    if len(train_neighbor_feat) > 0:
        train_neighbor_feat[:, 0] = np.where(np.isclose(train_neighbor_feat[:, 0], 0.0), train_neighbor_feat[:, 1], train_neighbor_feat[:, 0])
    X_model = np.hstack([X_core, train_neighbor_feat, train_neighbor_residual_std]).astype(np.float64)

    reg_params = _build_catboost_params(model_params, random_state, "MultiRMSE")
    global_model = CatBoostRegressor(**reg_params)
    global_model.fit(X_model, y_fit)

    route_clusterer = None
    if route_mode in {"pred_cluster", "region_cluster", "building_floor_region"}:
        unique_candidate = max(1, min(int(route_cluster_count), len(base_fit)))
        if unique_candidate >= 2:
            route_clusterer = KMeans(n_clusters=unique_candidate, n_init=10, random_state=int(random_state))
            route_clusterer.fit(_route_base_features(base_fit))

    resolved_route_labels = _build_route_labels(
        route_mode=str(route_mode),
        route_labels=route_fit_labels,
        route_clusterer=route_clusterer,
        base_predictions=base_fit,
    )

    route_classifier = None
    route_label_vocab = None
    expert_models: dict[str, CatBoostRegressor] = {}
    expert_sample_count: dict[str, int] = {}
    if resolved_route_labels is not None:
        labels_arr = np.asarray(resolved_route_labels).astype(str)
        unique_labels, counts = np.unique(labels_arr, return_counts=True)
        keep_labels = [str(label) for label, count in zip(unique_labels, counts) if int(count) >= int(route_min_samples)]
        if len(keep_labels) >= 2 and route_mode in {"building_floor", "building_floor_region"}:
            keep_mask = np.isin(labels_arr, keep_labels)
            route_classifier, route_label_vocab = _fit_route_classifier(X_model[keep_mask], labels_arr[keep_mask], int(random_state), model_params)
        elif len(keep_labels) >= 1:
            route_label_vocab = keep_labels
        for idx_label, label in enumerate(keep_labels):
            idx = np.where(labels_arr == label)[0]
            if len(idx) < int(route_min_samples):
                continue
            expert_model = CatBoostRegressor(**_build_catboost_params(model_params, int(random_state) + idx_label + 1, "MultiRMSE"))
            expert_model.fit(X_model[idx], y_fit[idx])
            expert_models[str(label)] = expert_model
            expert_sample_count[str(label)] = int(len(idx))

    correction_norm = np.linalg.norm(y_fit, axis=1)
    max_correction_norm = None if correction_clip_quantile is None else float(np.quantile(correction_norm, float(correction_clip_quantile))) if len(correction_norm) > 0 else None

    bundle = ResidualCalibrationBundle(
        enabled=True,
        feature_mode=str(feature_mode),
        pca_components=None if pca is None else int(pca.n_components_),
        scaler=scaler,
        pca=pca,
        global_model=global_model,
        route_classifier=route_classifier,
        route_clusterer=route_clusterer,
        route_mode=str(route_mode),
        route_feature_dim=2,
        route_label_vocab=route_label_vocab,
        expert_models=expert_models,
        expert_sample_count=expert_sample_count,
        train_feature_bank=np.asarray(X_core, dtype=np.float64),
        train_residual_bank=np.asarray(y_fit, dtype=np.float64),
        neighbor_k=int(neighbor_k),
        blend_alpha=float(blend_alpha),
        route_min_samples=int(route_min_samples),
        max_correction_norm=max_correction_norm,
        fit_sample_count=int(len(sample_idx)),
        target_mode=str(target_mode),
        grouping_summary=grouping_summary,
        config={
            "feature_mode": str(feature_mode),
            "target_mode": str(target_mode),
            "route_mode": str(route_mode),
            "route_cluster_count": int(route_cluster_count),
            "route_min_samples": int(route_min_samples),
            "neighbor_k": int(neighbor_k),
            "blend_alpha": float(blend_alpha),
            "model_params": {} if model_params is None else dict(model_params),
            "max_train_samples": None if max_train_samples is None else int(max_train_samples),
            "pca_components": None if pca is None else int(pca.n_components_),
            "correction_clip_quantile": None if correction_clip_quantile is None else float(correction_clip_quantile),
            "expert_count": int(len(expert_models)),
        },
    )

    residual_mean, residual_std = predict_residual_with_uncertainty(
        bundle,
        latent_features=np.asarray(latent_features, dtype=np.float64),
        base_predictions=np.asarray(base_predictions, dtype=np.float64),
        reconstruction_error=None if reconstruction_error is None else np.asarray(reconstruction_error, dtype=np.float64),
        signal_values=None if signal_values is None else np.asarray(signal_values, dtype=np.float64),
    )
    residual_pred_fit = residual_mean[sample_idx]
    residual_error = y_fit - residual_pred_fit
    bundle.train_metrics = {
        "residual_rmse_longitude": float(np.sqrt(np.mean(residual_error[:, 0] ** 2))),
        "residual_rmse_latitude": float(np.sqrt(np.mean(residual_error[:, 1] ** 2))),
        "correction_std_mean": float(np.mean(np.linalg.norm(residual_std[sample_idx], axis=1))),
        "fit_sample_count": int(len(sample_idx)),
        "raw_feature_dim": int(raw_features.shape[1]) if raw_features.ndim == 2 else 0,
        "effective_feature_dim": int(X_model.shape[1]) if X_model.ndim == 2 else 0,
        "expert_count": int(len(expert_models)),
        "route_mode": str(route_mode),
    }
    return bundle


def predict_residual_with_uncertainty(
    bundle: ResidualCalibrationBundle,
    *,
    latent_features: np.ndarray,
    base_predictions: np.ndarray,
    reconstruction_error: np.ndarray | None = None,
    signal_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not bundle.enabled or bundle.global_model is None or bundle.scaler is None:
        zero = np.zeros_like(base_predictions, dtype=np.float64)
        return zero, zero
    X_core = _transform_core_features(
        feature_mode=bundle.feature_mode or "pred_only",
        latent_features=latent_features,
        base_predictions=base_predictions,
        reconstruction_error=reconstruction_error,
        signal_values=signal_values,
        scaler=bundle.scaler,
        pca=bundle.pca,
    )
    X_model, neighbor_residual_std = _build_model_features(bundle, X_core)
    global_pred = _ensure_2d_prediction(bundle.global_model.predict(X_model), len(X_model))
    final_pred = global_pred.copy()
    route_labels_pred, route_conf = _predict_route_distribution(bundle, X_model, np.asarray(base_predictions, dtype=np.float64))
    disagreement = np.zeros_like(final_pred, dtype=np.float64)
    if route_labels_pred is not None:
        if route_conf is None:
            route_conf = np.ones(len(route_labels_pred), dtype=np.float64)
        for label, model in bundle.expert_models.items():
            idx = np.where(route_labels_pred == label)[0]
            if len(idx) == 0:
                continue
            expert_pred = _ensure_2d_prediction(model.predict(X_model[idx]), len(idx))
            strength = min(1.0, float(bundle.expert_sample_count.get(label, 0)) / max(1.0, float(bundle.route_min_samples)))
            weight = np.clip(float(bundle.blend_alpha) * strength * route_conf[idx], 0.0, 1.0).reshape(-1, 1)
            disagreement[idx] = np.abs(expert_pred - global_pred[idx])
            final_pred[idx] = (1.0 - weight) * global_pred[idx] + weight * expert_pred
    std_proxy = np.maximum(0.0, 0.7 * neighbor_residual_std + 0.3 * disagreement)
    final_pred = clip_corrections(final_pred, bundle.max_correction_norm)
    return final_pred.astype(np.float64), std_proxy.astype(np.float64)


def apply_residual_calibration(
    bundle: ResidualCalibrationBundle | None,
    *,
    latent_features: np.ndarray,
    base_predictions: np.ndarray,
    reconstruction_error: np.ndarray | None = None,
    signal_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if bundle is None or not bundle.enabled:
        zero = np.zeros_like(base_predictions, dtype=np.float64)
        return np.asarray(base_predictions, dtype=np.float64), zero, zero
    residual_mean, residual_std = predict_residual_with_uncertainty(
        bundle,
        latent_features=latent_features,
        base_predictions=base_predictions,
        reconstruction_error=reconstruction_error,
        signal_values=signal_values,
    )
    corrected = np.asarray(base_predictions, dtype=np.float64) + residual_mean
    return corrected, residual_mean, residual_std


def build_oof_residual_predictions(
    *,
    latent_features: np.ndarray,
    base_predictions: np.ndarray,
    y_true: np.ndarray,
    reconstruction_error: np.ndarray | None = None,
    signal_values: np.ndarray | None = None,
    feature_mode: str,
    random_state: int,
    max_train_samples: int | None = None,
    pca_components: int | None = None,
    n_splits: int = CALIBRATION_OOF_FOLDS,
    stratify_labels: np.ndarray | None = None,
    group_labels: np.ndarray | None = None,
    target_mode: str = "pointwise",
    correction_clip_quantile: float | None = CORRECTION_CLIP_QUANTILE,
    model_params: dict[str, Any] | None = None,
    route_mode: str = "region_cluster",
    route_labels: np.ndarray | None = None,
    route_cluster_count: int = 8,
    route_min_samples: int = 64,
    neighbor_k: int = 8,
    blend_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    n_samples = int(len(base_predictions))
    corrected_oof = np.asarray(base_predictions, dtype=np.float64).copy()
    correction_oof = np.zeros_like(corrected_oof, dtype=np.float64)
    correction_std_oof = np.zeros_like(corrected_oof, dtype=np.float64)
    if n_samples < 2:
        return corrected_oof, correction_oof, correction_std_oof, {
            "strategy": "fallback_no_oof",
            "requested_splits": int(n_splits),
            "effective_splits": 1,
            "sample_count": n_samples,
            "reason": "validation sample count is smaller than 2",
            "folds": [],
        }

    effective_splits = _effective_oof_splits(n_samples, n_splits)
    labels = None if stratify_labels is None else np.asarray(stratify_labels)
    groups = None if group_labels is None else np.asarray(group_labels)
    routing = None if route_labels is None else np.asarray(route_labels)
    if groups is not None and len(np.unique(groups)) >= 2:
        effective_splits = max(2, min(int(effective_splits), int(len(np.unique(groups)))))
        splitter = GroupKFold(n_splits=effective_splits)
        split_iter = splitter.split(np.arange(n_samples), groups=groups)
        split_strategy = "group_kfold"
    elif _can_use_stratified_kfold(labels, effective_splits):
        splitter = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=int(random_state))
        split_iter = splitter.split(np.arange(n_samples), labels)
        split_strategy = "stratified_kfold"
    else:
        splitter = KFold(n_splits=effective_splits, shuffle=True, random_state=int(random_state))
        split_iter = splitter.split(np.arange(n_samples))
        split_strategy = "kfold"

    fold_summaries: list[dict[str, Any]] = []
    covered = np.zeros(n_samples, dtype=bool)
    for fold_idx, (train_idx, valid_idx) in enumerate(split_iter, start=1):
        bundle = fit_residual_calibrator(
            latent_features=np.asarray(latent_features[train_idx], dtype=np.float64),
            base_predictions=np.asarray(base_predictions[train_idx], dtype=np.float64),
            y_true=np.asarray(y_true[train_idx], dtype=np.float64),
            reconstruction_error=None if reconstruction_error is None else np.asarray(reconstruction_error[train_idx], dtype=np.float64),
            signal_values=None if signal_values is None else np.asarray(signal_values[train_idx], dtype=np.float64),
            feature_mode=str(feature_mode),
            random_state=int(random_state) + fold_idx,
            max_train_samples=max_train_samples,
            pca_components=pca_components,
            correction_clip_quantile=correction_clip_quantile,
            group_labels=None if groups is None else np.asarray(groups[train_idx]),
            target_mode=str(target_mode),
            model_params=model_params,
            route_mode=str(route_mode),
            route_labels=None if routing is None else np.asarray(routing[train_idx]),
            route_cluster_count=int(route_cluster_count),
            route_min_samples=int(route_min_samples),
            neighbor_k=int(neighbor_k),
            blend_alpha=float(blend_alpha),
        )
        corrected_fold, correction_fold, correction_std_fold = apply_residual_calibration(
            bundle,
            latent_features=np.asarray(latent_features[valid_idx], dtype=np.float64),
            base_predictions=np.asarray(base_predictions[valid_idx], dtype=np.float64),
            reconstruction_error=None if reconstruction_error is None else np.asarray(reconstruction_error[valid_idx], dtype=np.float64),
            signal_values=None if signal_values is None else np.asarray(signal_values[valid_idx], dtype=np.float64),
        )
        corrected_oof[valid_idx] = corrected_fold
        correction_oof[valid_idx] = correction_fold
        correction_std_oof[valid_idx] = correction_std_fold
        covered[valid_idx] = True
        fold_summaries.append({
            "fold_index": int(fold_idx),
            "train_count": int(len(train_idx)),
            "valid_count": int(len(valid_idx)),
            "fit_sample_count": int(bundle.fit_sample_count),
            "route_mode": None if bundle.config is None else bundle.config.get("route_mode"),
            "expert_count": None if bundle.config is None else bundle.config.get("expert_count"),
            "train_metrics": bundle.train_metrics,
        })
    if not np.all(covered):
        raise RuntimeError("OOF calibration prediction did not cover all validation samples.")
    summary = {
        "strategy": "kfold_oof_residual_calibration",
        "split_strategy": split_strategy,
        "requested_splits": int(n_splits),
        "effective_splits": int(effective_splits),
        "sample_count": int(n_samples),
        "correction_clip_quantile": None if correction_clip_quantile is None else float(correction_clip_quantile),
        "target_mode": str(target_mode),
        "route_mode": str(route_mode),
        "route_cluster_count": int(route_cluster_count),
        "route_min_samples": int(route_min_samples),
        "neighbor_k": int(neighbor_k),
        "blend_alpha": float(blend_alpha),
        "group_count": 0 if groups is None else int(len(np.unique(groups))),
        "folds": fold_summaries,
    }
    return corrected_oof, correction_oof, correction_std_oof, summary
