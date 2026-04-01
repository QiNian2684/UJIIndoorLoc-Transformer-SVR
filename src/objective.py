from __future__ import annotations

import gc
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, KFold

from .calibration import CALIBRATION_OOF_FOLDS, apply_residual_calibration, fit_residual_calibrator
from .config import ExperimentConfig
from .data import DataBundle, RawDataBundle, preprocess_loaded_data
from .evaluate import build_svr_params, evaluate_predictions, fit_svr
from .inference import load_inference_bundle
from .models import WiFiTransformerAutoencoder
from .optuna_compat import optuna
from .storage import (
    append_live_event,
    build_trial_report_text,
    get_trial_dir,
    save_history_csv,
    save_model_bundle,
    save_predictions_csv,
    save_text_report,
    save_trial_metadata,
    update_live_status,
)
from .train import extract_feature_payload, extract_features, train_autoencoder
from .utils import save_json, set_seed, wall_clock
from .visualize import (
    save_error_boxplot,
    save_error_cdf,
    save_error_histogram,
    save_loss_curve,
    save_metric_trace,
    save_residual_scatter,
    save_true_vs_pred_scatter,
)


@dataclass
class BaseCandidateRecord:
    trial_number: int
    trial_dir: Path
    objective_value: float
    params: Dict[str, Any]


@dataclass
class PreparedBaseCandidate:
    record: BaseCandidateRecord
    data: DataBundle
    bundle: Dict[str, Any]
    X_train_feat: np.ndarray
    X_val_feat: np.ndarray
    X_eval_feat: np.ndarray
    train_reconstruction_error: np.ndarray
    val_reconstruction_error: np.ndarray
    eval_reconstruction_error: np.ndarray
    base_train_pred: np.ndarray
    base_val_pred: np.ndarray
    base_eval_pred: np.ndarray
    base_train_metrics: Dict[str, float]
    base_val_metrics: Dict[str, float]
    base_eval_metrics: Dict[str, float]
    base_train_oof_summary: Dict[str, Any]


def _derive_position_group_labels(meta_df: pd.DataFrame) -> np.ndarray | None:
    candidate_cols = ["BUILDINGID", "FLOOR", "SPACEID", "RELATIVEPOSITION"]
    if set(candidate_cols).issubset(meta_df.columns):
        return (
            meta_df[candidate_cols]
            .fillna(-9999)
            .astype(str)
            .agg("|".join, axis=1)
            .to_numpy()
        )
    coord_cols = ["LONGITUDE", "LATITUDE"]
    if set(coord_cols).issubset(meta_df.columns):
        return (
            meta_df[coord_cols]
            .round(4)
            .astype(str)
            .agg("|".join, axis=1)
            .to_numpy()
        )
    return None




class _TrialArtifactsMixin:
    @staticmethod
    def _build_prediction_df(
        meta_df: pd.DataFrame,
        y_true: np.ndarray,
        base_pred: np.ndarray,
        final_pred: np.ndarray,
        correction: np.ndarray,
        correction_std: np.ndarray,
        final_error_dist: np.ndarray,
    ) -> pd.DataFrame:
        base_error = np.linalg.norm(base_pred - y_true, axis=1)
        result_df = pd.DataFrame(
            {
                "true_LONGITUDE": y_true[:, 0],
                "true_LATITUDE": y_true[:, 1],
                "base_pred_LONGITUDE": base_pred[:, 0],
                "base_pred_LATITUDE": base_pred[:, 1],
                "pred_LONGITUDE": final_pred[:, 0],
                "pred_LATITUDE": final_pred[:, 1],
                "catboost_correction_LONGITUDE": correction[:, 0],
                "catboost_correction_LATITUDE": correction[:, 1],
                "catboost_std_LONGITUDE": correction_std[:, 0],
                "catboost_std_LATITUDE": correction_std[:, 1],
                "base_error_distance": base_error,
                "error_distance": final_error_dist,
                "residual_LONGITUDE": final_pred[:, 0] - y_true[:, 0],
                "residual_LATITUDE": final_pred[:, 1] - y_true[:, 1],
            }
        )
        optional_meta_cols = ["index", "BUILDINGID", "FLOOR", "SPACEID", "RELATIVEPOSITION", "USERID", "PHONEID"]
        for col in optional_meta_cols:
            if col in meta_df.columns:
                output_col = "original_index" if col == "index" else col
                result_df[output_col] = meta_df[col].to_numpy()
        ordered_cols = [
            "original_index",
            "BUILDINGID",
            "FLOOR",
            "SPACEID",
            "RELATIVEPOSITION",
            "USERID",
            "PHONEID",
            "true_LONGITUDE",
            "true_LATITUDE",
            "base_pred_LONGITUDE",
            "base_pred_LATITUDE",
            "pred_LONGITUDE",
            "pred_LATITUDE",
            "catboost_correction_LONGITUDE",
            "catboost_correction_LATITUDE",
            "catboost_std_LONGITUDE",
            "catboost_std_LATITUDE",
            "base_error_distance",
            "error_distance",
            "residual_LONGITUDE",
            "residual_LATITUDE",
        ]
        return result_df[[c for c in ordered_cols if c in result_df.columns]]

    @staticmethod
    def _empty_prediction_csv(path: Path) -> None:
        pd.DataFrame(
            columns=[
                "true_LONGITUDE",
                "true_LATITUDE",
                "base_pred_LONGITUDE",
                "base_pred_LATITUDE",
                "pred_LONGITUDE",
                "pred_LATITUDE",
                "catboost_correction_LONGITUDE",
                "catboost_correction_LATITUDE",
                "catboost_std_LONGITUDE",
                "catboost_std_LATITUDE",
                "base_error_distance",
                "error_distance",
            ]
        ).to_csv(path, index=False, encoding="utf-8-sig")

    @staticmethod
    def _count_model_params(model: torch.nn.Module) -> int:
        return int(sum(p.numel() for p in model.parameters()))

    @staticmethod
    def _save_prediction_artifacts(
        trial_dir: Path,
        prefix: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        error_dist: np.ndarray,
    ) -> None:
        save_true_vs_pred_scatter(y_true, y_pred, trial_dir / f"{prefix}_true_vs_pred.png", f"{prefix} true vs pred")
        save_error_histogram(error_dist, trial_dir / f"{prefix}_error_hist.png", f"{prefix} error histogram")
        save_error_boxplot(error_dist, trial_dir / f"{prefix}_error_box.png", f"{prefix} error box")
        save_error_cdf(error_dist, trial_dir / f"{prefix}_error_cdf.png", f"{prefix} error cdf")
        save_residual_scatter(y_true, y_pred, trial_dir / f"{prefix}_residual_scatter.png", f"{prefix} residual scatter")


class BaseTrialObjective(_TrialArtifactsMixin):
    def __init__(self, raw_data: RawDataBundle, config: ExperimentConfig, device: torch.device, experiment_root: str | Path) -> None:
        self.raw_data = raw_data
        self.config = config
        self.device = device
        self.experiment_root = Path(experiment_root)

    @staticmethod
    def suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
        model_dim = trial.suggest_categorical("model_dim", [64, 128, 192, 256, 512])
        valid_heads = [h for h in [2, 4, 8, 16, 32, 64] if model_dim % h == 0]
        num_heads = trial.suggest_categorical("num_heads", valid_heads)
        svr_kernel = trial.suggest_categorical("svr_kernel", ["rbf"])
        params: Dict[str, Any] = {
            "model_name": "WiFiTransformerAutoencoder(masked-weighted AE) + MultiOutputRegressor(SVR)",
            "search_stage": "base",
            "ae_loss_name": "masked_weighted_reconstruction_mse",
            "ae_observed_loss_weight": 1.0,
            "ae_missing_loss_weight": trial.suggest_float("ae_missing_loss_weight", 0.25, 5.0, log=True),
            "ae_missing_value_threshold": 1e-6,
            "feature_missing_threshold": trial.suggest_categorical("feature_missing_threshold", [0.98, 0.99]),
            "model_dim": model_dim,
            "num_heads": num_heads,
            "num_layers": trial.suggest_categorical("num_layers", [3, 4, 5, 6, 7, 8]),
            "ff_multiplier": trial.suggest_categorical("ff_multiplier", [1, 2, 4, 8, 16]),
            "dropout": trial.suggest_float("dropout", 0.00, 0.10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 8e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-9, 1e-4, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64]),
            "early_stopping_patience": trial.suggest_int("early_stopping_patience", 1, 5),
            "min_delta_ratio": trial.suggest_float("min_delta_ratio", 3e-6, 2.5e-3, log=True),
            "svr_kernel": svr_kernel,
            "svr_C": trial.suggest_float("svr_C", 10, 5e2, log=True),
            "svr_epsilon": trial.suggest_float("svr_epsilon", 2e-4, 3e-1, log=True),
            "svr_gamma": "scale",
            "svr_degree": 3,
            "svr_coef0": 0.0,
            "calibration_model": "none",
        }
        if svr_kernel in ["rbf", "poly"]:
            params["svr_gamma"] = trial.suggest_categorical("svr_gamma", ["scale"])
        if svr_kernel == "poly":
            params["svr_degree"] = trial.suggest_int("svr_degree", 2, 5)
            params["svr_coef0"] = trial.suggest_float("svr_coef0", 0.0, 2.0)
        return params

    def __call__(self, trial: optuna.Trial) -> float:
        trial_start = wall_clock()
        trial_dir = get_trial_dir(self.experiment_root, trial.number)
        params = self.suggest_params(trial)
        history: Dict[str, Any] | None = None
        model: WiFiTransformerAutoencoder | None = None
        data: DataBundle | None = None
        status_prefix = f"base trial={trial.number + 1:04d}/{self.config.optuna.base_n_trials:04d}"
        history_path = trial_dir / "training_history.csv"
        try:
            set_seed(
                self.config.seed,
                deterministic=self.config.reproducibility.deterministic,
                strict_deterministic_algorithms=self.config.reproducibility.strict_deterministic_algorithms,
                deterministic_warn_only=self.config.reproducibility.deterministic_warn_only,
                cublas_workspace_config=self.config.reproducibility.cublas_workspace_config,
            )
            print("\n" + "#" * 100)
            print(f"[BaseTrialStart] {status_prefix}")
            print(f"[BaseTrialStart] folder={trial_dir}")
            print(f"[BaseTrialStart] params={params}")
            data = preprocess_loaded_data(
                raw_data=self.raw_data,
                feature_missing_threshold=float(params["feature_missing_threshold"]),
                building_id=self.config.preprocess.building_id,
                val_size=self.config.preprocess.val_size,
                random_state=self.config.seed,
            )
            save_json(data.preprocess_summary, trial_dir / "preprocess_summary.json")
            model = WiFiTransformerAutoencoder(
                input_dim=int(data.X_train.shape[1]),
                model_dim=int(params["model_dim"]),
                num_heads=int(params["num_heads"]),
                num_layers=int(params["num_layers"]),
                dropout=float(params["dropout"]),
                ff_multiplier=int(params["ff_multiplier"]),
            ).to(self.device)
            model_num_parameters = self._count_model_params(model)
            save_json({
                "model_num_parameters": model_num_parameters,
                "transformer_config": {
                    "input_dim": int(data.X_train.shape[1]),
                    "model_dim": int(params["model_dim"]),
                    "num_heads": int(params["num_heads"]),
                    "num_layers": int(params["num_layers"]),
                    "dropout": float(params["dropout"]),
                    "ff_multiplier": int(params["ff_multiplier"]),
                },
            }, trial_dir / "model_summary.json")
            model, history = train_autoencoder(
                model=model,
                X_train=data.X_train,
                X_val=data.X_val,
                device=self.device,
                epochs=self.config.training.max_epochs,
                batch_size=int(params["batch_size"]),
                learning_rate=float(params["learning_rate"]),
                early_stopping_patience=int(params["early_stopping_patience"]),
                min_delta_ratio=float(params["min_delta_ratio"]),
                train_log_interval=self.config.training.train_log_interval,
                num_workers=self.config.training.num_workers,
                weight_decay=float(params["weight_decay"]),
                observed_loss_weight=float(params["ae_observed_loss_weight"]),
                missing_loss_weight=float(params["ae_missing_loss_weight"]),
                missing_value_threshold=float(params["ae_missing_value_threshold"]),
                trial=trial,
                enable_pruning=self.config.optuna.enable_pruning,
                status_prefix=status_prefix,
                history_save_path=history_path,
            )
            save_history_csv(history, history_path)
            save_loss_curve(history, trial_dir / "loss_curve.png", log_scale=False)
            save_loss_curve(history, trial_dir / "loss_curve_logscale.png", log_scale=True)
            save_metric_trace(history, trial_dir / "training_monitor_trace.png")
            if bool(history.get("pruned", False)):
                metrics = {
                    "status": "pruned",
                    "objective_name": "validation_base_distance_mean",
                    "objective_value": None,
                    "model_num_parameters": model_num_parameters,
                    "training": {
                        "best_epoch": int(history.get("best_epoch", 0)),
                        "best_val_loss": float(history.get("best_val_loss", float("inf"))),
                        "pruned_epoch": int(history.get("pruned_epoch") or 0),
                        "runtime_seconds": float(wall_clock() - trial_start),
                        "stop_reason": history.get("stop_reason", "optuna_pruned"),
                    },
                }
                save_trial_metadata(trial_dir, trial.number, None, params, metrics, "pruned")
                save_text_report(build_trial_report_text(trial.number, "pruned", params, data.preprocess_summary, history, None, None, {"message": "This base-stage trial was pruned during transformer training."}), trial_dir / "trial_report.txt")
                self._empty_prediction_csv(trial_dir / "predictions_validation.csv")
                self._empty_prediction_csv(trial_dir / "predictions_evaluation.csv")
                trial.set_user_attr("trial_dir", str(trial_dir))
                trial.set_user_attr("status", "pruned")
                trial.set_user_attr("stage", "base")
                raise optuna.TrialPruned(f"base trial {trial.number} pruned at epoch {history.get('pruned_epoch')}")
            train_payload = extract_feature_payload(
                model,
                data.X_train,
                self.device,
                self.config.training.feature_batch_size,
                f"{status_prefix} train_feature_extraction",
                self.config.training.num_workers,
                include_reconstruction=False,
            )
            val_payload = extract_feature_payload(
                model,
                data.X_val,
                self.device,
                self.config.training.feature_batch_size,
                f"{status_prefix} validation_feature_extraction",
                self.config.training.num_workers,
                include_reconstruction=True,
            )
            eval_payload = extract_feature_payload(
                model,
                data.X_eval,
                self.device,
                self.config.training.feature_batch_size,
                f"{status_prefix} evaluation_feature_extraction",
                self.config.training.num_workers,
                include_reconstruction=True,
            )
            X_train_feat = train_payload["latent_features"]
            X_val_feat = val_payload["latent_features"]
            X_eval_feat = eval_payload["latent_features"]
            val_reconstruction_error = data.X_val.astype(np.float32) - val_payload["reconstruction"].astype(np.float32)
            eval_reconstruction_error = data.X_eval.astype(np.float32) - eval_payload["reconstruction"].astype(np.float32)
            svr_params = build_svr_params(params)
            print(f"[SVR] {status_prefix} fitting base SVR with params={svr_params}")
            svr_model = fit_svr(X_train_feat, data.y_train_true, svr_params)
            base_val_pred = np.asarray(svr_model.predict(X_val_feat), dtype=np.float64)
            base_eval_pred = np.asarray(svr_model.predict(X_eval_feat), dtype=np.float64)
            validation_metrics, validation_error_dist = evaluate_predictions(data.y_val_true, base_val_pred)
            evaluation_metrics, eval_error_dist = evaluate_predictions(data.y_eval_true, base_eval_pred)
            zero_corr_val = np.zeros_like(base_val_pred, dtype=np.float64)
            zero_corr_eval = np.zeros_like(base_eval_pred, dtype=np.float64)
            val_pred_df = self._build_prediction_df(data.val_df_split, data.y_val_true, base_val_pred, base_val_pred, zero_corr_val, zero_corr_val, validation_error_dist)
            eval_pred_df = self._build_prediction_df(data.eval_df_processed, data.y_eval_true, base_eval_pred, base_eval_pred, zero_corr_eval, zero_corr_eval, eval_error_dist)
            save_predictions_csv(val_pred_df, trial_dir / "predictions_validation.csv")
            save_predictions_csv(eval_pred_df, trial_dir / "predictions_evaluation.csv")
            self._save_prediction_artifacts(trial_dir, "validation_base", data.y_val_true, base_val_pred, validation_error_dist)
            self._save_prediction_artifacts(trial_dir, "evaluation_base", data.y_eval_true, base_eval_pred, eval_error_dist)
            transformer_config = {
                "input_dim": int(data.X_train.shape[1]),
                "model_dim": int(params["model_dim"]),
                "num_heads": int(params["num_heads"]),
                "num_layers": int(params["num_layers"]),
                "dropout": float(params["dropout"]),
                "ff_multiplier": int(params["ff_multiplier"]),
            }
            transformer_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({
                "model_state_dict": transformer_state_dict,
                "transformer_config": transformer_config,
                "params": params,
                "preprocess_summary": data.preprocess_summary,
                "calibration_summary": {"calibration_model": "none", "search_stage": "base", "objective_metric": "validation_base_distance_mean"},
            }, trial_dir / "transformer_best.pt")
            save_model_bundle({
                "seed": self.config.seed,
                "pipeline_name": "data_preprocess -> transformer_autoencoder_encoder -> multioutput_svr",
                "params": params,
                "preprocess_summary": data.preprocess_summary,
                "calibration_summary": {"calibration_model": "none", "search_stage": "base", "objective_metric": "validation_base_distance_mean"},
                "transformer_config": transformer_config,
                "transformer_state_dict": transformer_state_dict,
                "svr_model": svr_model,
                "residual_calibrator": None,
                "scaler_X": data.scaler_X,
                "scaler_y": data.scaler_y,
                "feature_mask": data.feature_mask,
                "feature_names": data.feature_names,
                "cached_train_features": X_train_feat.astype(np.float32),
                "cached_val_features": X_val_feat.astype(np.float32),
                "cached_eval_features": X_eval_feat.astype(np.float32),
                "cached_train_reconstruction_error": np.zeros_like(data.X_train, dtype=np.float32),
                "cached_val_reconstruction_error": val_reconstruction_error.astype(np.float32),
                "cached_eval_reconstruction_error": eval_reconstruction_error.astype(np.float32),
                "cached_base_train_pred": np.asarray(svr_model.predict(X_train_feat), dtype=np.float64),
                "cached_base_val_pred": base_val_pred.astype(np.float64),
                "cached_base_eval_pred": base_eval_pred.astype(np.float64),
                "cached_base_val_metrics": validation_metrics,
                "cached_base_eval_metrics": evaluation_metrics,
            }, trial_dir / "inference_bundle.joblib")
            objective_value = float(validation_metrics["distance_mean"])
            metrics = {
                "status": "success",
                "objective_name": "validation_base_distance_mean",
                "objective_value": objective_value,
                "evaluation_target_metric_name": "evaluation_distance_mean",
                "model_num_parameters": model_num_parameters,
                "training": {
                    "best_epoch": int(history.get("best_epoch", 0)),
                    "best_val_loss": float(history.get("best_val_loss", float("inf"))),
                    "runtime_seconds": float(wall_clock() - trial_start),
                    "stop_reason": history.get("stop_reason", "finished"),
                },
                "preprocess": data.preprocess_summary,
                "validation_base": validation_metrics,
                "evaluation_base": evaluation_metrics,
            }
            save_trial_metadata(trial_dir, trial.number, objective_value, params, metrics, "success")
            save_text_report(build_trial_report_text(trial.number, "success", params, data.preprocess_summary, history, validation_metrics, evaluation_metrics, {
                "objective_name": "validation_base_distance_mean",
                "objective_value": objective_value,
                "evaluation_target_metric_name": "evaluation_distance_mean",
                "model_num_parameters": model_num_parameters,
                "saved_model_files": ["transformer_best.pt", "inference_bundle.joblib", "training_history.csv"],
            }), trial_dir / "trial_report.txt")
            trial.set_user_attr("trial_dir", str(trial_dir))
            trial.set_user_attr("status", "success")
            trial.set_user_attr("stage", "base")
            trial.set_user_attr("validation_distance_mean", objective_value)
            trial.set_user_attr("internal_validation_distance_mean", objective_value)
            trial.set_user_attr("evaluation_distance_mean", float(evaluation_metrics["distance_mean"]))
            trial.set_user_attr("feature_keep_ratio", float(data.preprocess_summary["feature_keep_ratio"]))
            trial.set_user_attr("model_num_parameters", model_num_parameters)
            print(f"[BaseTrialSuccess] {status_prefix} objective={objective_value:.8f} runtime={wall_clock() - trial_start:.2f}s saved_to={trial_dir}")
            return objective_value
        except optuna.TrialPruned:
            print(f"[BaseTrialPruned] {status_prefix} pruned. partial artifacts kept in {trial_dir}")
            raise
        except Exception as exc:
            fallback_value = float(1e12 + trial.number)
            error_payload = {
                "status": "failed",
                "objective_name": "validation_base_distance_mean",
                "objective_value": fallback_value,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "runtime_seconds": float(wall_clock() - trial_start),
            }
            save_json(error_payload, trial_dir / "error.json")
            if history is not None:
                save_history_csv(history, history_path)
                save_loss_curve(history, trial_dir / "loss_curve.png", log_scale=False)
                save_loss_curve(history, trial_dir / "loss_curve_logscale.png", log_scale=True)
                save_metric_trace(history, trial_dir / "training_monitor_trace.png")
            save_trial_metadata(trial_dir, trial.number, fallback_value, params, error_payload, "failed")
            self._empty_prediction_csv(trial_dir / "predictions_validation.csv")
            self._empty_prediction_csv(trial_dir / "predictions_evaluation.csv")
            trial.set_user_attr("trial_dir", str(trial_dir))
            trial.set_user_attr("status", "failed")
            trial.set_user_attr("stage", "base")
            trial.set_user_attr("error_type", type(exc).__name__)
            trial.set_user_attr("error_message", str(exc))
            print(f"[BaseTrialFailed] {status_prefix} {type(exc).__name__}: {exc}")
            return fallback_value
        finally:
            del data, model, history
            gc.collect()
            if torch.cuda.is_available() and self.device.type == "cuda":
                torch.cuda.empty_cache()




def _derive_building_floor_route_labels(meta_df: pd.DataFrame) -> np.ndarray | None:
    if {"BUILDINGID", "FLOOR"}.issubset(meta_df.columns):
        return (
            meta_df[["BUILDINGID", "FLOOR"]]
            .fillna(-9999)
            .astype(int)
            .astype(str)
            .agg("_".join, axis=1)
            .to_numpy()
        )
    return None


def _build_base_train_oof_predictions(
    X_train_feat: np.ndarray,
    y_train_true: np.ndarray,
    base_params: Dict[str, Any],
    train_meta_df: pd.DataFrame,
    *,
    random_state: int,
    requested_splits: int = 5,
) -> tuple[np.ndarray, Dict[str, Any], Dict[str, float]]:
    X_train_feat = np.asarray(X_train_feat, dtype=np.float32)
    y_train_true = np.asarray(y_train_true, dtype=np.float64)
    n_samples = int(len(X_train_feat))
    if n_samples == 0:
        raise ValueError("stage2 calibration requires non-empty train features.")
    if n_samples == 1:
        model = fit_svr(X_train_feat, y_train_true, build_svr_params(base_params))
        pred = np.asarray(model.predict(X_train_feat), dtype=np.float64)
        metrics, _ = evaluate_predictions(y_train_true, pred)
        return pred, {
            "strategy": "fallback_single_sample",
            "requested_splits": int(requested_splits),
            "effective_splits": 1,
            "sample_count": 1,
        }, metrics

    groups = _derive_position_group_labels(train_meta_df)
    effective_splits = max(2, min(int(requested_splits), n_samples))
    split_strategy = "kfold"
    indices = np.arange(n_samples)
    if groups is not None and len(np.unique(groups)) >= 2:
        effective_splits = max(2, min(int(effective_splits), int(len(np.unique(groups)))))
        splitter = GroupKFold(n_splits=effective_splits)
        split_iter = splitter.split(indices, groups=groups)
        split_strategy = "group_kfold"
    else:
        splitter = KFold(n_splits=effective_splits, shuffle=True, random_state=int(random_state))
        split_iter = splitter.split(indices)

    svr_params = build_svr_params(base_params)
    oof_pred = np.zeros_like(y_train_true, dtype=np.float64)
    covered = np.zeros(n_samples, dtype=bool)
    fold_summaries: list[dict[str, Any]] = []
    for fold_idx, (fit_idx, pred_idx) in enumerate(split_iter, start=1):
        fold_model = fit_svr(X_train_feat[fit_idx], y_train_true[fit_idx], svr_params)
        fold_pred = np.asarray(fold_model.predict(X_train_feat[pred_idx]), dtype=np.float64)
        oof_pred[pred_idx] = fold_pred
        covered[pred_idx] = True
        fold_metrics, _ = evaluate_predictions(y_train_true[pred_idx], fold_pred)
        fold_summaries.append({
            "fold_index": int(fold_idx),
            "fit_count": int(len(fit_idx)),
            "pred_count": int(len(pred_idx)),
            "distance_mean": float(fold_metrics["distance_mean"]),
        })

    if not np.all(covered):
        raise RuntimeError("base-stage OOF prediction did not cover all stage2 train samples.")

    metrics, _ = evaluate_predictions(y_train_true, oof_pred)
    summary = {
        "strategy": "stage1_train_oof_for_stage2",
        "split_strategy": split_strategy,
        "requested_splits": int(requested_splits),
        "effective_splits": int(effective_splits),
        "sample_count": int(n_samples),
        "distance_mean": float(metrics["distance_mean"]),
        "folds": fold_summaries,
    }
    return oof_pred, summary, metrics


class CatBoostTrialObjective(_TrialArtifactsMixin):
    def __init__(
        self,
        raw_data: RawDataBundle,
        config: ExperimentConfig,
        device: torch.device,
        experiment_root: str | Path,
        base_candidates: Iterable[BaseCandidateRecord],
        *,
        total_trials: int | None = None,
        live_status_root: str | Path | None = None,
    ) -> None:
        self.raw_data = raw_data
        self.config = config
        self.device = device
        self.experiment_root = Path(experiment_root)
        self.base_candidates = list(base_candidates)
        if not self.base_candidates:
            raise ValueError("CatBoost stage requires at least one prepared base candidate.")
        self.base_trial_choices = [int(c.trial_number) for c in self.base_candidates]
        self.base_candidate_map = {int(c.trial_number): c for c in self.base_candidates}
        self.cache: Dict[int, PreparedBaseCandidate] = {}
        self.total_trials = int(config.optuna.catboost_n_trials if total_trials is None else total_trials)
        self.live_status_root = None if live_status_root is None else Path(live_status_root)

    def _load_candidate(self, trial_number: int) -> PreparedBaseCandidate:
        if trial_number in self.cache:
            return self.cache[trial_number]
        record = self.base_candidate_map[trial_number]
        bundle = load_inference_bundle(record.trial_dir / "inference_bundle.joblib")
        base_params = dict(bundle["params"])
        data = preprocess_loaded_data(
            raw_data=self.raw_data,
            feature_missing_threshold=float(base_params["feature_missing_threshold"]),
            building_id=self.config.preprocess.building_id,
            val_size=self.config.preprocess.val_size,
            random_state=self.config.seed,
        )

        cached_train_features = bundle.get("cached_train_features")
        cached_val_features = bundle.get("cached_val_features")
        cached_eval_features = bundle.get("cached_eval_features")
        cached_train_reconstruction_error = bundle.get("cached_train_reconstruction_error")
        cached_val_reconstruction_error = bundle.get("cached_val_reconstruction_error")
        cached_eval_reconstruction_error = bundle.get("cached_eval_reconstruction_error")
        cached_base_train_pred = bundle.get("cached_base_train_pred")
        cached_base_train_oof_pred = bundle.get("cached_base_train_oof_pred")
        cached_base_train_oof_summary = bundle.get("cached_base_train_oof_summary")
        cached_base_train_metrics = bundle.get("cached_base_train_metrics")
        cached_base_val_pred = bundle.get("cached_base_val_pred")
        cached_base_eval_pred = bundle.get("cached_base_eval_pred")
        cached_base_val_metrics = bundle.get("cached_base_val_metrics")
        cached_base_eval_metrics = bundle.get("cached_base_eval_metrics")

        if (
            cached_train_features is not None
            and cached_val_features is not None
            and cached_eval_features is not None
            and cached_train_reconstruction_error is not None
            and cached_val_reconstruction_error is not None
            and cached_eval_reconstruction_error is not None
            and cached_base_val_pred is not None
            and cached_base_eval_pred is not None
        ):
            X_train_feat = np.asarray(cached_train_features, dtype=np.float32)
            X_val_feat = np.asarray(cached_val_features, dtype=np.float32)
            X_eval_feat = np.asarray(cached_eval_features, dtype=np.float32)
            train_reconstruction_error = np.asarray(cached_train_reconstruction_error, dtype=np.float32)
            val_reconstruction_error = np.asarray(cached_val_reconstruction_error, dtype=np.float32)
            eval_reconstruction_error = np.asarray(cached_eval_reconstruction_error, dtype=np.float32)
            base_val_pred = np.asarray(cached_base_val_pred, dtype=np.float64)
            base_eval_pred = np.asarray(cached_base_eval_pred, dtype=np.float64)
            base_train_pred_in_sample = None if cached_base_train_pred is None else np.asarray(cached_base_train_pred, dtype=np.float64)
            if cached_base_val_metrics is None or cached_base_eval_metrics is None:
                base_val_metrics, _ = evaluate_predictions(data.y_val_true, base_val_pred)
                base_eval_metrics, _ = evaluate_predictions(data.y_eval_true, base_eval_pred)
            else:
                base_val_metrics = dict(cached_base_val_metrics)
                base_eval_metrics = dict(cached_base_eval_metrics)
        else:
            model = WiFiTransformerAutoencoder(**bundle["transformer_config"])
            model.load_state_dict(bundle["transformer_state_dict"])
            model.to(self.device)
            model.eval()
            try:
                train_payload = extract_feature_payload(
                    model,
                    data.X_train,
                    self.device,
                    self.config.training.feature_batch_size,
                    f"stage2 prepare base={trial_number + 1:04d} train_feature_extraction",
                    self.config.training.num_workers,
                    include_reconstruction=True,
                )
                val_payload = extract_feature_payload(
                    model,
                    data.X_val,
                    self.device,
                    self.config.training.feature_batch_size,
                    f"stage2 prepare base={trial_number + 1:04d} val_feature_extraction",
                    self.config.training.num_workers,
                    include_reconstruction=True,
                )
                eval_payload = extract_feature_payload(
                    model,
                    data.X_eval,
                    self.device,
                    self.config.training.feature_batch_size,
                    f"stage2 prepare base={trial_number + 1:04d} eval_feature_extraction",
                    self.config.training.num_workers,
                    include_reconstruction=True,
                )
                X_train_feat = train_payload["latent_features"]
                X_val_feat = val_payload["latent_features"]
                X_eval_feat = eval_payload["latent_features"]
                train_reconstruction_error = data.X_train.astype(np.float32) - train_payload["reconstruction"].astype(np.float32)
                val_reconstruction_error = data.X_val.astype(np.float32) - val_payload["reconstruction"].astype(np.float32)
                eval_reconstruction_error = data.X_eval.astype(np.float32) - eval_payload["reconstruction"].astype(np.float32)
                base_train_pred_in_sample = np.asarray(bundle["svr_model"].predict(X_train_feat), dtype=np.float64)
                base_val_pred = np.asarray(bundle["svr_model"].predict(X_val_feat), dtype=np.float64)
                base_eval_pred = np.asarray(bundle["svr_model"].predict(X_eval_feat), dtype=np.float64)
                base_val_metrics, _ = evaluate_predictions(data.y_val_true, base_val_pred)
                base_eval_metrics, _ = evaluate_predictions(data.y_eval_true, base_eval_pred)
            finally:
                del model
                gc.collect()
                if torch.cuda.is_available() and self.device.type == "cuda":
                    torch.cuda.empty_cache()

        if cached_base_train_oof_pred is not None:
            base_train_pred = np.asarray(cached_base_train_oof_pred, dtype=np.float64)
            if cached_base_train_metrics is None:
                base_train_metrics, _ = evaluate_predictions(data.y_train_true, base_train_pred)
            else:
                base_train_metrics = dict(cached_base_train_metrics)
            base_train_oof_summary = dict(cached_base_train_oof_summary or {
                "strategy": "loaded_cached_stage1_train_oof_for_stage2",
                "sample_count": int(len(base_train_pred)),
            })
        else:
            base_train_pred, base_train_oof_summary, base_train_metrics = _build_base_train_oof_predictions(
                X_train_feat,
                data.y_train_true,
                base_params,
                data.train_df_split,
                random_state=self.config.seed,
                requested_splits=5,
            )
            if cached_base_train_pred is not None:
                base_train_oof_summary = {
                    **base_train_oof_summary,
                    "cached_train_in_sample_distance_mean": float(evaluate_predictions(data.y_train_true, np.asarray(cached_base_train_pred, dtype=np.float64))[0]["distance_mean"]),
                }
            elif 'base_train_pred_in_sample' in locals() and base_train_pred_in_sample is not None:
                base_train_oof_summary = {
                    **base_train_oof_summary,
                    "recomputed_train_in_sample_distance_mean": float(evaluate_predictions(data.y_train_true, base_train_pred_in_sample)[0]["distance_mean"]),
                }

        prepared = PreparedBaseCandidate(
            record=record,
            data=data,
            bundle=bundle,
            X_train_feat=X_train_feat,
            X_val_feat=X_val_feat,
            X_eval_feat=X_eval_feat,
            train_reconstruction_error=train_reconstruction_error,
            val_reconstruction_error=val_reconstruction_error,
            eval_reconstruction_error=eval_reconstruction_error,
            base_train_pred=base_train_pred,
            base_val_pred=base_val_pred,
            base_eval_pred=base_eval_pred,
            base_train_metrics=base_train_metrics,
            base_val_metrics=base_val_metrics,
            base_eval_metrics=base_eval_metrics,
            base_train_oof_summary=base_train_oof_summary,
        )
        self.cache[trial_number] = prepared
        return prepared

    @staticmethod
    def _catboost_model_params(params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "iterations": int(params["catboost_iterations"]),
            "depth": int(params["catboost_depth"]),
            "learning_rate": float(params["catboost_learning_rate"]),
            "l2_leaf_reg": float(params["catboost_l2_leaf_reg"]),
            "random_strength": float(params["catboost_random_strength"]),
            "bagging_temperature": float(params["catboost_bagging_temperature"]),
            "border_count": int(params["catboost_border_count"]),
            "min_data_in_leaf": int(params["catboost_min_data_in_leaf"]),
            "grow_policy": str(params["catboost_grow_policy"]),
            "leaf_estimation_iterations": int(params["catboost_leaf_estimation_iterations"]),
        }

    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        selected_base_trial = int(trial.suggest_categorical("base_trial_number", self.base_trial_choices))
        params: Dict[str, Any] = {
            "search_stage": "catboost",
            "base_trial_number": selected_base_trial,
            "base_trial_folder": f"{selected_base_trial + 1:04d}",
            "calibration_model": "catboost_residual_regressor",
            "calibration_oof_folds": CALIBRATION_OOF_FOLDS,
            "stage2_feature_mode": trial.suggest_categorical("stage2_feature_mode", ["paper_residual_pred", "raw_rssi_pred", "raw_rssi_latent_pred", "full_stage2"]),
            "stage2_target_mode": trial.suggest_categorical("stage2_target_mode", ["pointwise", "group_mean", "group_median"]),
            "stage2_route_mode": trial.suggest_categorical("stage2_route_mode", ["region_cluster", "pred_cluster", "building_floor_region"]),
            "stage2_route_cluster_count": trial.suggest_categorical("stage2_route_cluster_count", [4, 6, 8, 12, 16]),
            "stage2_route_min_samples": trial.suggest_categorical("stage2_route_min_samples", [48, 64, 96, 128, 160]),
            "stage2_neighbor_k": trial.suggest_categorical("stage2_neighbor_k", [4, 8, 12, 16]),
            "stage2_blend_alpha": trial.suggest_float("stage2_blend_alpha", 0.20, 0.95),
            "stage2_clip_quantile": trial.suggest_categorical("stage2_clip_quantile", [None, 0.995, 0.999]),
            "stage2_pca_components": trial.suggest_categorical("stage2_pca_components", [None, 16, 24, 32, 48, 64]),
            "catboost_iterations": trial.suggest_categorical("catboost_iterations", [300, 500, 700, 900, 1200]),
            "catboost_depth": trial.suggest_int("catboost_depth", 4, 10),
            "catboost_learning_rate": trial.suggest_float("catboost_learning_rate", 1e-2, 2e-1, log=True),
            "catboost_l2_leaf_reg": trial.suggest_float("catboost_l2_leaf_reg", 1.0, 20.0, log=True),
            "catboost_random_strength": trial.suggest_float("catboost_random_strength", 1e-3, 10.0, log=True),
            "catboost_bagging_temperature": trial.suggest_float("catboost_bagging_temperature", 0.0, 5.0),
            "catboost_border_count": trial.suggest_categorical("catboost_border_count", [64, 128, 254]),
            "catboost_min_data_in_leaf": trial.suggest_categorical("catboost_min_data_in_leaf", [1, 4, 8, 16, 32, 64]),
            "catboost_grow_policy": trial.suggest_categorical("catboost_grow_policy", ["SymmetricTree", "Depthwise"]),
            "catboost_leaf_estimation_iterations": trial.suggest_categorical("catboost_leaf_estimation_iterations", [1, 3, 5, 10]),
        }
        return params

    def __call__(self, trial: optuna.Trial) -> float:
        trial_start = wall_clock()
        trial_dir = get_trial_dir(self.experiment_root, trial.number)
        params = self.suggest_params(trial)
        status_prefix = f"catboost trial={trial.number + 1:04d}/{self.total_trials:04d}"
        if self.live_status_root is not None:
            update_live_status(self.live_status_root, {
                "phase": "stage2_running",
                "running_trial_zero_based": int(trial.number),
                "running_trial_folder": f"{trial.number + 1:04d}",
                "running_trial_stage": "catboost",
                "running_trial_status": "running",
            })
            append_live_event(self.live_status_root, {
                "event": "catboost_trial_started",
                "trial_number_zero_based": int(trial.number),
                "trial_folder": f"{trial.number + 1:04d}",
            })
        try:
            selected_trial = int(params["base_trial_number"])
            prepared = self._load_candidate(selected_trial)
            base_bundle = prepared.bundle
            base_params = dict(base_bundle["params"])
            print("\n" + "#" * 100)
            print(f"[CatBoostTrialStart] {status_prefix}")
            print(f"[CatBoostTrialStart] folder={trial_dir}")
            print(f"[CatBoostTrialStart] params={params}")
            print(f"[CatBoostTrialStart] selected_base={selected_trial + 1:04d} objective={prepared.record.objective_value:.8f}")

            train_group_labels = _derive_position_group_labels(prepared.data.train_df_split)
            val_group_labels = _derive_position_group_labels(prepared.data.val_df_split)
            train_route_labels = _derive_building_floor_route_labels(prepared.data.train_df_split)
            val_route_labels = _derive_building_floor_route_labels(prepared.data.val_df_split)

            catboost_model_params = self._catboost_model_params(params)
            residual_calibrator_train = fit_residual_calibrator(
                latent_features=prepared.X_train_feat,
                base_predictions=prepared.base_train_pred,
                y_true=prepared.data.y_train_true,
                reconstruction_error=prepared.train_reconstruction_error,
                signal_values=prepared.data.X_train,
                feature_mode=str(params["stage2_feature_mode"]),
                random_state=self.config.seed,
                max_train_samples=None,
                pca_components=None if params["stage2_pca_components"] is None else int(params["stage2_pca_components"]),
                correction_clip_quantile=params["stage2_clip_quantile"],
                group_labels=train_group_labels,
                target_mode=str(params["stage2_target_mode"]),
                model_params=catboost_model_params,
                route_mode=str(params["stage2_route_mode"]),
                route_labels=train_route_labels,
                route_cluster_count=int(params["stage2_route_cluster_count"]),
                route_min_samples=int(params["stage2_route_min_samples"]),
                neighbor_k=int(params["stage2_neighbor_k"]),
                blend_alpha=float(params["stage2_blend_alpha"]),
            )
            val_eval_pred, correction_val_oof, correction_std_val_oof = apply_residual_calibration(
                residual_calibrator_train,
                latent_features=prepared.X_val_feat,
                base_predictions=prepared.base_val_pred,
                reconstruction_error=prepared.val_reconstruction_error,
                signal_values=prepared.data.X_val,
            )

            full_train_latent = np.vstack([prepared.X_train_feat, prepared.X_val_feat])
            full_train_base_pred = np.vstack([prepared.base_train_pred, prepared.base_val_pred])
            full_train_y_true = np.vstack([prepared.data.y_train_true, prepared.data.y_val_true])
            full_train_reconstruction_error = np.vstack([prepared.train_reconstruction_error, prepared.val_reconstruction_error])
            full_train_signal_values = np.vstack([prepared.data.X_train, prepared.data.X_val])
            full_train_group_labels = None if train_group_labels is None and val_group_labels is None else np.concatenate([
                np.asarray(train_group_labels if train_group_labels is not None else np.array([f"train_{i}" for i in range(len(prepared.data.train_df_split))], dtype=object), dtype=object),
                np.asarray(val_group_labels if val_group_labels is not None else np.array([f"val_{i}" for i in range(len(prepared.data.val_df_split))], dtype=object), dtype=object),
            ])
            full_train_route_labels = None if train_route_labels is None and val_route_labels is None else np.concatenate([
                np.asarray(train_route_labels if train_route_labels is not None else np.array(["train_unknown"] * len(prepared.data.train_df_split), dtype=object), dtype=object),
                np.asarray(val_route_labels if val_route_labels is not None else np.array(["val_unknown"] * len(prepared.data.val_df_split), dtype=object), dtype=object),
            ])
            residual_calibrator_full = fit_residual_calibrator(
                latent_features=full_train_latent,
                base_predictions=full_train_base_pred,
                y_true=full_train_y_true,
                reconstruction_error=full_train_reconstruction_error,
                signal_values=full_train_signal_values,
                feature_mode=str(params["stage2_feature_mode"]),
                random_state=self.config.seed,
                max_train_samples=None,
                pca_components=None if params["stage2_pca_components"] is None else int(params["stage2_pca_components"]),
                correction_clip_quantile=params["stage2_clip_quantile"],
                group_labels=full_train_group_labels,
                target_mode=str(params["stage2_target_mode"]),
                model_params=catboost_model_params,
                route_mode=str(params["stage2_route_mode"]),
                route_labels=full_train_route_labels,
                route_cluster_count=int(params["stage2_route_cluster_count"]),
                route_min_samples=int(params["stage2_route_min_samples"]),
                neighbor_k=int(params["stage2_neighbor_k"]),
                blend_alpha=float(params["stage2_blend_alpha"]),
            )
            full_val_pred, _, _ = apply_residual_calibration(
                residual_calibrator_full,
                latent_features=prepared.X_val_feat,
                base_predictions=prepared.base_val_pred,
                reconstruction_error=prepared.val_reconstruction_error,
                signal_values=prepared.data.X_val,
            )
            eval_pred, correction_eval, correction_std_eval = apply_residual_calibration(
                residual_calibrator_full,
                latent_features=prepared.X_eval_feat,
                base_predictions=prepared.base_eval_pred,
                reconstruction_error=prepared.eval_reconstruction_error,
                signal_values=prepared.data.X_eval,
            )
            validation_metrics, validation_error_dist = evaluate_predictions(prepared.data.y_val_true, val_eval_pred)
            corrected_val_full_metrics, corrected_val_full_error = evaluate_predictions(prepared.data.y_val_true, full_val_pred)
            evaluation_metrics, eval_error_dist = evaluate_predictions(prepared.data.y_eval_true, eval_pred)
            correction_eval = np.asarray(correction_eval, dtype=np.float64)
            correction_std_eval = np.asarray(correction_std_eval, dtype=np.float64)
            objective_value = float(validation_metrics["distance_mean"])
            oof_summary = {
                "strategy": "train_holdout_validation_for_stage2",
                "train_sample_count": int(len(prepared.data.y_train_true)),
                "validation_sample_count": int(len(prepared.data.y_val_true)),
                "stage1_train_base_source": prepared.base_train_oof_summary,
                "train_fit_metrics": residual_calibrator_train.train_metrics,
                "holdout_validation_distance_mean": float(validation_metrics["distance_mean"]),
            }
            calibration_summary: Dict[str, Any] = {
                "calibration_model": "catboost_residual_regressor",
                "search_stage": "catboost",
                "selected_base_trial_number": selected_trial,
                "selected_base_trial_folder": f"{selected_trial + 1:04d}",
                "selected_base_objective": float(prepared.record.objective_value),
                "objective_metric": "validation_holdout_calibration_distance_mean",
                "objective_validation_strategy": "train_split_fit -> validation_split_holdout",
                "feature_mode": params["stage2_feature_mode"],
                "target_mode": params["stage2_target_mode"],
                "route_mode": params["stage2_route_mode"],
                "route_cluster_count": params["stage2_route_cluster_count"],
                "route_min_samples": params["stage2_route_min_samples"],
                "neighbor_k": params["stage2_neighbor_k"],
                "blend_alpha": params["stage2_blend_alpha"],
                "paper_alignment": {
                    "residual_target": True,
                    "input_contains_rssi": bool("rssi" in str(params["stage2_feature_mode"]) or "paper_residual" in str(params["stage2_feature_mode"])),
                    "input_contains_stage1_prediction": bool("pred" in str(params["stage2_feature_mode"]) or params["stage2_feature_mode"] == "full_stage2"),
                    "local_expert_routing": True,
                    "confidence_or_neighbor_features": True,
                },
                "pca_components": params["stage2_pca_components"],
                "clip_quantile": params["stage2_clip_quantile"],
                "stage1_train_oof_requested_folds": 5,
                "stage2_training_summary": oof_summary,
                "fit_sample_count_full_refit": residual_calibrator_full.fit_sample_count,
                "full_refit_train_metrics": residual_calibrator_full.train_metrics,
                "catboost_model_params": catboost_model_params,
            }
            val_pred_df = self._build_prediction_df(prepared.data.val_df_split, prepared.data.y_val_true, prepared.base_val_pred, val_eval_pred, correction_val_oof, correction_std_val_oof, validation_error_dist)
            eval_pred_df = self._build_prediction_df(prepared.data.eval_df_processed, prepared.data.y_eval_true, prepared.base_eval_pred, eval_pred, correction_eval, correction_std_eval, eval_error_dist)
            save_predictions_csv(val_pred_df, trial_dir / "predictions_validation.csv")
            save_predictions_csv(eval_pred_df, trial_dir / "predictions_evaluation.csv")
            self._save_prediction_artifacts(trial_dir, "validation_corrected_holdout", prepared.data.y_val_true, val_eval_pred, validation_error_dist)
            self._save_prediction_artifacts(trial_dir, "validation_corrected_full", prepared.data.y_val_true, full_val_pred, corrected_val_full_error)
            self._save_prediction_artifacts(trial_dir, "evaluation_corrected", prepared.data.y_eval_true, eval_pred, eval_error_dist)
            combined_params = {
                "search_strategy": "two_stage",
                "base_params": base_params,
                "catboost_params": params,
            }
            save_model_bundle({
                "seed": self.config.seed,
                "pipeline_name": "data_preprocess -> transformer_autoencoder_encoder -> multioutput_svr -> local_catboost_residual_calibrator",
                "params": combined_params,
                "preprocess_summary": prepared.data.preprocess_summary,
                "calibration_summary": calibration_summary,
                "transformer_config": base_bundle["transformer_config"],
                "transformer_state_dict": base_bundle["transformer_state_dict"],
                "svr_model": base_bundle["svr_model"],
                "residual_calibrator": residual_calibrator_full,
                "scaler_X": prepared.data.scaler_X,
                "scaler_y": prepared.data.scaler_y,
                "feature_mask": prepared.data.feature_mask,
                "feature_names": prepared.data.feature_names,
            }, trial_dir / "inference_bundle.joblib")
            metrics = {
                "status": "success",
                "objective_name": "validation_holdout_calibration_distance_mean",
                "objective_value": objective_value,
                "evaluation_target_metric_name": "evaluation_distance_mean",
                "selected_base_trial_number": selected_trial,
                "selected_base_trial_folder": f"{selected_trial + 1:04d}",
                "selected_base_train_oof_distance_mean": float(prepared.base_train_metrics["distance_mean"]),
                "selected_base_validation_distance_mean": float(prepared.base_val_metrics["distance_mean"]),
                "selected_base_evaluation_distance_mean": float(prepared.base_eval_metrics["distance_mean"]),
                "selected_base_train_oof_summary": prepared.base_train_oof_summary,
                "preprocess": prepared.data.preprocess_summary,
                "calibration": calibration_summary,
                "train_base_oof": prepared.base_train_metrics,
                "validation_base_full": prepared.base_val_metrics,
                "validation_objective": validation_metrics,
                "validation_corrected_full": corrected_val_full_metrics,
                "evaluation_base": prepared.base_eval_metrics,
                "evaluation": evaluation_metrics,
            }
            save_trial_metadata(trial_dir, trial.number, objective_value, combined_params, metrics, "success")
            report_text = build_trial_report_text(
                trial.number,
                "success",
                combined_params,
                prepared.data.preprocess_summary,
                None,
                validation_metrics,
                evaluation_metrics,
                {
                    "objective_name": "validation_holdout_calibration_distance_mean",
                    "objective_value": objective_value,
                    "evaluation_target_metric_name": "evaluation_distance_mean",
                    "selected_base_trial_folder": f"{selected_trial + 1:04d}",
                    "base_train_oof_distance_mean": float(prepared.base_train_metrics["distance_mean"]),
                    "base_internal_validation_distance_mean": float(prepared.base_val_metrics["distance_mean"]),
                    "base_evaluation_distance_mean": float(prepared.base_eval_metrics["distance_mean"]),
                    "corrected_validation_full_distance_mean": float(corrected_val_full_metrics["distance_mean"]),
                    "corrected_evaluation_distance_mean": float(evaluation_metrics["distance_mean"]),
                },
            )
            save_text_report(report_text, trial_dir / "trial_report.txt")
            trial.set_user_attr("trial_dir", str(trial_dir))
            trial.set_user_attr("status", "success")
            trial.set_user_attr("stage", "catboost")
            trial.set_user_attr("selected_base_trial_number", selected_trial)
            trial.set_user_attr("validation_distance_mean", objective_value)
            trial.set_user_attr("internal_validation_distance_mean", objective_value)
            trial.set_user_attr("evaluation_distance_mean", float(evaluation_metrics["distance_mean"]))
            trial.set_user_attr("evaluation_distance_mean_base", float(prepared.base_eval_metrics["distance_mean"]))
            trial.set_user_attr("feature_keep_ratio", float(prepared.data.preprocess_summary["feature_keep_ratio"]))
            if self.live_status_root is not None:
                update_live_status(self.live_status_root, {
                    "running_trial_zero_based": int(trial.number),
                    "running_trial_folder": f"{trial.number + 1:04d}",
                    "running_trial_status": "success",
                    "latest_trial_objective_value": float(objective_value),
                    "latest_trial_selected_base_folder": f"{selected_trial + 1:04d}",
                })
                append_live_event(self.live_status_root, {
                    "event": "catboost_trial_succeeded",
                    "trial_number_zero_based": int(trial.number),
                    "trial_folder": f"{trial.number + 1:04d}",
                    "objective_value": float(objective_value),
                    "selected_base_trial_number_zero_based": int(selected_trial),
                    "selected_base_trial_folder": f"{selected_trial + 1:04d}",
                    "runtime_seconds": float(wall_clock() - trial_start),
                })
            print(f"[CatBoostTrialSuccess] {status_prefix} objective={objective_value:.8f} runtime={wall_clock() - trial_start:.2f}s saved_to={trial_dir}")
            return objective_value
        except Exception as exc:
            fallback_value = float(2e12 + trial.number)
            error_payload = {
                "status": "failed",
                "objective_name": "validation_holdout_calibration_distance_mean",
                "objective_value": fallback_value,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "runtime_seconds": float(wall_clock() - trial_start),
            }
            save_json(error_payload, trial_dir / "error.json")
            save_trial_metadata(trial_dir, trial.number, fallback_value, params, error_payload, "failed")
            self._empty_prediction_csv(trial_dir / "predictions_validation.csv")
            self._empty_prediction_csv(trial_dir / "predictions_evaluation.csv")
            trial.set_user_attr("trial_dir", str(trial_dir))
            trial.set_user_attr("status", "failed")
            trial.set_user_attr("stage", "catboost")
            trial.set_user_attr("error_type", type(exc).__name__)
            trial.set_user_attr("error_message", str(exc))
            if self.live_status_root is not None:
                update_live_status(self.live_status_root, {
                    "running_trial_zero_based": int(trial.number),
                    "running_trial_folder": f"{trial.number + 1:04d}",
                    "running_trial_status": "failed",
                    "latest_trial_error_type": type(exc).__name__,
                    "latest_trial_error_message": str(exc),
                })
                append_live_event(self.live_status_root, {
                    "event": "catboost_trial_failed",
                    "trial_number_zero_based": int(trial.number),
                    "trial_folder": f"{trial.number + 1:04d}",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "runtime_seconds": float(wall_clock() - trial_start),
                })
            print(f"[CatBoostTrialFailed] {status_prefix} {type(exc).__name__}: {exc}")
            return fallback_value
        finally:
            gc.collect()
            if torch.cuda.is_available() and self.device.type == "cuda":
                torch.cuda.empty_cache()
