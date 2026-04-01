import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from .utils import append_jsonl, copy_dir, ensure_dir, now_timestamp, read_json, save_json


PIPELINE_SCHEME = {
    "name": "data_preprocess -> transformer_autoencoder_encoder -> multioutput_svr -> local_catboost_residual_calibrator",
    "stage_1": {
        "name": "transformer_autoencoder_encoder + multioutput_svr",
        "output": "base coordinate prediction",
    },
    "stage_2": {
        "name": "local_catboost_residual_calibrator",
        "input": "stage-1 latent features + stage-1 prediction + residual-related context",
        "output": "residual correction over stage-1 prediction",
    },
}


def create_experiment_root(result_root: str) -> Path:
    result_root_path = ensure_dir(result_root)
    experiment_root = result_root_path / now_timestamp()
    return ensure_dir(experiment_root)


def get_trial_dir(experiment_root: str | Path, trial_number_zero_based: int) -> Path:
    experiment_root = Path(experiment_root)
    folder_name = f"{trial_number_zero_based + 1:04d}"
    return ensure_dir(experiment_root / folder_name)


def save_trial_metadata(trial_dir: str | Path, trial_number: int, trial_value: float | None, params: Dict[str, Any], metrics: Dict[str, Any], status: str) -> None:
    trial_dir = Path(trial_dir)
    save_json(PIPELINE_SCHEME, trial_dir / "pipeline_scheme.json")
    save_json(params, trial_dir / "params.json")
    save_json(metrics, trial_dir / "metrics.json")
    save_json({"trial_number_zero_based": trial_number, "trial_folder": trial_dir.name, "objective_value": trial_value, "status": status}, trial_dir / "trial_info.json")


def save_predictions_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_history_csv(history: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epoch_count = len(history.get("epoch", []))
    if epoch_count == 0:
        pd.DataFrame(columns=["epoch", "train_loss", "val_loss"]).to_csv(path, index=False, encoding="utf-8-sig")
        return
    df = pd.DataFrame({
        "epoch": history.get("epoch", []),
        "train_loss": history.get("train_loss", []),
        "val_loss": history.get("val_loss", []),
        "train_observed_loss": history.get("train_observed_loss", []),
        "val_observed_loss": history.get("val_observed_loss", []),
        "train_missing_loss": history.get("train_missing_loss", []),
        "val_missing_loss": history.get("val_missing_loss", []),
        "train_observed_fraction": history.get("train_observed_fraction", []),
        "val_observed_fraction": history.get("val_observed_fraction", []),
        "train_missing_fraction": history.get("train_missing_fraction", []),
        "val_missing_fraction": history.get("val_missing_fraction", []),
        "train_loss_delta": history.get("train_loss_delta", []),
        "val_loss_delta": history.get("val_loss_delta", []),
        "val_improvement_ratio": history.get("val_improvement_ratio", []),
        "best_val_loss_running": history.get("best_val_loss_running", []),
        "patience_counter": history.get("patience_counter", []),
        "learning_rate": history.get("learning_rate", []),
        "epoch_seconds": history.get("epoch_seconds", []),
        "elapsed_seconds": history.get("elapsed_seconds", []),
    })
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_text_report(text: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_model_bundle(bundle: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def build_trial_report_text(
    trial_number: int,
    status: str,
    params: Dict[str, Any],
    preprocess_summary: Dict[str, Any] | None,
    training_history: Dict[str, Any] | None,
    validation_metrics: Dict[str, Any] | None,
    evaluation_metrics: Dict[str, Any] | None,
    extra: Dict[str, Any] | None = None,
) -> str:
    lines: list[str] = []
    lines.append("=" * 110)
    lines.append(f"Trial {trial_number + 1:04d} Report")
    lines.append("=" * 110)
    lines.append(f"Status: {status}")
    lines.append("")
    lines.append("[Pipeline]")
    lines.append(json.dumps(PIPELINE_SCHEME, ensure_ascii=False, indent=2))
    lines.append("")
    lines.append("[Parameters]")
    lines.append(json.dumps(params, ensure_ascii=False, indent=2))
    lines.append("")
    if preprocess_summary is not None:
        lines.append("[Data Cleaning Summary]")
        lines.append(json.dumps(preprocess_summary, ensure_ascii=False, indent=2))
        lines.append("")
    if training_history is not None:
        lines.append("[Training Summary]")
        lines.append(json.dumps({"best_epoch": training_history.get("best_epoch"), "best_val_loss": training_history.get("best_val_loss"), "loss_name": training_history.get("loss_name"), "observed_loss_weight": training_history.get("observed_loss_weight"), "missing_loss_weight": training_history.get("missing_loss_weight"), "missing_value_threshold": training_history.get("missing_value_threshold"), "stop_reason": training_history.get("stop_reason"), "pruned": training_history.get("pruned"), "pruned_epoch": training_history.get("pruned_epoch")}, ensure_ascii=False, indent=2))
        lines.append("")
        lines.append("[Epoch-by-Epoch Trace]")
        epochs = training_history.get("epoch", [])
        for i, epoch in enumerate(epochs):
            train_loss = training_history.get("train_loss", [])[i]
            val_loss = training_history.get("val_loss", [])[i]
            train_delta = training_history.get("train_loss_delta", [])[i]
            val_delta = training_history.get("val_loss_delta", [])[i]
            improve = training_history.get("val_improvement_ratio", [])[i]
            patience = training_history.get("patience_counter", [])[i]
            epoch_seconds = training_history.get("epoch_seconds", [])[i] if i < len(training_history.get("epoch_seconds", [])) else None
            elapsed_seconds = training_history.get("elapsed_seconds", [])[i] if i < len(training_history.get("elapsed_seconds", [])) else None
            train_obs_loss = training_history.get("train_observed_loss", [])[i] if i < len(training_history.get("train_observed_loss", [])) else None
            val_obs_loss = training_history.get("val_observed_loss", [])[i] if i < len(training_history.get("val_observed_loss", [])) else None
            train_missing_loss = training_history.get("train_missing_loss", [])[i] if i < len(training_history.get("train_missing_loss", [])) else None
            val_missing_loss = training_history.get("val_missing_loss", [])[i] if i < len(training_history.get("val_missing_loss", [])) else None
            val_obs_fraction = training_history.get("val_observed_fraction", [])[i] if i < len(training_history.get("val_observed_fraction", [])) else None
            lines.append(f"epoch={epoch:03d} | train_loss={train_loss:.10f} | val_loss={val_loss:.10f} | train_obs_loss={'' if train_obs_loss is None else f'{train_obs_loss:.10f}'} | val_obs_loss={'' if val_obs_loss is None else f'{val_obs_loss:.10f}'} | train_missing_loss={'' if train_missing_loss is None else f'{train_missing_loss:.10f}'} | val_missing_loss={'' if val_missing_loss is None else f'{val_missing_loss:.10f}'} | val_obs_fraction={'' if val_obs_fraction is None else f'{val_obs_fraction:.6f}'} | train_loss_delta={'' if train_delta is None else f'{train_delta:.10f}'} | val_loss_delta={'' if val_delta is None else f'{val_delta:.10f}'} | val_improvement_ratio={'' if improve is None else f'{improve:.10f}'} | patience={patience} | epoch_seconds={'' if epoch_seconds is None else f'{epoch_seconds:.4f}'} | elapsed_seconds={'' if elapsed_seconds is None else f'{elapsed_seconds:.4f}'}")
        lines.append("")
    if validation_metrics is not None:
        lines.append("[Internal Validation Metrics]")
        lines.append(json.dumps(validation_metrics, ensure_ascii=False, indent=2))
        lines.append("")
    if evaluation_metrics is not None:
        lines.append("[Official Evaluation Metrics]")
        lines.append(json.dumps(evaluation_metrics, ensure_ascii=False, indent=2))
        lines.append("")
    if extra:
        lines.append("[Extra Information]")
        lines.append(json.dumps(extra, ensure_ascii=False, indent=2))
        lines.append("")
    return "\n".join(lines)


def save_study_summary(experiment_root: str | Path, summary_df: pd.DataFrame, best_trial_number_zero_based: int | None, best_value: float | None, best_params: Dict[str, Any] | None) -> None:
    experiment_root = Path(experiment_root)
    summary_df.to_csv(experiment_root / "study_summary.csv", index=False, encoding="utf-8-sig")
    save_json({"best_trial_number_zero_based": None if best_trial_number_zero_based is None else int(best_trial_number_zero_based), "best_trial_folder": None if best_trial_number_zero_based is None else f"{best_trial_number_zero_based + 1:04d}", "best_objective_value": None if best_value is None else float(best_value), "best_params": {} if best_params is None else best_params}, experiment_root / "best_trial_summary.json")


def save_study_text_summary(text: str, experiment_root: str | Path) -> None:
    save_text_report(text, Path(experiment_root) / "study_report.txt")


def copy_best_trial_to_zero(experiment_root: str | Path, best_trial_number_zero_based: int) -> Path:
    experiment_root = Path(experiment_root)
    src = experiment_root / f"{best_trial_number_zero_based + 1:04d}"
    dst = experiment_root / "0000"
    copy_dir(src, dst)
    return dst


def save_live_status(experiment_root: str | Path, payload: Dict[str, Any]) -> None:
    save_json(payload, Path(experiment_root) / "live_status.json")


def update_live_status(experiment_root: str | Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    path = Path(experiment_root) / "live_status.json"
    current: Dict[str, Any] = {}
    if path.exists():
        try:
            current = read_json(path)
        except Exception:
            current = {}
    merged = dict(current)
    merged.update(payload)
    save_json(merged, path)
    return merged


def append_live_event(experiment_root: str | Path, payload: Dict[str, Any]) -> None:
    event_payload = {
        "event_time": datetime.now().isoformat(timespec="seconds"),
        **payload,
    }
    append_jsonl(event_payload, Path(experiment_root) / "runtime_events.jsonl")


def build_study_text_report(study_name: str, total_trials: int, success_count: int, pruned_count: int, failed_count: int, best_trial_number_zero_based: int | None, best_value: float | None, best_trial_copy_dir: str | None, best_params: Dict[str, Any] | None, elapsed_readable: str) -> str:
    lines = [
        "=" * 110,
        "Optuna Study Report",
        "=" * 110,
        f"study_name: {study_name}",
        f"total_trials: {total_trials}",
        f"success_trials: {success_count}",
        f"pruned_trials: {pruned_count}",
        f"failed_trials: {failed_count}",
        f"best_trial_folder: {None if best_trial_number_zero_based is None else f'{best_trial_number_zero_based + 1:04d}'}",
        f"best_objective_value: {best_value}",
        f"best_trial_copy_folder: {best_trial_copy_dir}",
        f"elapsed: {elapsed_readable}",
        "",
        "[Best Parameters]",
        json.dumps({} if best_params is None else best_params, ensure_ascii=False, indent=2),
        "",
    ]
    return "\n".join(lines)
