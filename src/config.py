from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from .utils import read_json


@dataclass
class PathConfig:
    train_csv: str
    eval_csv: str
    result_root: str


@dataclass
class PreprocessConfig:
    building_id: int
    val_size: float


@dataclass
class OptunaConfig:
    base_n_trials: int
    catboost_n_trials: int
    top_base_candidates: int
    direction: str
    sampler_seed: int
    study_name_prefix: str
    enable_pruning: bool
    pruner_n_startup_trials: int
    pruner_n_warmup_steps: int
    pruner_interval_steps: int


@dataclass
class TrainingConfig:
    max_epochs: int
    feature_batch_size: int
    train_log_interval: int
    num_workers: int


@dataclass
class DeviceConfig:
    require_cuda: bool
    device_index: int


@dataclass
class ReproducibilityConfig:
    deterministic: bool
    strict_deterministic_algorithms: bool
    deterministic_warn_only: bool
    cublas_workspace_config: str


@dataclass
class ExperimentConfig:
    seed: int
    paths: PathConfig
    preprocess: PreprocessConfig
    optuna: OptunaConfig
    training: TrainingConfig
    device: DeviceConfig
    reproducibility: ReproducibilityConfig
    raw: Dict[str, Any]


DEFAULT_REPRODUCIBILITY = {
    "deterministic": True,
    "strict_deterministic_algorithms": False,
    "deterministic_warn_only": True,
    "cublas_workspace_config": ":4096:8",
}

DEFAULT_OPTUNA = {
    "base_n_trials": 300,
    "catboost_n_trials": 150,
    "top_base_candidates": 1,
    "direction": "minimize",
    "sampler_seed": 42,
    "study_name_prefix": "wifi_transformer_svr_staged",
    "enable_pruning": True,
    "pruner_n_startup_trials": 20,
    "pruner_n_warmup_steps": 10,
    "pruner_interval_steps": 1,
}

DEFAULT_TRAINING = {
    "max_epochs": 150,
    "feature_batch_size": 256,
    "train_log_interval": 100,
    "num_workers": 0,
}

DEFAULT_DEVICE = {
    "require_cuda": True,
    "device_index": 0,
}

DEFAULT_PATHS = {
    "result_root": "result",
}


def _merge_defaults(raw: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(defaults)
    merged.update(raw)
    return merged


def load_config(config_path: str | Path) -> ExperimentConfig:
    config_dict = read_json(config_path)

    paths_dict = _merge_defaults(dict(config_dict.get("paths", {})), DEFAULT_PATHS)
    if "eval_csv" not in paths_dict:
        legacy_test_csv = paths_dict.get("test_csv")
        if legacy_test_csv is None:
            raise KeyError("config.paths 必须提供 eval_csv；为兼容旧版本，也接受 test_csv。")
        paths_dict["eval_csv"] = legacy_test_csv
    if "train_csv" not in paths_dict:
        raise KeyError("config.paths 必须提供 train_csv。")

    if "preprocess" not in config_dict:
        raise KeyError("config.preprocess 必须提供 building_id 与 val_size。")
    preprocess_dict = dict(config_dict["preprocess"])

    optuna_dict = _merge_defaults(dict(config_dict.get("optuna", {})), DEFAULT_OPTUNA)
    legacy_n_trials = config_dict.get("optuna", {}).get("n_trials") if isinstance(config_dict.get("optuna", {}), dict) else None
    if legacy_n_trials is not None and "base_n_trials" not in config_dict.get("optuna", {}):
        optuna_dict["base_n_trials"] = int(legacy_n_trials)
    if legacy_n_trials is not None and "catboost_n_trials" not in config_dict.get("optuna", {}):
        optuna_dict["catboost_n_trials"] = 150

    training_dict = _merge_defaults(dict(config_dict.get("training", {})), DEFAULT_TRAINING)
    device_dict = _merge_defaults(dict(config_dict.get("device", {})), DEFAULT_DEVICE)
    reproducibility_dict = _merge_defaults(dict(config_dict.get("reproducibility", {})), DEFAULT_REPRODUCIBILITY)

    seed = int(config_dict.get("seed", optuna_dict["sampler_seed"]))
    if "sampler_seed" not in config_dict.get("optuna", {}):
        optuna_dict["sampler_seed"] = seed

    return ExperimentConfig(
        seed=seed,
        paths=PathConfig(
            train_csv=str(paths_dict["train_csv"]),
            eval_csv=str(paths_dict["eval_csv"]),
            result_root=str(paths_dict["result_root"]),
        ),
        preprocess=PreprocessConfig(
            building_id=int(preprocess_dict["building_id"]),
            val_size=float(preprocess_dict["val_size"]),
        ),
        optuna=OptunaConfig(**{
            "base_n_trials": int(optuna_dict["base_n_trials"]),
            "catboost_n_trials": int(optuna_dict["catboost_n_trials"]),
            "top_base_candidates": int(optuna_dict["top_base_candidates"]),
            "direction": str(optuna_dict["direction"]),
            "sampler_seed": int(optuna_dict["sampler_seed"]),
            "study_name_prefix": str(optuna_dict["study_name_prefix"]),
            "enable_pruning": bool(optuna_dict["enable_pruning"]),
            "pruner_n_startup_trials": int(optuna_dict["pruner_n_startup_trials"]),
            "pruner_n_warmup_steps": int(optuna_dict["pruner_n_warmup_steps"]),
            "pruner_interval_steps": int(optuna_dict["pruner_interval_steps"]),
        }),
        training=TrainingConfig(**{
            "max_epochs": int(training_dict["max_epochs"]),
            "feature_batch_size": int(training_dict["feature_batch_size"]),
            "train_log_interval": int(training_dict["train_log_interval"]),
            "num_workers": int(training_dict["num_workers"]),
        }),
        device=DeviceConfig(**{
            "require_cuda": bool(device_dict["require_cuda"]),
            "device_index": int(device_dict["device_index"]),
        }),
        reproducibility=ReproducibilityConfig(**{
            "deterministic": bool(reproducibility_dict["deterministic"]),
            "strict_deterministic_algorithms": bool(reproducibility_dict["strict_deterministic_algorithms"]),
            "deterministic_warn_only": bool(reproducibility_dict["deterministic_warn_only"]),
            "cublas_workspace_config": str(reproducibility_dict["cublas_workspace_config"]),
        }),
        raw=config_dict,
    )
