from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from .config import ExperimentConfig
from .data import load_raw_data
from .objective import BaseCandidateRecord, BaseTrialObjective, CatBoostTrialObjective
from .optuna_compat import FrozenTrial, MedianPruner, Study, TPESampler, optuna
from .storage import (
    append_live_event,
    build_study_text_report,
    copy_best_trial_to_zero,
    create_experiment_root,
    save_live_status,
    save_study_summary,
    save_study_text_summary,
    update_live_status,
)
from .utils import copy_dir, ensure_dir, format_seconds, get_device, print_section, read_json, save_json, set_seed, wall_clock
from .visualize import save_optimization_curve, save_trial_state_bar, save_trial_value_scatter


class StudyProgressCallback:
    def __init__(self, experiment_root: Path, study_name: str, total_trials: int, start_time: float) -> None:
        self.experiment_root = experiment_root
        self.study_name = study_name
        self.total_trials = total_trials
        self.start_time = start_time
        self.last_best_trial_number: int | None = None

    @staticmethod
    def _build_summary_df(study: Study) -> pd.DataFrame:
        rows = []
        for trial in study.trials:
            row = {
                "trial_number_zero_based": trial.number,
                "trial_folder": f"{trial.number + 1:04d}",
                "value": float(trial.value) if trial.value is not None else None,
                "state": str(trial.state),
                "status": trial.user_attrs.get("status", "unknown"),
                "stage": trial.user_attrs.get("stage", "unknown"),
                "trial_dir": trial.user_attrs.get("trial_dir", ""),
                "internal_validation_distance_mean": trial.user_attrs.get("internal_validation_distance_mean", trial.user_attrs.get("validation_distance_mean", None)),
                "evaluation_distance_mean": trial.user_attrs.get("evaluation_distance_mean", None),
                "selected_base_trial_number": trial.user_attrs.get("selected_base_trial_number", None),
                "feature_keep_ratio": trial.user_attrs.get("feature_keep_ratio", None),
                "model_num_parameters": trial.user_attrs.get("model_num_parameters", None),
                "error_type": trial.user_attrs.get("error_type", ""),
                "error_message": trial.user_attrs.get("error_message", ""),
            }
            row.update({f"param__{k}": v for k, v in trial.params.items()})
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _get_best_success_trial(study: Study) -> FrozenTrial | None:
        success_trials = [t for t in study.trials if t.user_attrs.get("status") == "success" and t.value is not None]
        if not success_trials:
            return None
        return min(success_trials, key=lambda t: float(t.value))

    def __call__(self, study: Study, trial: FrozenTrial) -> None:
        summary_df = self._build_summary_df(study)
        best_success_trial = self._get_best_success_trial(study)
        best_copy_dir = None
        if best_success_trial is not None:
            save_study_summary(self.experiment_root, summary_df, best_success_trial.number, float(best_success_trial.value), best_success_trial.params)
            if self.last_best_trial_number != best_success_trial.number:
                best_copy_dir = copy_best_trial_to_zero(self.experiment_root, best_success_trial.number)
                self.last_best_trial_number = best_success_trial.number
            else:
                best_copy_dir = self.experiment_root / "0000"
        else:
            save_study_summary(self.experiment_root, summary_df, None, None, None)
        success_values = [float(t.value) for t in study.trials if t.user_attrs.get("status") == "success" and t.value is not None]
        if success_values:
            save_optimization_curve(success_values, self.experiment_root / "optimization_history.png")
            save_trial_value_scatter(success_values, self.experiment_root / "trial_value_scatter.png")
        state_counts: Dict[str, int] = {}
        for t in study.trials:
            state_name = str(t.state).split(".")[-1]
            state_counts[state_name] = state_counts.get(state_name, 0) + 1
        save_trial_state_bar(state_counts, self.experiment_root / "trial_state_distribution.png")
        success_count = int((summary_df["status"] == "success").sum()) if not summary_df.empty else 0
        pruned_count = int((summary_df["status"] == "pruned").sum()) if not summary_df.empty else 0
        failed_count = int((summary_df["status"] == "failed").sum()) if not summary_df.empty else 0
        elapsed = wall_clock() - self.start_time
        study_report = build_study_text_report(
            study_name=self.study_name,
            total_trials=self.total_trials,
            success_count=success_count,
            pruned_count=pruned_count,
            failed_count=failed_count,
            best_trial_number_zero_based=None if best_success_trial is None else best_success_trial.number,
            best_value=None if best_success_trial is None else float(best_success_trial.value),
            best_trial_copy_dir=None if best_copy_dir is None else str(best_copy_dir),
            best_params=None if best_success_trial is None else best_success_trial.params,
            elapsed_readable=format_seconds(elapsed),
        )
        save_study_text_summary(study_report, self.experiment_root)
        update_live_status(self.experiment_root, {
            "phase": "stage2_running",
            "latest_completed_trial_zero_based": int(trial.number),
            "latest_completed_trial_folder": f"{trial.number + 1:04d}",
            "latest_trial_state": str(trial.state),
            "success_trials": success_count,
            "pruned_trials": pruned_count,
            "failed_trials": failed_count,
            "completed_trials": len(study.trials),
            "target_trials": self.total_trials,
            "elapsed_seconds": elapsed,
            "elapsed_readable": format_seconds(elapsed),
            "best_success_trial_zero_based": None if best_success_trial is None else int(best_success_trial.number),
            "best_success_trial_folder": None if best_success_trial is None else f"{best_success_trial.number + 1:04d}",
            "best_success_value": None if best_success_trial is None else float(best_success_trial.value),
        })
        append_live_event(self.experiment_root, {
            "event": "study_progress_updated",
            "latest_completed_trial_zero_based": int(trial.number),
            "latest_completed_trial_folder": f"{trial.number + 1:04d}",
            "completed_trials": len(study.trials),
            "target_trials": self.total_trials,
            "best_success_trial_zero_based": None if best_success_trial is None else int(best_success_trial.number),
            "best_success_trial_folder": None if best_success_trial is None else f"{best_success_trial.number + 1:04d}",
            "best_success_value": None if best_success_trial is None else float(best_success_trial.value),
        })
        best_text = "None"
        if best_success_trial is not None:
            best_text = f"{best_success_trial.number + 1:04d} | value={float(best_success_trial.value):.8f}"
        print(f"[StudyProgress] completed={len(study.trials)}/{self.total_trials} | latest_trial={trial.number + 1:04d} | state={trial.state.name} | best={best_text}")


def _get_best_success_trial(study: Study) -> FrozenTrial | None:
    success_trials = [t for t in study.trials if t.user_attrs.get("status") == "success" and t.value is not None]
    if not success_trials:
        return None
    return min(success_trials, key=lambda t: float(t.value))


def _build_base_candidate_records(stage_root: Path, summary_df: pd.DataFrame, top_k: int) -> List[BaseCandidateRecord]:
    if summary_df.empty:
        return []
    success_df = summary_df[summary_df["status"] == "success"].copy()
    if success_df.empty:
        return []
    success_df = success_df.sort_values(by="value", ascending=True).head(max(1, int(top_k)))
    records: List[BaseCandidateRecord] = []
    for _, row in success_df.iterrows():
        trial_number = int(row["trial_number_zero_based"])
        trial_dir = stage_root / f"{trial_number + 1:04d}"
        params_path = trial_dir / "params.json"
        params = {} if not params_path.exists() else read_json(params_path)
        records.append(BaseCandidateRecord(trial_number=trial_number, trial_dir=trial_dir, objective_value=float(row["value"]), params=params))
    return records


def _run_single_stage(
    *,
    stage_root: Path,
    study_name: str,
    n_trials: int,
    config: ExperimentConfig,
    objective,
    enable_pruning: bool,
    start_time: float,
) -> tuple[Study, StudyProgressCallback, pd.DataFrame, FrozenTrial | None]:
    stage_root = ensure_dir(stage_root)
    pruner = MedianPruner(
        n_startup_trials=config.optuna.pruner_n_startup_trials,
        n_warmup_steps=config.optuna.pruner_n_warmup_steps,
        interval_steps=config.optuna.pruner_interval_steps,
    ) if enable_pruning else None
    storage_desc = "in_memory"
    study = optuna.create_study(
        study_name=study_name,
        direction=config.optuna.direction,
        sampler=TPESampler(seed=config.optuna.sampler_seed),
        pruner=pruner,
    )
    callback = StudyProgressCallback(experiment_root=stage_root, study_name=study_name, total_trials=n_trials, start_time=start_time)
    print(f"[Optuna] stage_root={stage_root}")
    print(f"[Optuna] n_trials={n_trials}")
    print(f"[Optuna] direction={config.optuna.direction}")
    print(f"[Optuna] pruning={enable_pruning}")
    print(f"[Optuna] storage={storage_desc}")
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True, show_progress_bar=False, callbacks=[callback])
    summary_df = callback._build_summary_df(study)
    best_trial = _get_best_success_trial(study)
    return study, callback, summary_df, best_trial


def run_experiment(config: ExperimentConfig) -> Path:
    total_start = wall_clock()
    print_section("Program Start")
    experiment_root = create_experiment_root(config.paths.result_root)
    base_root = ensure_dir(experiment_root / "base_search")
    catboost_root = ensure_dir(experiment_root / "catboost_search")
    print(f"[Output] experiment_root={experiment_root}")
    save_json(config.raw, experiment_root / "used_config.json")
    set_seed(
        config.seed,
        deterministic=config.reproducibility.deterministic,
        strict_deterministic_algorithms=config.reproducibility.strict_deterministic_algorithms,
        deterministic_warn_only=config.reproducibility.deterministic_warn_only,
        cublas_workspace_config=config.reproducibility.cublas_workspace_config,
    )
    device = get_device(require_cuda=config.device.require_cuda, device_index=config.device.device_index)
    raw_data = load_raw_data(train_path=config.paths.train_csv, eval_path=config.paths.eval_csv)
    save_json(
        {
            "seed": config.seed,
            "raw_train_shape": list(raw_data.train_df.shape),
            "raw_eval_shape": list(raw_data.eval_df.shape),
            "building_id": int(config.preprocess.building_id),
            "val_size": float(config.preprocess.val_size),
            "dataset_policy": "trainingData -> train/internal_validation ; validationData -> official_evaluation",
            "search_strategy": {
                "type": "two_stage",
                "base_n_trials": int(config.optuna.base_n_trials),
                "catboost_n_trials": int(config.optuna.catboost_n_trials),
                "top_base_candidates": int(config.optuna.top_base_candidates),
            },
        },
        experiment_root / "data_summary.json",
    )

    print_section("Stage 1: Base Model Search (Transformer + SVR)")
    base_objective = BaseTrialObjective(raw_data=raw_data, config=config, device=device, experiment_root=base_root)
    base_study, base_callback, base_summary_df, best_base_trial = _run_single_stage(
        stage_root=base_root,
        study_name=f"{config.optuna.study_name_prefix}_base_{experiment_root.name}",
        n_trials=int(config.optuna.base_n_trials),
        config=config,
        objective=base_objective,
        enable_pruning=bool(config.optuna.enable_pruning),
        start_time=total_start,
    )

    base_candidates = _build_base_candidate_records(base_root, base_summary_df, config.optuna.top_base_candidates)
    if not base_candidates:
        raise RuntimeError("Stage 1 completed but no successful base candidate was found, so stage 2 cannot start.")

    print_section("Stage 2: Local CatBoost Residual Search")
    catboost_objective = CatBoostTrialObjective(raw_data=raw_data, config=config, device=device, experiment_root=catboost_root, base_candidates=base_candidates)
    catboost_study, catboost_callback, catboost_summary_df, best_catboost_trial = _run_single_stage(
        stage_root=catboost_root,
        study_name=f"{config.optuna.study_name_prefix}_catboost_{experiment_root.name}",
        n_trials=int(config.optuna.catboost_n_trials),
        config=config,
        objective=catboost_objective,
        enable_pruning=False,
        start_time=total_start,
    )

    base_best_copy = copy_best_trial_to_zero(base_root, best_base_trial.number) if best_base_trial is not None else None
    catboost_best_copy = copy_best_trial_to_zero(catboost_root, best_catboost_trial.number) if best_catboost_trial is not None else None
    if catboost_best_copy is not None:
        copy_dir(catboost_best_copy, experiment_root / "0000")

    base_success_count = int((base_summary_df["status"] == "success").sum()) if not base_summary_df.empty else 0
    base_pruned_count = int((base_summary_df["status"] == "pruned").sum()) if not base_summary_df.empty else 0
    base_failed_count = int((base_summary_df["status"] == "failed").sum()) if not base_summary_df.empty else 0
    catboost_success_count = int((catboost_summary_df["status"] == "success").sum()) if not catboost_summary_df.empty else 0
    catboost_pruned_count = int((catboost_summary_df["status"] == "pruned").sum()) if not catboost_summary_df.empty else 0
    catboost_failed_count = int((catboost_summary_df["status"] == "failed").sum()) if not catboost_summary_df.empty else 0

    selected_base_trial_number = None if best_catboost_trial is None else best_catboost_trial.user_attrs.get("selected_base_trial_number")
    final_summary = {
        "search_strategy": "two_stage",
        "base_stage": {
            "n_trials": int(config.optuna.base_n_trials),
            "success_trials": base_success_count,
            "pruned_trials": base_pruned_count,
            "failed_trials": base_failed_count,
            "best_trial_number_zero_based": None if best_base_trial is None else int(best_base_trial.number),
            "best_trial_folder": None if best_base_trial is None else f"{best_base_trial.number + 1:04d}",
            "best_trial_copy_folder": None if base_best_copy is None else str(base_best_copy.relative_to(experiment_root)),
            "best_value": None if best_base_trial is None else float(best_base_trial.value),
            "best_params": {} if best_base_trial is None else best_base_trial.params,
        },
        "catboost_stage": {
            "n_trials": int(config.optuna.catboost_n_trials),
            "success_trials": catboost_success_count,
            "pruned_trials": catboost_pruned_count,
            "failed_trials": catboost_failed_count,
            "best_trial_number_zero_based": None if best_catboost_trial is None else int(best_catboost_trial.number),
            "best_trial_folder": None if best_catboost_trial is None else f"{best_catboost_trial.number + 1:04d}",
            "best_trial_copy_folder": None if catboost_best_copy is None else str(catboost_best_copy.relative_to(experiment_root)),
            "best_value": None if best_catboost_trial is None else float(best_catboost_trial.value),
            "best_params": {} if best_catboost_trial is None else best_catboost_trial.params,
            "selected_base_trial_number_zero_based": None if selected_base_trial_number is None else int(selected_base_trial_number),
            "selected_base_trial_folder": None if selected_base_trial_number is None else f"{int(selected_base_trial_number) + 1:04d}",
        },
        "top_base_candidates": [
            {
                "trial_number_zero_based": int(r.trial_number),
                "trial_folder": f"{r.trial_number + 1:04d}",
                "objective_value": float(r.objective_value),
            }
            for r in base_candidates
        ],
        "final_selected_bundle": None if catboost_best_copy is None else "0000/inference_bundle.joblib",
        "total_runtime_seconds": float(wall_clock() - total_start),
        "total_runtime_readable": format_seconds(wall_clock() - total_start),
    }
    save_json(final_summary, experiment_root / "final_summary.json")
    summary_text = [
        "=" * 110,
        "Two-Stage Experiment Summary",
        "=" * 110,
        f"base_stage_trials: {config.optuna.base_n_trials}",
        f"stage2_catboost_trials: {config.optuna.catboost_n_trials}",
        f"top_base_candidates: {config.optuna.top_base_candidates}",
        f"base_best_folder: {None if best_base_trial is None else f'{best_base_trial.number + 1:04d}'}",
        f"base_best_value: {None if best_base_trial is None else float(best_base_trial.value)}",
        f"stage2_best_folder: {None if best_catboost_trial is None else f'{best_catboost_trial.number + 1:04d}'}",
        f"stage2_best_value: {None if best_catboost_trial is None else float(best_catboost_trial.value)}",
        f"selected_base_for_best_catboost: {None if selected_base_trial_number is None else f'{int(selected_base_trial_number) + 1:04d}'}",
        f"elapsed: {format_seconds(wall_clock() - total_start)}",
        "",
        "[Base Best Params]",
        "{}" if best_base_trial is None else str(best_base_trial.params),
        "",
        "[CatBoost Best Params]",
        "{}" if best_catboost_trial is None else str(best_catboost_trial.params),
        "",
    ]
    save_study_text_summary("\n".join(summary_text), experiment_root)
    print_section("Experiment Finished")
    if best_base_trial is not None:
        print(f"[BestBaseTrial] folder={best_base_trial.number + 1:04d} value={float(best_base_trial.value):.8f}")
    if best_catboost_trial is not None:
        print(f"[BestCatBoostTrial] folder={best_catboost_trial.number + 1:04d} value={float(best_catboost_trial.value):.8f}")
        print(f"[BestCatBoostSelectedBase] folder={int(selected_base_trial_number) + 1:04d}")
    print(f"[Done] total_time={format_seconds(wall_clock() - total_start)}")
    return experiment_root


def load_existing_base_candidates(
    base_source_path: str | Path,
    *,
    top_k: int,
    selected_trial_number_zero_based: int | None = None,
) -> tuple[Path, List[BaseCandidateRecord]]:
    """
    Accept any of the following inputs:
    1) a full experiment root containing base_search/
    2) a base_search directory itself
    3) a single base trial directory containing inference_bundle.joblib
    """
    base_source_path = Path(base_source_path)

    if (base_source_path / "base_search").exists():
        stage_root = base_source_path / "base_search"
    elif (base_source_path / "study_summary.csv").exists() and any(p.is_dir() and p.name.isdigit() for p in base_source_path.iterdir()):
        stage_root = base_source_path
    elif (base_source_path / "inference_bundle.joblib").exists():
        stage_root = base_source_path.parent
        params = read_json(base_source_path / "params.json") if (base_source_path / "params.json").exists() else {}
        metrics = read_json(base_source_path / "metrics.json") if (base_source_path / "metrics.json").exists() else {}
        objective_value = float(metrics.get("objective_value", 0.0))
        try:
            trial_number = int(base_source_path.name) - 1
        except ValueError:
            trial_number = 0
        return stage_root, [
            BaseCandidateRecord(
                trial_number=trial_number,
                trial_dir=base_source_path,
                objective_value=objective_value,
                params=params,
            )
        ]
    else:
        raise FileNotFoundError(
            f"Cannot resolve base-stage source from path: {base_source_path}. "
            "Expected an experiment root, a base_search directory, or a single base trial directory."
        )

    if selected_trial_number_zero_based is not None:
        trial_dir = stage_root / f"{int(selected_trial_number_zero_based) + 1:04d}"
        if not (trial_dir / "inference_bundle.joblib").exists():
            raise FileNotFoundError(f"Selected base trial folder not found or missing inference bundle: {trial_dir}")
        params = read_json(trial_dir / "params.json") if (trial_dir / "params.json").exists() else {}
        metrics = read_json(trial_dir / "metrics.json") if (trial_dir / "metrics.json").exists() else {}
        return stage_root, [
            BaseCandidateRecord(
                trial_number=int(selected_trial_number_zero_based),
                trial_dir=trial_dir,
                objective_value=float(metrics.get("objective_value", 0.0)),
                params=params,
            )
        ]

    summary_path = stage_root / "study_summary.csv"
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        records = _build_base_candidate_records(stage_root, summary_df, top_k=top_k)
        if records:
            return stage_root, records

    fallback_trial_dir = stage_root / "0000"
    if not (fallback_trial_dir / "inference_bundle.joblib").exists():
        raise FileNotFoundError(
            f"No usable base candidates were found under {stage_root}. "
            "study_summary.csv is missing or empty and 0000/inference_bundle.joblib does not exist."
        )
    params = read_json(fallback_trial_dir / "params.json") if (fallback_trial_dir / "params.json").exists() else {}
    metrics = read_json(fallback_trial_dir / "metrics.json") if (fallback_trial_dir / "metrics.json").exists() else {}
    return stage_root, [
        BaseCandidateRecord(
            trial_number=0,
            trial_dir=fallback_trial_dir,
            objective_value=float(metrics.get("objective_value", 0.0)),
            params=params,
        )
    ]



def _is_subpath(candidate: Path, root: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False



def _resolve_source_experiment_root(base_source_path: Path, base_stage_root: Path) -> Path | None:
    candidates: List[Path] = []
    if base_source_path.exists():
        candidates.append(base_source_path)
        if base_source_path.is_file():
            candidates.append(base_source_path.parent)
    candidates.extend([
        base_stage_root,
        base_stage_root.parent,
        base_source_path.parent,
    ])

    for candidate in candidates:
        candidate = Path(candidate)
        if not candidate.exists() or not candidate.is_dir():
            continue
        if (candidate / "base_search").exists() or (candidate / "used_config.json").exists() or (candidate / "final_summary.json").exists():
            return candidate
    if base_stage_root.parent.exists() and base_stage_root.parent != base_stage_root:
        return base_stage_root.parent
    return None



def _copy_snapshot_dir_if_possible(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_dir():
        return False
    if src.resolve() == dst.resolve():
        return False
    if _is_subpath(dst, src):
        return False
    copy_dir(src, dst)
    return True



def prepare_stage2_only_workspace(
    *,
    experiment_root: Path,
    base_source_path: Path,
    base_stage_root: Path,
) -> Dict[str, Any]:
    snapshot_root = ensure_dir(experiment_root / "source_snapshot")
    source_experiment_root = _resolve_source_experiment_root(base_source_path, base_stage_root)

    manifest: Dict[str, Any] = {
        "requested_base_source_path": str(base_source_path),
        "resolved_base_stage_root": str(base_stage_root),
        "resolved_source_experiment_root": None if source_experiment_root is None else str(source_experiment_root),
        "copied_items": [],
    }

    if source_experiment_root is not None:
        target = snapshot_root / "source_result_root"
        if _copy_snapshot_dir_if_possible(source_experiment_root, target):
            manifest["copied_items"].append({
                "type": "result_root_snapshot",
                "source": str(source_experiment_root),
                "target": str(target.relative_to(experiment_root)),
            })

    base_stage_target = snapshot_root / "reused_base_search"
    if _copy_snapshot_dir_if_possible(base_stage_root, base_stage_target):
        manifest["copied_items"].append({
            "type": "base_search_snapshot",
            "source": str(base_stage_root),
            "target": str(base_stage_target.relative_to(experiment_root)),
        })

    config_candidates = []
    if source_experiment_root is not None:
        config_candidates.append(source_experiment_root / "used_config.json")
    config_candidates.extend([
        base_stage_root.parent / "used_config.json",
        base_source_path / "used_config.json",
    ])
    seen_config_sources: set[str] = set()
    for idx, candidate in enumerate(config_candidates, start=1):
        candidate = Path(candidate)
        candidate_key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if candidate_key in seen_config_sources:
            continue
        seen_config_sources.add(candidate_key)
        if candidate.exists() and candidate.is_file():
            target = snapshot_root / f"source_used_config_{idx:02d}.json"
            target.write_text(candidate.read_text(encoding="utf-8"), encoding="utf-8")
            manifest["copied_items"].append({
                "type": "source_used_config",
                "source": str(candidate),
                "target": str(target.relative_to(experiment_root)),
            })

    save_json(manifest, experiment_root / "source_snapshot_manifest.json")
    return manifest



def run_catboost_stage_only(
    config: ExperimentConfig,
    *,
    base_source_path: str | Path,
    output_root: str | Path | None = None,
    selected_trial_number_zero_based: int | None = None,
    top_k: int | None = None,
    catboost_n_trials_override: int | None = None,
) -> Path:
    total_start = wall_clock()
    print_section("Program Start (Stage 2 Only)")

    base_source_path = Path(base_source_path)
    if output_root is None:
        output_root = config.paths.result_root
    experiment_root = create_experiment_root(str(output_root))
    catboost_root = ensure_dir(experiment_root / "catboost_search")
    print(f"[Output] experiment_root={experiment_root}")

    effective_top_k = int(config.optuna.top_base_candidates if top_k is None else top_k)
    effective_catboost_n_trials = int(config.optuna.catboost_n_trials if catboost_n_trials_override is None else catboost_n_trials_override)

    save_json(config.raw, experiment_root / "used_config.json")
    save_json(
        {
            "mode": "stage2_only",
            "base_source_path": str(base_source_path),
            "output_root": str(output_root),
            "experiment_root": str(experiment_root),
            "selected_trial_number_zero_based": None if selected_trial_number_zero_based is None else int(selected_trial_number_zero_based),
            "top_k": effective_top_k,
            "catboost_n_trials": effective_catboost_n_trials,
        },
        experiment_root / "stage2_only_request.json",
    )
    save_live_status(
        experiment_root,
        {
            "mode": "stage2_only",
            "phase": "initializing",
            "status": "running",
            "base_source_path": str(base_source_path),
            "output_root": str(output_root),
            "experiment_root": str(experiment_root),
            "selected_trial_number_zero_based": None if selected_trial_number_zero_based is None else int(selected_trial_number_zero_based),
            "top_k": effective_top_k,
            "catboost_n_trials": effective_catboost_n_trials,
            "completed_trials": 0,
            "target_trials": effective_catboost_n_trials,
        },
    )
    append_live_event(
        experiment_root,
        {
            "event": "stage2_only_initialized",
            "base_source_path": str(base_source_path),
            "experiment_root": str(experiment_root),
            "top_k": effective_top_k,
            "catboost_n_trials": effective_catboost_n_trials,
        },
    )

    try:
        set_seed(
            config.seed,
            deterministic=config.reproducibility.deterministic,
            strict_deterministic_algorithms=config.reproducibility.strict_deterministic_algorithms,
            deterministic_warn_only=config.reproducibility.deterministic_warn_only,
            cublas_workspace_config=config.reproducibility.cublas_workspace_config,
        )
        device = get_device(require_cuda=config.device.require_cuda, device_index=config.device.device_index)

        base_stage_root, base_candidates = load_existing_base_candidates(
            base_source_path,
            top_k=effective_top_k,
            selected_trial_number_zero_based=selected_trial_number_zero_based,
        )
        if not base_candidates:
            raise RuntimeError("No reusable base candidates were found, so stage 2 cannot start.")

        snapshot_manifest = prepare_stage2_only_workspace(
            experiment_root=experiment_root,
            base_source_path=base_source_path,
            base_stage_root=base_stage_root,
        )
        update_live_status(
            experiment_root,
            {
                "phase": "snapshot_completed",
                "base_stage_root": str(base_stage_root),
                "reused_base_candidate_count": len(base_candidates),
                "source_snapshot_manifest": "source_snapshot_manifest.json",
            },
        )
        append_live_event(
            experiment_root,
            {
                "event": "source_snapshot_prepared",
                "base_stage_root": str(base_stage_root),
                "copied_item_count": len(snapshot_manifest.get("copied_items", [])),
            },
        )

        raw_data = load_raw_data(train_path=config.paths.train_csv, eval_path=config.paths.eval_csv)
        save_json(
            {
                "seed": config.seed,
                "raw_train_shape": list(raw_data.train_df.shape),
                "raw_eval_shape": list(raw_data.eval_df.shape),
                "building_id": int(config.preprocess.building_id),
                "val_size": float(config.preprocess.val_size),
                "dataset_policy": "trainingData -> train/internal_validation ; validationData -> official_evaluation",
                "search_strategy": {
                    "type": "stage2_only",
                    "catboost_n_trials": effective_catboost_n_trials,
                    "top_base_candidates": effective_top_k,
                },
                "base_stage_root": str(base_stage_root),
                "source_snapshot_manifest": "source_snapshot_manifest.json",
                "base_candidates": [
                    {
                        "trial_number_zero_based": int(r.trial_number),
                        "trial_folder": r.trial_dir.name,
                        "objective_value": float(r.objective_value),
                        "trial_dir": str(r.trial_dir),
                    }
                    for r in base_candidates
                ],
            },
            experiment_root / "data_summary.json",
        )
        update_live_status(
            experiment_root,
            {
                "phase": "stage2_ready",
                "raw_train_shape": list(raw_data.train_df.shape),
                "raw_eval_shape": list(raw_data.eval_df.shape),
                "running_trial_zero_based": None,
                "running_trial_folder": None,
                "running_trial_stage": None,
                "running_trial_status": None,
            },
        )
        append_live_event(
            experiment_root,
            {
                "event": "stage2_only_ready",
                "reused_base_candidate_count": len(base_candidates),
                "raw_train_shape": list(raw_data.train_df.shape),
                "raw_eval_shape": list(raw_data.eval_df.shape),
            },
        )

        print_section("Stage 2: Local CatBoost Residual Search (Reuse Existing Base Stage)")
        catboost_objective = CatBoostTrialObjective(
            raw_data=raw_data,
            config=config,
            device=device,
            experiment_root=catboost_root,
            base_candidates=base_candidates,
            total_trials=effective_catboost_n_trials,
            live_status_root=experiment_root,
        )
        update_live_status(experiment_root, {"phase": "stage2_running"})
        append_live_event(experiment_root, {"event": "stage2_search_started"})
        catboost_study, catboost_callback, catboost_summary_df, best_catboost_trial = _run_single_stage(
            stage_root=catboost_root,
            study_name=f"{config.optuna.study_name_prefix}_catboost_only_{experiment_root.name}",
            n_trials=effective_catboost_n_trials,
            config=config,
            objective=catboost_objective,
            enable_pruning=False,
            start_time=total_start,
        )

        catboost_best_copy = copy_best_trial_to_zero(catboost_root, best_catboost_trial.number) if best_catboost_trial is not None else None
        if catboost_best_copy is not None:
            copy_dir(catboost_best_copy, experiment_root / "0000")

        catboost_success_count = int((catboost_summary_df["status"] == "success").sum()) if not catboost_summary_df.empty else 0
        catboost_pruned_count = int((catboost_summary_df["status"] == "pruned").sum()) if not catboost_summary_df.empty else 0
        catboost_failed_count = int((catboost_summary_df["status"] == "failed").sum()) if not catboost_summary_df.empty else 0
        selected_base_trial_number = None if best_catboost_trial is None else best_catboost_trial.user_attrs.get("selected_base_trial_number")

        final_summary = {
            "search_strategy": "stage2_only",
            "base_source_path": str(base_source_path),
            "base_stage_root": str(base_stage_root),
            "source_snapshot_manifest": "source_snapshot_manifest.json",
            "catboost_stage": {
                "n_trials": effective_catboost_n_trials,
                "success_trials": catboost_success_count,
                "pruned_trials": catboost_pruned_count,
                "failed_trials": catboost_failed_count,
                "best_trial_number_zero_based": None if best_catboost_trial is None else int(best_catboost_trial.number),
                "best_trial_folder": None if best_catboost_trial is None else f"{best_catboost_trial.number + 1:04d}",
                "best_trial_copy_folder": None if catboost_best_copy is None else str(catboost_best_copy.relative_to(experiment_root)),
                "best_value": None if best_catboost_trial is None else float(best_catboost_trial.value),
                "best_params": {} if best_catboost_trial is None else best_catboost_trial.params,
                "selected_base_trial_number_zero_based": None if selected_base_trial_number is None else int(selected_base_trial_number),
                "selected_base_trial_folder": None if selected_base_trial_number is None else f"{int(selected_base_trial_number) + 1:04d}",
            },
            "reused_base_candidates": [
                {
                    "trial_number_zero_based": int(r.trial_number),
                    "trial_folder": f"{r.trial_number + 1:04d}",
                    "objective_value": float(r.objective_value),
                }
                for r in base_candidates
            ],
            "final_selected_bundle": None if catboost_best_copy is None else "0000/inference_bundle.joblib",
            "total_runtime_seconds": float(wall_clock() - total_start),
            "total_runtime_readable": format_seconds(wall_clock() - total_start),
        }
        save_json(final_summary, experiment_root / "final_summary.json")
        summary_text = [
            "=" * 110,
            "Stage-2-Only Experiment Summary",
            "=" * 110,
            f"base_source_path: {base_source_path}",
            f"base_stage_root: {base_stage_root}",
            f"stage2_catboost_trials: {effective_catboost_n_trials}",
            f"reused_base_candidates: {len(base_candidates)}",
            f"snapshot_manifest: source_snapshot_manifest.json",
            f"stage2_best_folder: {None if best_catboost_trial is None else f'{best_catboost_trial.number + 1:04d}'}",
            f"stage2_best_value: {None if best_catboost_trial is None else float(best_catboost_trial.value)}",
            f"selected_base_for_best_catboost: {None if selected_base_trial_number is None else f'{int(selected_base_trial_number) + 1:04d}'}",
            f"elapsed: {format_seconds(wall_clock() - total_start)}",
            "",
            "[CatBoost Best Params]",
            "{}" if best_catboost_trial is None else str(best_catboost_trial.params),
            "",
        ]
        save_study_text_summary("\n".join(summary_text), experiment_root)
        update_live_status(
            experiment_root,
            {
                "phase": "finished",
                "status": "success",
                "running_trial_zero_based": None,
                "running_trial_folder": None,
                "running_trial_stage": None,
                "running_trial_status": None,
                "final_summary_path": "final_summary.json",
                "best_trial_folder": None if best_catboost_trial is None else f"{best_catboost_trial.number + 1:04d}",
                "best_trial_value": None if best_catboost_trial is None else float(best_catboost_trial.value),
                "total_runtime_seconds": float(wall_clock() - total_start),
                "total_runtime_readable": format_seconds(wall_clock() - total_start),
            },
        )
        append_live_event(
            experiment_root,
            {
                "event": "stage2_only_finished",
                "best_trial_folder": None if best_catboost_trial is None else f"{best_catboost_trial.number + 1:04d}",
                "best_trial_value": None if best_catboost_trial is None else float(best_catboost_trial.value),
                "total_runtime_seconds": float(wall_clock() - total_start),
            },
        )

        print_section("Stage-2-Only Experiment Finished")
        if best_catboost_trial is not None:
            print(f"[BestCatBoostTrial] folder={best_catboost_trial.number + 1:04d} value={float(best_catboost_trial.value):.8f}")
            print(f"[BestCatBoostSelectedBase] folder={int(selected_base_trial_number) + 1:04d}")
        print(f"[Done] total_time={format_seconds(wall_clock() - total_start)}")
        return experiment_root
    except Exception as exc:
        error_payload = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "base_source_path": str(base_source_path),
            "runtime_seconds": float(wall_clock() - total_start),
        }
        save_json(error_payload, experiment_root / "stage2_only_error.json")
        update_live_status(
            experiment_root,
            {
                "phase": "failed",
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "running_trial_status": "failed",
                "total_runtime_seconds": float(wall_clock() - total_start),
                "total_runtime_readable": format_seconds(wall_clock() - total_start),
            },
        )
        append_live_event(
            experiment_root,
            {
                "event": "stage2_only_failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "total_runtime_seconds": float(wall_clock() - total_start),
            },
        )
        raise

