import os
import argparse
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")

from src.config import load_config
from src.experiment import run_catboost_stage_only
from src.utils import read_json


def _normalize_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _find_historical_config_path(base_source: Path) -> Path | None:
    search_roots = [base_source]
    if base_source.is_file():
        search_roots.append(base_source.parent)
    if base_source.is_dir():
        search_roots.extend([base_source.parent, base_source.parent.parent])

    for root in search_roots:
        candidate = root / "used_config.json"
        if candidate.exists():
            return candidate
        candidate = root / "configs" / "default_config.json"
        if candidate.exists():
            return candidate
    return None


def _load_stage2_settings(config_path: Path) -> dict[str, Any]:
    raw = read_json(config_path)
    settings = raw.get("stage2_only", {})
    if settings is None:
        return {}
    if not isinstance(settings, dict):
        raise TypeError("配置文件里的 stage2_only 必须是对象。")
    return settings


def _resolve_setting(cli_value: Any, config_value: Any) -> Any:
    return cli_value if cli_value is not None else config_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WiFi 指纹定位：跳过第一阶段，直接复用既有结果训练第二阶段 Residual CatBoost"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config_stage2.json",
        help="第二阶段配置文件路径，默认读取 configs/default_config_stage2.json",
    )
    parser.add_argument(
        "--base-source",
        type=str,
        default=None,
        help="已完成第一阶段的结果来源路径。若不提供，则从配置文件 stage2_only.base_source 读取。",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="第二阶段结果输出根目录。若不提供，则从配置文件 stage2_only.output_root 读取；仍为空时默认落到 config.paths.result_root 下的新时间戳目录。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="复用的第一阶段候选数。若不提供，则从配置文件 stage2_only.top_k 读取；仍为空时沿用 optuna.top_base_candidates。",
    )
    parser.add_argument(
        "--catboost-n-trials",
        type=int,
        default=None,
        help="第二阶段 CatBoost trial 数。若不提供，则从配置文件 stage2_only.catboost_n_trials 读取；仍为空时沿用 optuna.catboost_n_trials。",
    )
    parser.add_argument(
        "--selected-base-trial",
        type=int,
        default=None,
        help="只复用指定的一阶段 trial（零基编号）。若不提供，则从配置文件 stage2_only.selected_base_trial 读取。",
    )
    return parser.parse_args()


def run_stage2_only_entry(
    *,
    launcher_config_path: str | Path = "configs/default_config_stage2.json",
    base_source: str | Path | None = None,
    output_root: str | Path | None = None,
    top_k: int | None = None,
    catboost_n_trials: int | None = None,
    selected_base_trial: int | None = None,
) -> Path:
    launcher_config_path = Path(launcher_config_path)
    launcher_config = load_config(launcher_config_path)
    stage2_settings = _load_stage2_settings(launcher_config_path)

    resolved_base_source = _normalize_optional_str(_resolve_setting(base_source, stage2_settings.get("base_source")))
    if resolved_base_source is None:
        raise ValueError(
            "没有找到第一阶段结果目录。\n"
            "请直接修改 configs/default_config_stage2.json 里的 stage2_only.base_source，"
            "然后直接运行 run_stage2_only.py。"
        )
    base_source_path = Path(resolved_base_source)

    historical_config_path = _find_historical_config_path(base_source_path)
    if historical_config_path is not None and historical_config_path.resolve() != launcher_config_path.resolve():
        print(f"[Config] 检测到历史配置: {historical_config_path}")
        print(f"[Config] 本次 stage2-only 运行不会再复用历史配置；严格使用: {launcher_config_path}")
    else:
        print(f"[Config] 使用第二阶段配置文件: {launcher_config_path}")

    resolved_output_root = _normalize_optional_str(_resolve_setting(output_root, stage2_settings.get("output_root")))
    resolved_top_k = _resolve_setting(top_k, stage2_settings.get("top_k"))
    resolved_catboost_n_trials = _resolve_setting(catboost_n_trials, stage2_settings.get("catboost_n_trials"))
    resolved_selected_base_trial = _resolve_setting(selected_base_trial, stage2_settings.get("selected_base_trial"))

    experiment_root = run_catboost_stage_only(
        launcher_config,
        base_source_path=base_source_path,
        output_root=resolved_output_root,
        selected_trial_number_zero_based=resolved_selected_base_trial,
        top_k=resolved_top_k,
        catboost_n_trials_override=resolved_catboost_n_trials,
    )
    print(f"\n第二阶段结果目录：{experiment_root}")
    return Path(experiment_root)


def main() -> None:
    args = parse_args()
    run_stage2_only_entry(
        launcher_config_path=args.config,
        base_source=args.base_source,
        output_root=args.output_root,
        top_k=args.top_k,
        catboost_n_trials=args.catboost_n_trials,
        selected_base_trial=args.selected_base_trial,
    )


if __name__ == "__main__":
    main()
