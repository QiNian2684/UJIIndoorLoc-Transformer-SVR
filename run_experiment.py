import os
import argparse
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")

from src.config import load_config
from src.experiment import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WiFi 指纹定位：分阶段 Optuna 调参（Transformer+SVR 基模型 -> Local CatBoost Residual）"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.json",
        help="配置文件路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    experiment_root = run_experiment(config)
    print(f"\n实验结果目录：{experiment_root}")


if __name__ == "__main__":
    main()
