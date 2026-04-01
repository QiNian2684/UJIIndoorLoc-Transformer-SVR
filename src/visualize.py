from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
FIG_DPI = 450

def _prepare_path(save_path: str | Path) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    return save_path

def save_loss_curve(history: Dict[str, Iterable[float]], save_path: str | Path, log_scale: bool = False) -> None:
    save_path = _prepare_path(save_path)
    train_loss = list(history.get("train_loss", []))
    val_loss = list(history.get("val_loss", []))
    epochs = np.arange(1, len(train_loss) + 1)
    if len(epochs) == 0:
        return
    plt.figure(figsize=(10, 7))
    plt.plot(epochs, train_loss, label="train_loss", linewidth=2.2)
    plt.plot(epochs, val_loss, label="val_loss", linewidth=2.2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Transformer Autoencoder Training Curve")
    if log_scale:
        plt.yscale("log")
        plt.title("Transformer Autoencoder Training Curve (Log Scale)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()

def save_error_cdf(error_distances: np.ndarray, save_path: str | Path, title: str) -> None:
    save_path = _prepare_path(save_path)
    error_distances = np.asarray(error_distances, dtype=np.float64)
    error_distances = error_distances[np.isfinite(error_distances)]
    if error_distances.size == 0:
        return
    sorted_errors = np.sort(error_distances)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    plt.figure(figsize=(10, 7))
    plt.plot(sorted_errors, cdf, linewidth=2.4)
    plt.xlabel("Error Distance")
    plt.ylabel("CDF")
    plt.title(title)
    plt.xlim(left=0)
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()

def save_true_vs_pred_scatter(y_true: np.ndarray, y_pred: np.ndarray, save_path: str | Path, title: str) -> None:
    save_path = _prepare_path(save_path)
    if len(y_true) == 0:
        return
    plt.figure(figsize=(9, 9))
    plt.scatter(y_true[:, 0], y_true[:, 1], s=18, alpha=0.65, label="True")
    plt.scatter(y_pred[:, 0], y_pred[:, 1], s=18, alpha=0.65, label="Predicted")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()

def save_error_histogram(error_distances: np.ndarray, save_path: str | Path, title: str) -> None:
    save_path = _prepare_path(save_path)
    error_distances = np.asarray(error_distances, dtype=np.float64)
    error_distances = error_distances[np.isfinite(error_distances)]
    if error_distances.size == 0:
        return
    plt.figure(figsize=(10, 7))
    plt.hist(error_distances, bins=min(50, max(10, error_distances.size // 5)), alpha=0.85)
    plt.xlabel("Error Distance")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()

def save_error_boxplot(error_distances: np.ndarray, save_path: str | Path, title: str) -> None:
    save_path = _prepare_path(save_path)
    error_distances = np.asarray(error_distances, dtype=np.float64)
    error_distances = error_distances[np.isfinite(error_distances)]
    if error_distances.size == 0:
        return
    plt.figure(figsize=(10, 6))
    plt.boxplot(error_distances, vert=False, showmeans=True)
    plt.xlabel("Error Distance")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()

def save_residual_scatter(y_true: np.ndarray, y_pred: np.ndarray, save_path: str | Path, title: str) -> None:
    save_path = _prepare_path(save_path)
    if len(y_true) == 0:
        return
    residuals = y_pred - y_true
    plt.figure(figsize=(10, 7))
    plt.scatter(residuals[:, 0], residuals[:, 1], s=18, alpha=0.65)
    plt.axhline(0.0, linewidth=1.2)
    plt.axvline(0.0, linewidth=1.2)
    plt.xlabel("Longitude Residual")
    plt.ylabel("Latitude Residual")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()

def save_metric_trace(history: Dict[str, Iterable[float]], save_path: str | Path) -> None:
    save_path = _prepare_path(save_path)
    epochs = np.array(list(history.get("epoch", [])), dtype=np.int32)
    if len(epochs) == 0:
        return
    patience = np.array(list(history.get("patience_counter", [])), dtype=np.float64)
    best_running = np.array(list(history.get("best_val_loss_running", [])), dtype=np.float64)
    plt.figure(figsize=(10, 7))
    plt.plot(epochs, best_running, label="best_val_loss_running", linewidth=2.2)
    plt.plot(epochs, patience, label="patience_counter", linewidth=2.0)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Monitoring Trace")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()

def save_optimization_curve(values: list[float], save_path: str | Path) -> None:
    save_path = _prepare_path(save_path)
    if len(values) == 0:
        return
    trial_ids = np.arange(1, len(values) + 1)
    best_so_far = np.minimum.accumulate(values)
    plt.figure(figsize=(11, 7))
    plt.plot(trial_ids, values, label="trial_value", linewidth=1.7)
    plt.plot(trial_ids, best_so_far, label="best_so_far", linewidth=2.4)
    plt.xlabel("Trial")
    plt.ylabel("Validation Mean Error Distance")
    plt.title("Optuna Optimization History")
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()

def save_trial_value_scatter(values: list[float], save_path: str | Path) -> None:
    save_path = _prepare_path(save_path)
    if len(values) == 0:
        return
    trial_ids = np.arange(1, len(values) + 1)
    plt.figure(figsize=(11, 7))
    plt.scatter(trial_ids, values, s=24, alpha=0.75)
    plt.xlabel("Trial")
    plt.ylabel("Validation Mean Error Distance")
    plt.title("Trial Objective Scatter")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()

def save_trial_state_bar(state_counts: Dict[str, int], save_path: str | Path) -> None:
    save_path = _prepare_path(save_path)
    if not state_counts:
        return
    labels = list(state_counts.keys())
    values = [state_counts[k] for k in labels]
    plt.figure(figsize=(9, 6))
    plt.bar(labels, values)
    plt.xlabel("Trial State")
    plt.ylabel("Count")
    plt.title("Trial State Distribution")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
