from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from .optuna_compat import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .storage import save_history_csv
from .utils import NaNLossError, format_seconds, print_section

EpochCallback = Callable[[Dict[str, Any]], None]


def _masked_weighted_reconstruction_components(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    observed_loss_weight: float,
    missing_loss_weight: float,
    missing_value_threshold: float,
) -> Dict[str, torch.Tensor]:
    squared_error = (pred - target).pow(2)
    missing_mask = (target <= missing_value_threshold).to(dtype=pred.dtype)
    observed_mask = 1.0 - missing_mask

    observed_sq_sum = (squared_error * observed_mask).sum()
    missing_sq_sum = (squared_error * missing_mask).sum()
    observed_count = observed_mask.sum()
    missing_count = missing_mask.sum()

    observed_loss = observed_sq_sum / observed_count.clamp_min(1.0)
    missing_loss = missing_sq_sum / missing_count.clamp_min(1.0)
    total_loss = observed_loss_weight * observed_loss + missing_loss_weight * missing_loss
    return {
        "total_loss": total_loss,
        "observed_loss": observed_loss,
        "missing_loss": missing_loss,
        "observed_sq_sum": observed_sq_sum,
        "missing_sq_sum": missing_sq_sum,
        "observed_count": observed_count,
        "missing_count": missing_count,
    }


def _compute_epoch_loss_summary(
    observed_sq_sum: float,
    observed_count: float,
    missing_sq_sum: float,
    missing_count: float,
    *,
    observed_loss_weight: float,
    missing_loss_weight: float,
) -> tuple[float, float, float, float, float]:
    obs_loss = float(observed_sq_sum / max(observed_count, 1.0))
    miss_loss = float(missing_sq_sum / max(missing_count, 1.0))
    total_loss = float(observed_loss_weight * obs_loss + missing_loss_weight * miss_loss)
    observed_fraction = float(observed_count / max(observed_count + missing_count, 1.0))
    missing_fraction = float(missing_count / max(observed_count + missing_count, 1.0))
    return total_loss, obs_loss, miss_loss, observed_fraction, missing_fraction


def train_autoencoder(
    model: nn.Module,
    X_train: np.ndarray,
    X_val: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    early_stopping_patience: int,
    min_delta_ratio: float,
    train_log_interval: int,
    num_workers: int = 0,
    weight_decay: float = 1e-5,
    observed_loss_weight: float = 1.0,
    missing_loss_weight: float = 0.05,
    missing_value_threshold: float = 1e-6,
    trial: optuna.Trial | None = None,
    enable_pruning: bool = False,
    status_prefix: str = "",
    history_save_path: str | Path | None = None,
    epoch_callback: EpochCallback | None = None,
) -> Tuple[nn.Module, Dict[str, List[float] | int | float | bool | str | None]]:
    print_section(f"{status_prefix} Train Transformer Autoencoder".strip())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    pin_memory = device.type == "cuda"
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(X_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(X_val))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state = None
    history: Dict[str, List[float] | int | float | bool | str | None] = {
        "epoch": [], "train_loss": [], "val_loss": [], "train_loss_delta": [], "val_loss_delta": [],
        "train_observed_loss": [], "val_observed_loss": [], "train_missing_loss": [], "val_missing_loss": [],
        "train_observed_fraction": [], "val_observed_fraction": [], "train_missing_fraction": [], "val_missing_fraction": [],
        "val_improvement_ratio": [], "best_val_loss_running": [], "patience_counter": [],
        "learning_rate": [], "epoch_seconds": [], "elapsed_seconds": [],
        "loss_name": "masked_weighted_reconstruction_mse",
        "observed_loss_weight": float(observed_loss_weight),
        "missing_loss_weight": float(missing_loss_weight),
        "missing_value_threshold": float(missing_value_threshold),
        "best_epoch": 0, "best_val_loss": None, "pruned": False, "pruned_epoch": None, "stop_reason": "finished",
    }
    start_time = time.time()
    total_train_batches = len(train_loader)
    total_val_batches = len(val_loader)
    print(f"[Train] {status_prefix} epochs={epochs}")
    print(f"[Train] {status_prefix} batch_size={batch_size}")
    print(f"[Train] {status_prefix} learning_rate={learning_rate}")
    print(f"[Train] {status_prefix} weight_decay={weight_decay}")
    print(f"[Train] {status_prefix} observed_loss_weight={observed_loss_weight}")
    print(f"[Train] {status_prefix} missing_loss_weight={missing_loss_weight}")
    print(f"[Train] {status_prefix} missing_value_threshold={missing_value_threshold}")
    print(f"[Train] {status_prefix} early_stopping_patience={early_stopping_patience}")
    print(f"[Train] {status_prefix} min_delta_ratio={min_delta_ratio}")
    print(f"[Train] {status_prefix} enable_pruning={enable_pruning}")
    prev_train_loss = None
    prev_val_loss = None
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        print(f"[Progress] {status_prefix} Epoch {epoch}/{epochs} started.")
        model.train()
        train_observed_sq_sum = 0.0
        train_missing_sq_sum = 0.0
        train_observed_count = 0.0
        train_missing_count = 0.0
        for batch_idx, (xb, _) in enumerate(train_loader, start=1):
            xb = xb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss_parts = _masked_weighted_reconstruction_components(
                pred,
                xb,
                observed_loss_weight=observed_loss_weight,
                missing_loss_weight=missing_loss_weight,
                missing_value_threshold=missing_value_threshold,
            )
            loss = loss_parts["total_loss"]
            if torch.isnan(loss) or torch.isinf(loss):
                raise NaNLossError("Training loss became NaN or Inf.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_observed_sq_sum += float(loss_parts["observed_sq_sum"].detach().item())
            train_missing_sq_sum += float(loss_parts["missing_sq_sum"].detach().item())
            train_observed_count += float(loss_parts["observed_count"].detach().item())
            train_missing_count += float(loss_parts["missing_count"].detach().item())
            if batch_idx == 1 or batch_idx == total_train_batches or batch_idx % train_log_interval == 0:
                print(
                    f"[Train] {status_prefix} epoch={epoch:03d} batch={batch_idx:04d}/{total_train_batches:04d} "
                    f"batch_total_loss={float(loss.detach().item()):.8f} "
                    f"batch_obs_loss={float(loss_parts['observed_loss'].detach().item()):.8f} "
                    f"batch_missing_loss={float(loss_parts['missing_loss'].detach().item()):.8f}"
                )
            del pred, loss, xb, loss_parts
        avg_train_loss, avg_train_observed_loss, avg_train_missing_loss, train_observed_fraction, train_missing_fraction = _compute_epoch_loss_summary(
            train_observed_sq_sum,
            train_observed_count,
            train_missing_sq_sum,
            train_missing_count,
            observed_loss_weight=observed_loss_weight,
            missing_loss_weight=missing_loss_weight,
        )
        model.eval()
        val_observed_sq_sum = 0.0
        val_missing_sq_sum = 0.0
        val_observed_count = 0.0
        val_missing_count = 0.0
        with torch.no_grad():
            for batch_idx, (xb, _) in enumerate(val_loader, start=1):
                xb = xb.to(device, non_blocking=True)
                pred = model(xb)
                loss_parts = _masked_weighted_reconstruction_components(
                    pred,
                    xb,
                    observed_loss_weight=observed_loss_weight,
                    missing_loss_weight=missing_loss_weight,
                    missing_value_threshold=missing_value_threshold,
                )
                val_observed_sq_sum += float(loss_parts["observed_sq_sum"].detach().item())
                val_missing_sq_sum += float(loss_parts["missing_sq_sum"].detach().item())
                val_observed_count += float(loss_parts["observed_count"].detach().item())
                val_missing_count += float(loss_parts["missing_count"].detach().item())
                if batch_idx == 1 or batch_idx == total_val_batches:
                    print(
                        f"[Validate] {status_prefix} epoch={epoch:03d} batch={batch_idx:04d}/{total_val_batches:04d} "
                        f"batch_total_loss={float(loss_parts['total_loss'].detach().item()):.8f} "
                        f"batch_obs_loss={float(loss_parts['observed_loss'].detach().item()):.8f} "
                        f"batch_missing_loss={float(loss_parts['missing_loss'].detach().item()):.8f}"
                    )
                del pred, xb, loss_parts
        avg_val_loss, avg_val_observed_loss, avg_val_missing_loss, val_observed_fraction, val_missing_fraction = _compute_epoch_loss_summary(
            val_observed_sq_sum,
            val_observed_count,
            val_missing_sq_sum,
            val_missing_count,
            observed_loss_weight=observed_loss_weight,
            missing_loss_weight=missing_loss_weight,
        )
        train_loss_delta = None if prev_train_loss is None else float(avg_train_loss - prev_train_loss)
        val_loss_delta = None if prev_val_loss is None else float(avg_val_loss - prev_val_loss)
        if best_val_loss == float("inf"):
            improvement_ratio = None
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[Checkpoint] {status_prefix} initial best model saved at epoch {epoch}.")
        else:
            improvement_ratio = float((best_val_loss - avg_val_loss) / max(best_val_loss, 1e-12))
            if improvement_ratio > min_delta_ratio:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                patience_counter = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print(f"[Checkpoint] {status_prefix} new best model at epoch {epoch} | improvement={improvement_ratio * 100:.4f}% | best_val_loss={best_val_loss:.8f}")
            else:
                patience_counter += 1
                print(f"[EarlyStop] {status_prefix} no sufficient improvement | patience={patience_counter}/{early_stopping_patience}")
        epoch_seconds = float(time.time() - epoch_start)
        elapsed_seconds = float(time.time() - start_time)
        history["epoch"].append(int(epoch))
        history["train_loss"].append(float(avg_train_loss))
        history["val_loss"].append(float(avg_val_loss))
        history["train_observed_loss"].append(float(avg_train_observed_loss))
        history["val_observed_loss"].append(float(avg_val_observed_loss))
        history["train_missing_loss"].append(float(avg_train_missing_loss))
        history["val_missing_loss"].append(float(avg_val_missing_loss))
        history["train_observed_fraction"].append(float(train_observed_fraction))
        history["val_observed_fraction"].append(float(val_observed_fraction))
        history["train_missing_fraction"].append(float(train_missing_fraction))
        history["val_missing_fraction"].append(float(val_missing_fraction))
        history["train_loss_delta"].append(None if train_loss_delta is None else float(train_loss_delta))
        history["val_loss_delta"].append(None if val_loss_delta is None else float(val_loss_delta))
        history["val_improvement_ratio"].append(None if improvement_ratio is None else float(improvement_ratio))
        history["best_val_loss_running"].append(float(best_val_loss))
        history["patience_counter"].append(int(patience_counter))
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
        history["epoch_seconds"].append(epoch_seconds)
        history["elapsed_seconds"].append(elapsed_seconds)
        print(
            f"[EpochSummary] {status_prefix} epoch={epoch:03d}/{epochs:03d} "
            f"train_loss={avg_train_loss:.8f} val_loss={avg_val_loss:.8f} "
            f"train_obs_loss={avg_train_observed_loss:.8f} val_obs_loss={avg_val_observed_loss:.8f} "
            f"train_missing_loss={avg_train_missing_loss:.8f} val_missing_loss={avg_val_missing_loss:.8f} "
            f"obs_fraction={val_observed_fraction:.4f} miss_fraction={val_missing_fraction:.4f} "
            f"best_val_loss={best_val_loss:.8f} epoch_time={format_seconds(epoch_seconds)} elapsed={format_seconds(elapsed_seconds)}"
        )
        if history_save_path is not None:
            save_history_csv(history, history_save_path)
        if epoch_callback is not None:
            epoch_callback({
                "epoch": int(epoch),
                "train_loss": float(avg_train_loss),
                "val_loss": float(avg_val_loss),
                "train_observed_loss": float(avg_train_observed_loss),
                "val_observed_loss": float(avg_val_observed_loss),
                "train_missing_loss": float(avg_train_missing_loss),
                "val_missing_loss": float(avg_val_missing_loss),
                "best_val_loss": float(best_val_loss),
                "patience_counter": int(patience_counter),
                "elapsed_seconds": elapsed_seconds,
            })
        if trial is not None:
            trial.report(float(avg_val_loss), step=epoch)
        if enable_pruning and trial is not None and trial.should_prune():
            history["pruned"] = True
            history["pruned_epoch"] = int(epoch)
            history["stop_reason"] = "optuna_pruned"
            print(f"[Pruning] {status_prefix} Optuna pruned this trial at epoch {epoch}.")
            break
        if patience_counter >= early_stopping_patience:
            history["stop_reason"] = "early_stopping"
            print(f"[EarlyStop] {status_prefix} stop triggered. Restoring epoch {best_epoch} with best_val_loss={best_val_loss:.8f}.")
            break
        prev_train_loss = float(avg_train_loss)
        prev_val_loss = float(avg_val_loss)
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        print(f"[Restore] {status_prefix} best model restored from epoch {best_epoch}.")
    history["best_val_loss"] = float(best_val_loss)
    history["best_epoch"] = int(best_epoch)
    if history["stop_reason"] == "finished" and len(history["epoch"]) < epochs:
        history["stop_reason"] = "stopped"
    print(f"[TrainComplete] {status_prefix} total_time={format_seconds(time.time() - start_time)}")
    return model, history


@torch.no_grad()
def extract_feature_payload(
    model: nn.Module,
    X_data: np.ndarray,
    device: torch.device,
    batch_size: int,
    stage_name: str,
    num_workers: int = 0,
    include_reconstruction: bool = False,
) -> Dict[str, np.ndarray]:
    print_section(stage_name)
    pin_memory = device.type == "cuda"
    loader = DataLoader(torch.from_numpy(X_data), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    model.eval()
    latent_features: list[np.ndarray] = []
    reconstructions: list[np.ndarray] = []
    total_batches = len(loader)
    start_time = time.time()
    print(f"[Feature] {stage_name} sample_count={len(X_data)}")
    print(f"[Feature] {stage_name} batch_size={batch_size}")
    print(f"[Feature] {stage_name} total_batches={total_batches}")
    print(f"[Feature] {stage_name} include_reconstruction={include_reconstruction}")
    for batch_idx, xb in enumerate(loader, start=1):
        xb = xb.to(device, non_blocking=True)
        feat = model.encode(xb)
        latent_features.append(feat.detach().cpu().numpy())
        if include_reconstruction:
            recon = model.decoder(feat)
            reconstructions.append(recon.detach().cpu().numpy())
            del recon
        if batch_idx == 1 or batch_idx == total_batches or batch_idx % max(1, total_batches // 5) == 0:
            print(f"[Feature] {stage_name} batch={batch_idx:04d}/{total_batches:04d}")
        del xb, feat

    feature_matrix = np.concatenate(latent_features, axis=0) if latent_features else np.empty((0, 0), dtype=np.float32)
    payload: Dict[str, np.ndarray] = {"latent_features": feature_matrix.astype(np.float32)}
    if include_reconstruction:
        recon_matrix = np.concatenate(reconstructions, axis=0) if reconstructions else np.empty((0, 0), dtype=np.float32)
        payload["reconstruction"] = recon_matrix.astype(np.float32)

    print(
        f"[Feature] {stage_name} latent_shape={payload['latent_features'].shape}"
        + (f" reconstruction_shape={payload['reconstruction'].shape}" if 'reconstruction' in payload else "")
        + f" elapsed={format_seconds(time.time() - start_time)}"
    )
    return payload


@torch.no_grad()
def extract_features(model: nn.Module, X_data: np.ndarray, device: torch.device, batch_size: int, stage_name: str, num_workers: int = 0) -> np.ndarray:
    payload = extract_feature_payload(
        model=model,
        X_data=X_data,
        device=device,
        batch_size=batch_size,
        stage_name=stage_name,
        num_workers=num_workers,
        include_reconstruction=False,
    )
    return payload["latent_features"]
