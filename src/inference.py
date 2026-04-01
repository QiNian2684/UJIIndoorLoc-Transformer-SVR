from __future__ import annotations

from pathlib import Path
from typing import Any

import gc
import joblib
import numpy as np
import pandas as pd
import torch

from .calibration import apply_residual_calibration, compute_reconstruction_error
from .models import WiFiTransformerAutoencoder


DEFAULT_INFERENCE_BATCH_SIZE = 2048


def load_inference_bundle(bundle_path: str | Path) -> dict[str, Any]:
    return joblib.load(bundle_path)


@torch.no_grad()
def _forward_in_batches(
    model: WiFiTransformerAutoencoder,
    X_scaled: np.ndarray,
    device: torch.device,
    batch_size: int,
    include_reconstruction: bool = False,
) -> dict[str, np.ndarray]:
    features: list[np.ndarray] = []
    reconstructions: list[np.ndarray] = []
    for start in range(0, len(X_scaled), max(1, int(batch_size))):
        end = min(len(X_scaled), start + max(1, int(batch_size)))
        xb = torch.from_numpy(X_scaled[start:end]).to(device)
        feat_t = model.encode(xb)
        features.append(feat_t.detach().cpu().numpy().astype(np.float32))
        if include_reconstruction:
            recon_t = model.decoder(feat_t)
            reconstructions.append(recon_t.detach().cpu().numpy().astype(np.float32))
            del recon_t
        del xb, feat_t
    payload: dict[str, np.ndarray] = {
        "latent_features": np.vstack(features) if features else np.empty((0, model.model_dim), dtype=np.float32),
    }
    if include_reconstruction:
        payload["reconstruction"] = np.vstack(reconstructions) if reconstructions else np.empty_like(X_scaled, dtype=np.float32)
    return payload


@torch.no_grad()
def predict_with_saved_bundle(
    bundle_path: str | Path,
    dataframe: pd.DataFrame,
    device: str | torch.device = "cpu",
    return_base_prediction: bool = False,
    inference_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    bundle = load_inference_bundle(bundle_path)
    device = torch.device(device)
    feature_cols = list(range(520))
    X = dataframe.iloc[:, feature_cols].replace(100, -105).fillna(-105)
    X = X.loc[:, bundle["feature_mask"]]
    X_scaled = bundle["scaler_X"].transform(X.to_numpy()).astype(np.float32)

    model = WiFiTransformerAutoencoder(**bundle["transformer_config"])
    model.load_state_dict(bundle["transformer_state_dict"])
    model.to(device)
    model.eval()

    try:
        payload = _forward_in_batches(
            model,
            X_scaled,
            device,
            batch_size=inference_batch_size,
            include_reconstruction=True,
        )
        features = payload["latent_features"]
        base_pred = np.asarray(bundle["svr_model"].predict(features), dtype=np.float64)
        reconstruction_error = compute_reconstruction_error(X_scaled, payload.get("reconstruction"))
        corrected_pred, _, _ = apply_residual_calibration(
            bundle.get("residual_calibrator"),
            latent_features=features,
            base_predictions=base_pred,
            reconstruction_error=reconstruction_error,
            signal_values=X_scaled,
        )
        if return_base_prediction:
            return corrected_pred, base_pred
        return corrected_pred
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.empty_cache()
