from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader


def _clone_batch(X: torch.Tensor) -> torch.Tensor:
    return X.detach().clone()


def _mask_features(X: torch.Tensor, value_dim: int, mask_dim: int, ratio: float,
                   value_to_channel: Optional[Sequence[int]] = None) -> torch.Tensor:
    """
    Randomly drop observed entries to simulate missingness.
    """
    if mask_dim <= 0:
        return X
    output = X.clone()
    values = output[:, :, :value_dim]
    masks = output[:, :, value_dim:]
    observed = masks > 0.5
    drop = torch.rand_like(masks) < ratio
    removed = observed & drop
    if value_to_channel is not None:
        idx = torch.tensor(value_to_channel, device=output.device, dtype=torch.long)
        removal_values = torch.index_select(removed, dim=2, index=idx)
    else:
        removal_values = removed
    values = values.masked_fill(removal_values, 0.0)
    masks = masks * (~removed)
    output[:, :, :value_dim] = values
    output[:, :, value_dim:] = masks.float()
    return output


def _truncate_sequence(X: torch.Tensor, ratio: float) -> torch.Tensor:
    """
    Zero out the tail of the sequence according to the ratio of timesteps to keep.
    """
    B, T, _ = X.shape
    keep_steps = max(int(math.ceil(T * ratio)), 1)
    if keep_steps >= T:
        return X
    truncated = X.clone()
    truncated[:, keep_steps:, :] = 0.0
    return truncated


def _evaluate_with_transform(model, loader: DataLoader, device: str, transform_fn,
                             metric_fn, max_batches: int = 8):
    model.eval()
    probs, labels = [], []
    seen = 0
    with torch.no_grad():
        for X, D, y in loader:
            X = X.to(device, non_blocking=True)
            D = D.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            X_aug = transform_fn(X)
            logits, _ = model(X_aug, D)
            probs.append(logits.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())
            seen += 1
            if seen >= max_batches:
                break
    if not probs:
        return {}
    import numpy as np

    y_true = np.concatenate(labels).ravel()
    y_prob = np.concatenate(probs).ravel()
    metrics = metric_fn(y_true, y_prob)
    metrics = {k: float(v) for k, v in metrics.items()}
    return metrics


def run_missingness_curve(model, loader, device: str, metric_fn, mask_rates: Iterable[float],
                          max_batches: int, value_dim: int, mask_dim: int,
                          value_to_channel: Optional[Sequence[int]] = None):
    curves = []
    for rate in mask_rates:
        def transform(X, rate=rate):
            return _mask_features(X, value_dim, mask_dim, rate, value_to_channel)
        metrics = _evaluate_with_transform(model, loader, device, transform, metric_fn, max_batches)
        curves.append({"mask_rate": float(rate), "metrics": metrics})
    return curves


def run_truncation_curve(model, loader, device: str, metric_fn, truncation_ratios: Iterable[float],
                         max_batches: int):
    curves = []
    for ratio in truncation_ratios:
        def transform(X, ratio=ratio):
            return _truncate_sequence(X, ratio)
        metrics = _evaluate_with_transform(model, loader, device, transform, metric_fn, max_batches)
        curves.append({"keep_ratio": float(ratio), "metrics": metrics})
    return curves
