from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from collections import defaultdict

import torch
from torch import nn

try:  # pragma: no cover - optional imports
    from model_codes.ConCare_Model_v3 import (
        MissingAwareTemporalAttention,
        SingleAttentionPerFeatureNew,
        FinalAttentionQKV,
    )
except Exception:  # pragma: no cover
    MissingAwareTemporalAttention = tuple()  # type: ignore
    SingleAttentionPerFeatureNew = tuple()   # type: ignore
    FinalAttentionQKV = tuple()              # type: ignore


ATTENTION_TYPES = tuple(t for t in (
    MissingAwareTemporalAttention,
    SingleAttentionPerFeatureNew,
    FinalAttentionQKV,
) if t)


@dataclass
class SignalRecorderConfig:
    """
    Configuration controlling how many batches to probe and which stats to capture.
    """
    max_batches_per_epoch: int = 2
    quantiles: tuple = (0.05, 0.5, 0.95)
    save_dir: Path = Path("instrumentation_data")
    run_id: Optional[str] = None


def _group_key(param_name: str) -> str:
    if "." not in param_name:
        return param_name
    return param_name.split(".")[0]


def _tensor_summary(tensor: torch.Tensor, quantiles: tuple) -> Dict[str, float]:
    if tensor is None:
        return {}
    flat = tensor.detach().to(torch.float32).reshape(-1)
    if flat.numel() == 0:
        return {}
    summary = {
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "max": float(flat.max().item()),
        "min": float(flat.min().item()),
    }
    q_vals = torch.quantile(flat, torch.tensor(list(quantiles), device=flat.device))
    for idx, q in enumerate(quantiles):
        summary[f"q{int(q*100)}"] = float(q_vals[idx].item())
    return summary


class SignalRecorder:
    """
    Register lightweight hooks on key modules to capture activations, attention weights,
    and gradient norms for a few probe batches per epoch.
    """

    def __init__(self, model: nn.Module, config: SignalRecorderConfig):
        self.model = model
        self.config = config
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._active_batch: bool = False
        self._current_epoch: Optional[int] = None
        self._current_stage: Optional[str] = None
        self._batch_cache: Optional[Dict] = None
        self._epochs: List[Dict] = []
        self._register_hooks()

    # ------------------------------------------------------------------ #
    # Lifecycle helpers
    # ------------------------------------------------------------------ #
    def begin_epoch(self, epoch: int, stage: str = "train"):
        self._current_epoch = epoch
        self._current_stage = stage
        self._epoch_batches: List[Dict] = []

    def end_epoch(self):
        if self._current_epoch is None:
            return
        epoch_summary = self._aggregate_epoch()
        epoch_summary["epoch"] = self._current_epoch
        epoch_summary["stage"] = self._current_stage or "train"
        epoch_summary["num_batches"] = len(self._epoch_batches)
        self._epochs.append(epoch_summary)
        self._current_epoch = None
        self._current_stage = None

    def begin_batch(self, batch_idx: int, stage: str = "train") -> bool:
        capture = (stage == "train") and (batch_idx < self.config.max_batches_per_epoch)
        self._active_batch = capture
        if capture:
            self._batch_cache = {
                "activations": defaultdict(list),
                "attentions": defaultdict(list),
                "hidden_state_norms": defaultdict(list),
                "gradient_norms": defaultdict(list),
                "batch_idx": batch_idx,
            }
        else:
            self._batch_cache = None
        return capture

    def end_batch(self):
        if not self._active_batch or self._batch_cache is None:
            self._active_batch = False
            self._batch_cache = None
            return
        batch_summary = {
            "batch_idx": self._batch_cache.get("batch_idx"),
            "activations": {k: self._merge_stats(v) for k, v in self._batch_cache["activations"].items()},
            "attentions": {k: self._merge_attention(v) for k, v in self._batch_cache["attentions"].items()},
            "hidden_state_norms": {k: self._merge_scalar_list(v) for k, v in self._batch_cache["hidden_state_norms"].items()},
            "gradient_norms": {k: self._merge_scalar_list(v) for k, v in self._batch_cache["gradient_norms"].items()},
        }
        self._epoch_batches.append(batch_summary)
        self._active_batch = False
        self._batch_cache = None

    def after_backward(self):
        if not self._active_batch or self._batch_cache is None:
            return
        grad_store = self._batch_cache["gradient_norms"]
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            key = _group_key(name)
            grad_store[key].append(float(param.grad.detach().norm().item()))

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def to_dict(self) -> Dict:
        data = {
            "config": asdict(self.config),
            "epochs": self._epochs,
        }
        if isinstance(data["config"]["save_dir"], Path):
            data["config"]["save_dir"] = str(data["config"]["save_dir"])
        return data

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)

    # ------------------------------------------------------------------ #
    # Hooks & aggregation
    # ------------------------------------------------------------------ #
    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if self._is_activation_module(module):
                self._handles.append(module.register_forward_hook(self._make_activation_hook(name)))
            if ATTENTION_TYPES and isinstance(module, ATTENTION_TYPES):
                self._handles.append(module.register_forward_hook(self._make_attention_hook(name)))

    def _is_activation_module(self, module: nn.Module) -> bool:
        return isinstance(module, (nn.GRU, nn.Linear, nn.LayerNorm))

    def _make_activation_hook(self, name: str):
        def hook(_, __, output):
            if not self._active_batch or self._batch_cache is None:
                return
            tensors = []
            if isinstance(output, tuple):
                tensors = [t for t in output if torch.is_tensor(t)]
            elif torch.is_tensor(output):
                tensors = [output]
            else:
                return
            for tensor in tensors:
                stats = _tensor_summary(tensor, self.config.quantiles)
                if not stats:
                    continue
                self._batch_cache["activations"][name].append(stats)
                norms = float(tensor.detach().norm().item())
                self._batch_cache["hidden_state_norms"][name].append(norms)
        return hook

    def _make_attention_hook(self, name: str):
        def hook(_, __, output):
            if not self._active_batch or self._batch_cache is None:
                return
            if isinstance(output, tuple) and len(output) >= 2 and torch.is_tensor(output[1]):
                attn = output[1].detach().to(torch.float32)
                attn_stats = {
                    "mean": float(attn.mean().item()),
                    "std": float(attn.std(unbiased=False).item()),
                    "entropy": float(self._attention_entropy(attn).item()),
                    "lag_curve": attn.mean(dim=0).cpu().tolist(),
                }
                self._batch_cache["attentions"][name].append(attn_stats)
        return hook

    @staticmethod
    def _attention_entropy(attn: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        p = torch.clamp(attn, eps, 1.0)
        return (-p * p.log()).sum(dim=1).mean()

    @staticmethod
    def _merge_stats(stats: List[Dict]) -> Dict[str, float]:
        if not stats:
            return {}
        keys = set().union(*(s.keys() for s in stats))
        merged = {}
        for key in keys:
            values = [s[key] for s in stats if key in s]
            if not values:
                continue
            sample = values[0]
            if isinstance(sample, list):
                max_len = max(len(v) for v in values)
                padded = []
                for v in values:
                    if len(v) < max_len:
                        v = v + [0.0] * (max_len - len(v))
                    padded.append(v)
                merged[key] = [float(sum(items) / len(items)) for items in zip(*padded)]
            else:
                merged[key] = float(sum(values) / len(values))
        return merged

    @staticmethod
    def _merge_attention(stats: List[Dict]) -> Dict:
        if not stats:
            return {}
        merged = {}
        for key in ["mean", "std", "entropy"]:
            values = [s[key] for s in stats if key in s]
            if values:
                merged[key] = float(sum(values) / len(values))
        curves = [s["lag_curve"] for s in stats if "lag_curve" in s]
        if curves:
            max_len = max(len(c) for c in curves)
            padded = []
            for curve in curves:
                if len(curve) < max_len:
                    curve = curve + [0.0] * (max_len - len(curve))
                padded.append(curve)
            merged["lag_curve"] = [float(sum(vals) / len(vals)) for vals in zip(*padded)]
        return merged

    @staticmethod
    def _merge_scalar_list(values: List[float]) -> Dict[str, float]:
        if not values:
            return {}
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        return {
            "mean": float(mean),
            "std": float(var ** 0.5),
            "min": float(min(values)),
            "max": float(max(values)),
        }

    def _aggregate_epoch(self) -> Dict:
        if not getattr(self, "_epoch_batches", None):
            return {"batches": []}
        grad = defaultdict(list)
        activ = defaultdict(list)
        attn = defaultdict(list)
        hidden = defaultdict(list)
        for batch in self._epoch_batches:
            for k, v in batch["gradient_norms"].items():
                grad[k].append(v["mean"])
            for k, v in batch["activations"].items():
                activ[k].append(v)
            for k, v in batch["attentions"].items():
                attn[k].append(v)
            for k, v in batch["hidden_state_norms"].items():
                hidden[k].append(v["mean"])
        summary = {
            "batches": self._epoch_batches,
            "gradient_norms": {k: self._merge_scalar_list(v) for k, v in grad.items()},
            "activation_stats": {k: self._merge_stats(v) for k, v in activ.items()},
            "attention_stats": {k: self._merge_stats(v) for k, v in attn.items()},
            "hidden_norms": {k: self._merge_scalar_list(v) for k, v in hidden.items()},
        }
        return summary
