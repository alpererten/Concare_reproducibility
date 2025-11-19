from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .signal_recorder import SignalRecorder, SignalRecorderConfig
from .perturbations import run_missingness_curve, run_truncation_curve


def _parse_float_list(raw: str) -> List[float]:
    if not raw:
        return []
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            continue
    return values


@dataclass
class InstrumentationConfig:
    enable_signals: bool = False
    max_batches_per_epoch: int = 2
    mask_rates: List[float] = None  # type: ignore
    truncation_keep_ratios: List[float] = None  # type: ignore
    perturbation_batches: int = 8
    output_dir: Path = Path("instrumentation_data")
    run_id: Optional[str] = None

    @classmethod
    def from_args(cls, args):
        mask_rates = _parse_float_list(getattr(args, "instrumentation_mask_rates", "0.0,0.2,0.4"))
        truncations = _parse_float_list(getattr(args, "instrumentation_truncation_keep_ratios", "1.0,0.75,0.5"))
        run_id = getattr(args, "instrumentation_tag", None) or datetime.now().strftime("%Y%m%d_%H%M%S")
        return cls(
            enable_signals=getattr(args, "instrument_signals", False),
            max_batches_per_epoch=getattr(args, "instrumentation_probe_batches", 2),
            mask_rates=mask_rates,
            truncation_keep_ratios=truncations,
            perturbation_batches=getattr(args, "instrumentation_eval_batches", 4),
            output_dir=Path(getattr(args, "instrumentation_dir", "instrumentation_data")),
            run_id=run_id,
        )


class InstrumentationManager:
    """
    Coordinates per-epoch signal recording and post-training perturbation sweeps
    (missingness curves, truncation sensitivity). Outputs JSON summaries that
    experiment_logs entries can reference.
    """

    def __init__(self, model, args):
        self.config = InstrumentationConfig.from_args(args)
        self.model = model
        self.signal_recorder: Optional[SignalRecorder] = None
        self.results: Dict = {}
        if self.config.enable_signals:
            recorder_cfg = SignalRecorderConfig(
                max_batches_per_epoch=self.config.max_batches_per_epoch,
                save_dir=self.config.output_dir,
                run_id=self.config.run_id,
            )
            self.signal_recorder = SignalRecorder(model, recorder_cfg)
        self.output_path = self.config.output_dir / f"{self.config.run_id}.json"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Training loop hooks
    # ------------------------------------------------------------------ #
    def begin_epoch(self, epoch: int):
        if self.signal_recorder:
            self.signal_recorder.begin_epoch(epoch)

    def end_epoch(self):
        if self.signal_recorder:
            self.signal_recorder.end_epoch()

    def begin_batch(self, batch_idx: int, stage: str = "train"):
        if self.signal_recorder:
            return self.signal_recorder.begin_batch(batch_idx, stage)
        return False

    def end_batch(self):
        if self.signal_recorder:
            self.signal_recorder.end_batch()

    def after_backward(self):
        if self.signal_recorder:
            self.signal_recorder.after_backward()

    # ------------------------------------------------------------------ #
    # Post-training evaluation
    # ------------------------------------------------------------------ #
    def finalize(self, best_epoch: int, best_val_auprc: float,
                 train_args, val_loader, metric_fn):
        """
        Run perturbation sweeps using the best checkpoint.
        """
        config_dict = asdict(self.config)
        config_dict["output_dir"] = str(config_dict["output_dir"])

        payload = {
            "config": config_dict,
            "convergence": {
                "best_epoch": best_epoch,
                "best_val_auprc": best_val_auprc,
            },
            "created_at": datetime.utcnow().isoformat(),
        }
        if self.signal_recorder:
            payload["signal_check"] = self.signal_recorder.to_dict()
            self.signal_recorder.close()

        model = self.model
        device = train_args.device

        perturbation = {}
        if self.config.mask_rates and hasattr(model, "value_dim") and model.mask_dim > 0:
            perturbation["missingness_curve"] = run_missingness_curve(
                model, val_loader, device, metric_fn, self.config.mask_rates,
                self.config.perturbation_batches, model.value_dim, model.mask_dim,
                getattr(model, "value_to_channel", None))
        if self.config.truncation_keep_ratios:
            perturbation["truncation_curve"] = run_truncation_curve(
                model, val_loader, device, metric_fn, self.config.truncation_keep_ratios,
                self.config.perturbation_batches)
        payload["perturbations"] = perturbation

        with self.output_path.open("w") as f:
            json.dump(payload, f, indent=2)
        self.results = payload
        return self.output_path
