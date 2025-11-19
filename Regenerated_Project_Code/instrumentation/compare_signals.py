#!/usr/bin/env python
"""
Compare instrumentation artifacts between a parent (baseline) and child run,
then update the child's experiment log YAML with a textual summary.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Dict

import yaml


def load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Instrumentation file not found: {path}")
    with path.open("r") as f:
        return json.load(f)


def _first_epoch(payload: Dict) -> Dict:
    epochs = payload.get("signal_check", {}).get("epochs", [])
    if not epochs:
        return {}
    return epochs[0]


def _summarize_gradients(parent_epoch: Dict, child_epoch: Dict) -> str:
    parent_grad = parent_epoch.get("gradient_norms", {})
    child_grad = child_epoch.get("gradient_norms", {})
    if not child_grad:
        return "No gradient statistics captured."

    lines = []
    for layer, stats in sorted(child_grad.items()):
        child_mean = stats.get("mean")
        parent_mean = parent_grad.get(layer, {}).get("mean")
        if child_mean is None:
            continue
        delta = None if parent_mean is None else child_mean - parent_mean
        if delta is None:
            lines.append(f"{layer}: mean={child_mean:.4f}")
        else:
            lines.append(f"{layer}: mean={child_mean:.4f} (Δ {delta:+.4f})")
    return "; ".join(lines)


def _summarize_attention(parent_epoch: Dict, child_epoch: Dict) -> str:
    parent_attn = parent_epoch.get("attention_stats", {})
    child_attn = child_epoch.get("attention_stats", {})
    if not child_attn:
        return "No attention statistics captured."

    lines = []
    for name, stats in sorted(child_attn.items()):
        child_entropy = stats.get("entropy")
        parent_entropy = parent_attn.get(name, {}).get("entropy")
        child_mean = stats.get("mean")
        parent_mean = parent_attn.get(name, {}).get("mean")
        parts = []
        if child_entropy is not None:
            if parent_entropy is None:
                parts.append(f"entropy={child_entropy:.3f}")
            else:
                parts.append(f"entropy={child_entropy:.3f} (Δ {child_entropy - parent_entropy:+.3f})")
        if child_mean is not None:
            if parent_mean is None:
                parts.append(f"mean={child_mean:.3f}")
            else:
                parts.append(f"mean={child_mean:.3f} (Δ {child_mean - parent_mean:+.3f})")
        if parts:
            lines.append(f"{name}: " + ", ".join(parts))
    return "; ".join(lines)


def _summarize_curves(parent_payload: Dict, child_payload: Dict) -> str:
    parent_curve = parent_payload.get("perturbations", {}).get("missingness_curve", [])
    child_curve = child_payload.get("perturbations", {}).get("missingness_curve", [])
    if not child_curve:
        return "Missingness curve not captured."

    def as_map(curve):
        return {round(entry.get("mask_rate", -1), 3): entry.get("metrics", {}) for entry in curve}

    parent_map = as_map(parent_curve)
    child_map = as_map(child_curve)
    lines = []
    for rate in sorted(child_map.keys()):
        child_metrics = child_map[rate]
        parent_metrics = parent_map.get(rate, {})
        c_auroc = child_metrics.get("auroc")
        p_auroc = parent_metrics.get("auroc")
        c_auprc = child_metrics.get("auprc")
        p_auprc = parent_metrics.get("auprc")
        parts = []
        if c_auroc is not None:
            if p_auroc is None:
                parts.append(f"AUROC={c_auroc:.3f}")
            else:
                parts.append(f"AUROC={c_auroc:.3f} (Δ {c_auroc - p_auroc:+.3f})")
        if c_auprc is not None:
            if p_auprc is None:
                parts.append(f"AUPRC={c_auprc:.3f}")
            else:
                parts.append(f"AUPRC={c_auprc:.3f} (Δ {c_auprc - p_auprc:+.3f})")
        rate_str = f"mask_rate={rate:.2f}: " + ", ".join(parts)
        lines.append(rate_str)
    return "; ".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare instrumentation artifacts and update experiment log.")
    parser.add_argument("--parent_json", required=True, help="Path to parent instrumentation JSON.")
    parser.add_argument("--child_json", required=True, help="Path to child instrumentation JSON.")
    parser.add_argument("--child_yaml", required=True, help="Experiment log YAML to update.")
    args = parser.parse_args()

    parent_payload = load_json(Path(args.parent_json))
    child_payload = load_json(Path(args.child_json))

    parent_epoch = _first_epoch(parent_payload)
    child_epoch = _first_epoch(child_payload)

    gradient_summary = _summarize_gradients(parent_epoch, child_epoch)
    attention_summary = _summarize_attention(parent_epoch, child_epoch)
    curve_summary = _summarize_curves(parent_payload, child_payload)

    signal_summary = textwrap.dedent(f"""
        Gradient norms vs parent: {gradient_summary}
        Attention stats vs parent: {attention_summary}
    """).strip()

    perturbation_summary = textwrap.dedent(f"""
        Missingness curve comparison: {curve_summary}
    """).strip()

    yaml_path = Path(args.child_yaml)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Experiment log not found: {yaml_path}")

    with yaml_path.open("r") as f:
        data = yaml.safe_load(f)

    instrumentation = data.setdefault("instrumentation", {})
    signal_block = instrumentation.setdefault("signal_check", {})
    perturb_block = instrumentation.setdefault("perturbations", {})

    signal_block["summary"] = signal_summary
    perturb_block["notes"] = perturbation_summary

    with yaml_path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print("Updated instrumentation summaries in", yaml_path)


if __name__ == "__main__":
    main()
