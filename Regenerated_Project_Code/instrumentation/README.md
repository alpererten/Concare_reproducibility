# Instrumentation Guide

The `--instrument_signals` flag wires ConCare runs into a lightweight instrumentation
pipeline that collects gradient norms, activations, attention statistics, and basic
robustness curves (missingness/truncation sweeps). This document explains how to enable
the feature, where the data lands, and how to interpret the results.


## Enabling instrumentation

Add the following CLI options to any `train.py` launch:

```bash
python train.py \
  --instrument_signals \
  --instrumentation_tag 2025-11-16_maskbias_off_A3 \
  --instrumentation_dir instrumentation_data \
  --instrumentation_probe_batches 2 \
  --instrumentation_mask_rates 0.0,0.2,0.4 \
  --instrumentation_truncation_keep_ratios 1.0,0.75,0.5
```

Key knobs:

| Flag | Purpose |
| ---- | ------- |
| `--instrument_signals` | Turns on hooks that record gradients/activations/attention stats. |
| `--instrumentation_tag` | Prefix for the JSON artifact file (`instrumentation_data/<tag>.json`). Use the same tag in your experiment log. |
| `--instrumentation_probe_batches` | How many *training* batches per epoch capture detailed stats (default 2). |
| `--instrumentation_mask_rates` / `--instrumentation_truncation_keep_ratios` | Mask-rate and keep-ratio grids for robustness curves. |
| `--instrumentation_eval_batches` | Number of validation batches used for each perturbation point (default 4). |
| `--instrumentation_dir` | Output folder for JSON artifacts (default `instrumentation_data/`). |

The JSON artifact is automatically referenced in the experiment log’s `instrumentation`
block so reviewers know where to find the raw data.


## JSON layout

Every run produces `instrumentation_data/<tag>.json` with the following sections:

```json
{
  "config": {... CLI options ...},
  "convergence": {"best_epoch": 24, "best_val_auprc": 0.5324},
  "signal_check": {
    "epochs": [
      {
        "epoch": 1,
        "gradient_norms": {"linear": {"mean": 0.42, ...}, ...},
        "activation_stats": {"GRUs.0": {"mean": 0.03, "std": 0.11, ...}},
        "attention_stats": {
          "MissingAwareTemporalAttention": {
            "mean": 0.12,
            "entropy": 1.45,
            "lag_curve": [0.31, 0.28, ...]
          }
        },
        "hidden_norms": {...},
        "batches": [...]
      },
      ...
    ]
  },
  "perturbations": {
    "missingness_curve": [
      {"mask_rate": 0.0, "metrics": {"auroc": 0.847, "auprc": 0.466}},
      {"mask_rate": 0.4, "metrics": {...}}
    ],
    "truncation_curve": [
      {"keep_ratio": 1.0, "metrics": {...}},
      {"keep_ratio": 0.5, "metrics": {...}}
    ]
  }
}
```


## How to analyze the signals

### Gradient norms

Look for systematic changes in `gradient_norms` across epochs:

- **Nearly identical norms** compared to the baseline indicate the new feature is
  optimizer-inert.
- **Large spikes or vanishing norms** in certain layers suggest the feature is
  altering the loss landscape (good or bad).

### Activation statistics

Inspect `activation_stats` and `hidden_norms` (mean/std/quantiles):

- **Saturated activations** (means drifting toward the extremes, tiny std) usually
  hurt expressiveness — expect recall drops or unstable training.
- **Higher variance** in early layers with stable downstream stats can indicate the
  model is exploring richer latent spaces (often positive).

### Attention summaries

`attention_stats` include `mean`, `entropy`, and a `lag_curve` (average attention mass
per relative lag). Use them to judge whether SMART is reweighting time as intended:

- **Higher entropy and flatter lag_curve** means temporal focus is diffused.
- **Sharp peaks near lag 0** mean the model prefers recent events, which might clash
  with long-horizon tasks.

Compare against the baseline run’s JSON to determine if the ablation is shifting the
temporal behavior meaningfully.

### Perturbation curves

`missingness_curve` and `truncation_curve` report AUROC/AUPRC for synthetic shifts:

- Plot the curves for baseline vs. new run; if the new model’s metric drops more slowly
  as mask rate increases, it’s genuinely "missing-aware".
- If the curves overlap, the feature isn’t buying robustness (possible regression).

Document the key takeaways in the experiment log’s `instrumentation` notes so reviewers
don’t have to dig through JSON manually.

### Automating comparisons

Use `instrumentation/compare_signals.py` to compare two instrumentation artifacts and
write the summary directly into the child experiment log:

```bash
python instrumentation/compare_signals.py \
  --parent_json instrumentation_data/baseline_a0_instrumented.json \
  --child_json instrumentation_data/2025-11-16_missing_aware_maskbias_off_A3.json \
  --child_yaml experiment_logs/missing-aware/2025-11-16_missing_aware_maskbias_off_A3.yaml
```

The script compares gradient means, attention entropies, and overlapping missingness
curves, then updates `instrumentation.signal_check.summary` and
`instrumentation.perturbations.notes` inside the child YAML.
