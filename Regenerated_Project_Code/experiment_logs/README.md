# Structural Experiment Logger

This folder stores YAML snapshots for every notable architecture or training change.  
Each file is named `exp_<YYYY_MM_DD>_<slug>.yaml` so entries can be diffed and reviewed easily.

## Required fields

```yaml
experiment_id: "2025-11-15_abl_layernorm_01"   # unique identifier
parent_id: "2025-11-10_baseline_03"            # optional: previous experiment this builds on
date: "2025-11-15"
owner: "your_name"
dataset: "MIMIC-III in-hospital mortality"
change: "enable_pre_residual_layernorm: true"  # concise description of what changed
belief:
  expected_final_gain: "small_positive"         # or neutral/negative
  expected_stability_gain: "moderate"
  expected_convergence_speed_gain: "small_positive"
metrics:
  mimic_iii:
    auroc: 0.8702
    auprc: 0.5317
    min_se_p: 0.5082
    source: "train_val_test_log_2025-11-11_22-50-41.txt"
effect_size:
  mimic_iii:
    auprc_delta_vs_parent: -0.0415
decision: "keep"  # or reject / needs_followup
notes: >
  Extra free-form context, including qualitative observations, issues, or next steps.
```

## Workflow

1. After each controlled experiment (ablations, hyper-parameter sweeps, architecture tweaks), copy `template.yaml` to a new file name like `exp_2025_11_15_concare_de_minus.yaml`.
2. Fill in the schema above. All numeric metrics should point to their log source so reviewers can trace them.
3. Summarize the decision (`keep`, `reject`, `follow_up`) and the rationale in `notes`.
4. Send structured logs along with PRs so reviewers can quickly understand what changed and how it affected metrics.

## Existing entries

- `exp_2019_aaai_baseline_concare.yaml`: Baseline ConCare metrics reproduced from the AAAI 2020 paper.
- `exp_2019_aaai_ablation2_concare_mc_minus.yaml`: ConCareMC− ablation (no multi-channel encoders + no DeCov).
- `exp_2019_aaai_ablation3_concare_de_minus.yaml`: ConCareDE− ablation (removes demographics, keeps multi-channel encoders).

Add new experiment files alongside these to keep the log growing.
