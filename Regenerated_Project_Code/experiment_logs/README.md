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
task: "in-hospital-mortality"
change: "enable_pre_residual_layernorm: true"  # concise description of what changed
belief:
  expected_final_gain: "small_positive"         # or neutral/negative
  expected_stability_gain: "moderate"
  expected_convergence_speed_gain: "small_positive"
config:
  command: "python train.py ..."
  args:
    epochs: 100
    batch_size: 256
    ...
metrics:
  mimic_iii:
    val:
      best_epoch: 27
      auroc: 0.8702
      auprc: 0.5317
      f1: 0.5143
      precision: 0.4952
      recall: 0.5376
      threshold: 0.42
    test_threshold_free:
      auroc: 0.8729
      auprc: 0.5211
      loss: 0.3321
    test_authors:
      acc: 0.8569
      auroc: 0.8729
      auprc: 0.5211
      minpse: 0.5092
      f1: 0.4911
      threshold: 0.42
    test_fixed_thr:
      threshold: 0.66
      acc: 0.8711
      f1: 0.4763
      minpse: 0.5222
    source: "Regenerated_Project_Code/results/train_val_test_log_2025-11-11_22-50-41.txt"
effect_size:
  mimic_iii:
    vs_parent:
      info:
        run: "Regenerated_Project_Code/results/train_val_test_log_2025-11-08_18-12-00.txt"
        SEL: "Regenerated_Project_Code/experiment_logs/exp_2025_11_08_baseline.yaml"
      delta:
        val:
          auroc: 0.0021
          auprc: -0.0034
          ...
      pct:
        val:
          auroc: 0.0024
          auprc: -0.0063
          ...
    vs_siblings:
      sibling:
        info:
          run: null
          SEL: null
        delta:
          ...
        pct:
          ...
decision: "keep"  # or reject / needs_followup
notes: >
  Extra free-form context, including qualitative observations, issues, or next steps.
```

### Effect-size expectations

- Always fill in `effect_size.mimic_iii.vs_parent` so reviewers can see exact deltas against the parent experiment.
  - `info.run` should be the raw training log (e.g., `Regenerated_Project_Code/results/train_val_test_log_2025-11-16_14-19-58 A0.txt`).
  - `info.SEL` should point to the parent structural experiment log (e.g., `Regenerated_Project_Code/experiment_logs/missing-aware/2025-11-16_baseline_concare_ram.yaml`).
  - Populate the `delta` block with absolute differences (child − parent) for every metric you reported.
  - Populate the matching `pct` block with percentage change: `(child − parent) / parent`.
- `effect_size.mimic_iii.vs_siblings` compares against the strongest sibling experiment under the same parent.
  Leave the entire section as `null`/`None` placeholders until at least one other sibling log exists, then
  update it with the sibling’s log references and deltas.

## Workflow

1. After each controlled experiment (ablations, hyper-parameter sweeps, architecture tweaks), copy `template.yaml` to a new file name like `exp_2025_11_15_concare_de_minus.yaml`.
2. Fill in the schema above. All numeric metrics should point to their log source so reviewers can trace them.
3. Compute effect sizes versus the declared parent (and sibling best, when available) before marking the log complete.
4. Summarize the decision (`keep`, `reject`, `follow_up`) and the rationale in `notes`.
5. Send structured logs along with PRs so reviewers can quickly understand what changed and how it affected metrics.

## Existing entries

- `exp_2019_aaai_baseline_concare.yaml`: Baseline ConCare metrics reproduced from the AAAI 2020 paper.
- `exp_2019_aaai_ablation2_concare_mc_minus.yaml`: ConCareMC− ablation (no multi-channel encoders + no DeCov).
- `exp_2019_aaai_ablation3_concare_de_minus.yaml`: ConCareDE− ablation (removes demographics, keeps multi-channel encoders).

Add new experiment files alongside these to keep the log growing.
