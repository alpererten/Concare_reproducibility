# 🧠 ConCare Reproducibility – Quick Start Guide

This project reproduces and extends the **ConCare (AAAI 2020)** model for clinical risk prediction on the MIMIC-III dataset.  
It includes modernized training, caching, and evaluation pipelines compatible with CUDA GPUs and RAM-based datasets.

---

## ⚙️ 1. Environment Setup

Create and activate your environment (example using Conda):

```bash
conda create -n concare_env python=3.10
conda activate concare_env
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install numpy pandas scikit-learn tqdm matplotlib
```

**Requirements:**
- Python 3.10+
- PyTorch 2.5+ with CUDA 12.x (recommended)
- NVIDIA GPU with at least 8 GB VRAM (16 GB+ preferred)

---

## 📂 2. Repository Structure

```
Regenerated_Project_Code/
│
├── train.py                  # Main training & evaluation loop
├── materialize_ram.py        # Builds normalized RAM-based datasets
├── ram_dataset.py            # Dataset loader for cached .npz data
├── metrics.py                # Modern local metrics (AUPRC, AUROC, etc.)
├── metrics_authors.py        # Authors’ original metrics for parity check
├── model_codes/
│   ├── ConCare_Model_v3.py   # Full ConCare (multi-channel + DeCov)
│   ├── ConCare_MC_minus.py   # ConCareMC- ablation (visit-level, no DeCov)
│   └── ConCare_DE_minus.py   # ConCareDE- ablation (full ConCare, no demographics)
├── experiment_logs/          # YAML logger for structural experiments & ablations
├── data/
│   ├── normalized_data_cache/  # Cached train/val/test .npz files
│   └── demographic/            # Optional demographic CSVs per patient
└── results/
    └── train_val_test_log_*.txt  # Auto-saved logs with timestamp
```

---

## 🧩 3. Data Preparation

Before training, you must have preprocessed and discretized MIMIC-III data following the ConCare pipeline.  
Once ready, **materialize the RAM datasets**:

```bash
python materialize_ram.py --timestep 0.8
```

This creates normalized `.npz` files under `data/normalized_data_cache/`:
- `train.npz`, `val.npz`, `test.npz`
- `np_norm_stats.npz` (feature scaling)
- Optional demographic caches (`D_train.npz`, etc.)

---

## 🚀 4. Training the Model

Run standard training (RAM-based, AMP enabled):

```bash
python train.py --epochs 100 --batch_size 256 --lr 1e-3 --append_masks --amp --papers_metrics_mode
```

This will:
- Train using cached data for faster I/O  
- Save the **best model** to `trained_models/best_concare.pt`  
- Log **both local and authors’ metrics** (AUROC, AUPRC, MinPSE, F1, etc.)  
- Save results in:
  ```
  results/train_val_test_log_<timestamp>.txt
  results/test_results_<timestamp>.txt
  ```

### 🔬 ConCareMC- (w/o DeCov) Ablation

To match the paper's *ConCareMC-* study (visit-level embedding only, no decorrelation loss), pass:

```bash
python train.py --model_variant concare_mc_minus --epochs 100 --batch_size 256 --lr 1e-3 --append_masks --amp
```

The trainer automatically disables the DeCov loss (`lambda_decov → 0`) for this variant, so no extra flags are required.

---

## 📊 5. Understanding Metrics

Two evaluation modes are supported:

| Metric Source | Description |
|----------------|--------------|
| **Local (default)** | Uses modern `scikit-learn` metrics for AUPRC, AUROC, F1 |
| **Authors’ (AAAI 2020)** | Replicates ConCare’s original `print_metrics_binary` for strict parity |

Enable authors’ metrics with:
```bash
--papers_metrics_mode
```

At the end of training, both metric sets are printed and logged.

---

## 🧾 6. Example Logs

Each run produces timestamped result files under `results/`:

```
train_val_test_log_2025-10-31_22-15-12.txt
test_results_2025-10-31_22-15-12.txt
```

Each file includes:
- Epoch-wise training & validation losses
- AUPRC, AUROC, F1 trends
- Authors’ metrics (AUROC, AUPRC, MinPSE, F1)
- Final test performance summary

---

## 🧰 7. Optional Flags

| Flag | Description |
|------|--------------|
| `--amp` | Enables mixed-precision training |
| `--append_masks` | Adds time-series mask features |
| `--lambda_decov` | Sets decorrelation loss weight (default = 1e-3) |
| `--model_variant` | `concare_full`, `concare_mc_minus`, or `concare_de_minus` |
| `--num_workers` | Override DataLoader workers (`-1` auto, `0` serial) |
| `--weight_decay` | Adds Adam weight decay |
| `--compile` | Enables `torch.compile` (PyTorch 2.0+) |
| `--diag` | Runs dataset and model diagnostics |
| `--papers_metrics_mode` | Enables authors’ metrics printing |
| `--missing_aware_extension` | Turns on the SMART-style missing-aware attention path (see below) |
| `--lr_scheduler` | Optional LR decay during fine-tuning (`none`, `cosine`) |
| `--keep_prob` | Dropout keep probability (default 0.5; lower value = more dropout) |
| `--missing_aware_disable_mask_bias` | In SMART mode, skip mask-biased feature pooling (acts like vanilla ConCare) |
| `--missing_aware_disable_temporal_attention` | In SMART mode, reuse ConCare’s original per-feature attention |

### 🧠 SMART-style Missing-Aware Extension (Optional)

Enable ConCare’s SMART-inspired path (mask-aware temporal attention + latent reconstruction) with `--missing_aware_extension`.  
Two-stage training mirrors the paper:

1. **Latent pre-training** (mask/reconstruct health contexts with EMA teacher)
2. **Fine-tuning** (freeze encoder for a few epochs, then unfreeze)

During pre-training we randomly remove observed measurements (probability sampled from the provided min/max range), feed the masked tensor through the student encoder, and minimize an L1 reconstruction loss against an EMA “teacher” encoder that sees the pristine sequence. No labels are used in this phase. Fine-tuning swaps in the standard BCE head; optionally keep the encoder frozen for a few epochs so only the classifier learns before unlocking the full network.

This mirrors the masked-token idea from language models: the ConCare encoder (multi-channel GRUs + cross-feature attention) is treated as the “language encoder,” and latent pre-training teaches it how to infer missing clinical tokens before we ever see labels. Once fine-tuned, the decoder maps those richer contexts to mortality risk, which is why it get more confident predictions even when vitals are sparse.

Key flags:

| Flag | Meaning |
|------|---------|
| `--missing_aware_pretrain_epochs` | Number of reconstruction epochs before classification (default 0 = skip) |
| `--missing_aware_mask_ratio_min/max` | Range for random removal probability during pre-training |
| `--missing_aware_pretrain_lr` | Optional LR just for the pre-training stage (falls back to `--lr`) |
| `--missing_aware_ema_decay` | EMA decay for the teacher encoder |
| `--missing_aware_freeze_epochs` | How long to freeze the encoder when fine-tuning (decoder-only training) |
| `--missing_aware_aux_weight` | Weight of latent reconstruction auxiliary loss during fine-tuning (0 to disable) |
| `--missing_aware_disable_mask_bias` | Disable mask-biased feature attention if you want a pure ConCare ablation |
| `--missing_aware_disable_temporal_attention` | Disable the mask-aware per-feature temporal attention (fallback to original ConCare attention) |

Example command (25 pre-train epochs, freeze encoder for first 5 fine-tune epochs):

```bash
python train.py \
  --epochs 100 --batch_size 256 \
  --model_variant concare_full \
  --missing_aware_extension \
  --missing_aware_pretrain_epochs 10 \
  --missing_aware_freeze_epochs 5 \
  --missing_aware_mask_ratio_min 0.1 \
  --missing_aware_mask_ratio_max 0.4 \
  --missing_aware_unfreeze_lr 2e-4 \
  --lr 5e-4 \
  --append_masks --amp  --missing_aware_pretrain_lr 2e-4 \
  --early_stop_patience 10 --early_stop_min_delta 0.001 --lr_scheduler cosine \
  --missing_aware_aux_weight 0.05  --weight_decay 1e-4
```

Baseline ConCare runs are unaffected unless `--missing_aware_extension` is explicitly supplied.

Reference: Zhihao Yu et al., “SMART: Towards Pre-trained Missing-Aware Model for Patient Health Status Prediction” (NeurIPS 2024).

---

## 🔁 8. Repeated Cross-Validation & Early Stopping

Need to match the paper’s *mean ± std* reporting? The trainer now supports repeated stratified K-fold CV plus configurable early stopping.

| Flag | Description |
|------|-------------|
| `--cv_folds` | Number of folds (>1 enables CV mode) |
| `--cv_repeats` | How many random reshuffles to run |
| `--cv_pool_splits` | Cached splits to pool before folding (default `train,val`) |
| `--cv_val_ratio` | Portion of each training fold carved out for validation |
| `--cv_seed` | Seed for split reproducibility |
| `--early_stop_patience` | Epochs to wait for val AUPRC improvement (0 disables) |
| `--early_stop_min_delta` | Minimum gain in AUPRC to reset patience |

Example (ConCareMC−, AMP, 3×5 repeated CV, patience 15):

```bash
python train.py \
  --model_variant concare_mc_minus \
  --epochs 100 --batch_size 256 --lr 1e-3 \
  --append_masks --amp --papers_metrics_mode \
  --cv_folds 10 --cv_repeats 3 --cv_val_ratio 0.1 \
  --cv_pool_splits train,val \
  --early_stop_patience 15 --early_stop_min_delta 0.002 --device cuda
```

### 📊 Reproducing Paper Ablations

| Variant | Flag | Difference vs. full ConCare | Notes |
|---------|------|-----------------------------|-------|
| ConCareMC− | `--model_variant concare_mc_minus` | Removes multi-channel feature encoders and DeCov (single GRU over visits) | Matches paper's Ablation 2 |
| ConCareDE− | `--model_variant concare_de_minus` | Keeps multi-channel + DeCov but drops the demographic feature channel | Matches paper's Ablation 3 |

Sample command for ConCareDE− (paper ablation 3):

```bash
python train.py \
  --model_variant concare_de_minus \
  --epochs 100 --batch_size 256 --lr 1e-3 \
  --append_masks --amp --papers_metrics_mode \
  --device cuda --lambda_decov 1e-3
```

Each fold writes its own `train_val_test_log_<timestamp>_repX_foldY.txt`, and a consolidated `cv_summary_<timestamp>.txt` captures mean/std across folds plus authors-style statistics. For a single-run “early stop only” experiment, keep `--cv_folds 0` (default) but set the patience/min-delta knobs.

---

## 📈 9. Reproducing Authors’ Metrics Only

To compute the authors’ metrics on saved predictions:

```python
from metrics_authors import print_metrics_binary

# y_true: ground truth labels (0/1)
# y_prob: predicted probabilities for the positive class
print_metrics_binary(y_true, y_prob, verbose=1)
```

This will return all metrics including **AUC of ROC**, **AUC of PRC**, **Min(+P, Se)**, and **F1** exactly as in the paper.

---

## ✅ 10. Typical Workflow Summary

1. Prepare preprocessed MIMIC-III data  
2. Generate cached datasets:
   ```bash
   python materialize_ram.py --timestep 0.8
   ```
3. Train and evaluate:
   ```bash
   python train.py --epochs 100 --batch_size 256 --lr 1e-3 --append_masks --amp --papers_metrics_mode
   ```
4. Review logs in `results/`  

---

## 🗂️ 11. Structural Experiment Logger

- Every ablation or major training change should be captured as a YAML file under `experiment_logs/`.
- Copy `experiment_logs/template.yaml`, fill in the identifier, what changed, belief, measured metrics, effect sizes vs. the parent run, and the final decision.
- See `experiment_logs/README.md` plus the pre-filled AAAI 2020 entries (baseline + Ablations 2/3) for concrete examples.
- Attach these files (or excerpts) to PRs so reviewers can quickly understand experiment intent and outcomes.
5. Compare authors’ vs local metrics for parity validation  

---

### 🧪 Notes

- **Decorrelator (DeCov)**: The `--lambda_decov` loss term is optional. Setting it to 0 matches the authors’ reported configuration. The `concare_mc_minus` model variant forces this term to zero automatically.
- **Demographics**: Ensure your `RAMDataset` includes demographic vectors (age, gender, etc.) for full replication.
- **Hardware**: Training on RTX 4090 / A100 / V100 yields epoch times ≈ 1–2 minutes.

---

**Maintainer:** *Alper Erten*  
**Institution:** Georgia Institute of Technology  
**Project Goal:** Reproducible and extensible ConCare replication for MIMIC-III.  
**License:** Research-only use.  
