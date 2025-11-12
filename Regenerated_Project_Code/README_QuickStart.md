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
│   └── ConCare_MC_minus.py   # ConCareMC- ablation (visit-level, no DeCov)
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
| `--model_variant` | `concare_full` (default) or `concare_mc_minus` ablation |
| `--weight_decay` | Adds Adam weight decay |
| `--compile` | Enables `torch.compile` (PyTorch 2.0+) |
| `--diag` | Runs dataset and model diagnostics |
| `--papers_metrics_mode` | Enables authors’ metrics printing |

---

## 📈 8. Reproducing Authors’ Metrics Only

To compute the authors’ metrics on saved predictions:

```python
from metrics_authors import print_metrics_binary

# y_true: ground truth labels (0/1)
# y_prob: predicted probabilities for the positive class
print_metrics_binary(y_true, y_prob, verbose=1)
```

This will return all metrics including **AUC of ROC**, **AUC of PRC**, **Min(+P, Se)**, and **F1** exactly as in the paper.

---

## ✅ 9. Typical Workflow Summary

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
