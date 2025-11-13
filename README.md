# ConCare Reproducibility

Reproduction and ablations of the ConCare model (AAAI 2020) on the MIMIC-III dataset.

## Getting Started

1. Create the environment described in `Regenerated_Project_Code/README_QuickStart.md` (Python 3.10+, PyTorch 2.5+, CUDA 12.x recommended).
2. From the repo root, materialize the RAM caches so every experiment reads the same normalized tensors:
   ```bash
   cd Regenerated_Project_Code
   python materialize_ram.py --timestep 0.8 --append_masks
   ```
3. Train/evaluate via `python train.py ...` inside `Regenerated_Project_Code/`. Logs live in `results/` and checkpoints in `trained_models/`.

## Reproducing ConCareMC− (w/o DeCov)

This branch already contains the ConCareMC− architecture in `model_codes/ConCare_MC_minus.py` and wires it into `train.py`. To match the paper’s ablation that keeps only the health-status embedding of visits (no DeCov, no feature-level attention):

```bash
cd Regenerated_Project_Code
python train.py \
  --model_variant concare_mc_minus \
  --epochs 100 \
  --batch_size 256 \
  --lr 1e-3 \
  --append_masks \
  --amp \
  --papers_metrics_mode
```

Key notes:
- `--model_variant concare_mc_minus` loads the visit-level encoder + demographic fusion and forces `lambda_decov=0`, faithfully reproducing ConCareMC− (a.k.a. ConCareMC- w/o DeCov).
- Add `--diag` for a quick tensor sanity check before training.
- Provide `--cache_dir` or `--parity_mode ...` flags if you need alternative data scopes or the authors’ preprocessing pipeline.

Refer to `Regenerated_Project_Code/README_QuickStart.md` for additional flags, metrics descriptions, and troubleshooting tips.
