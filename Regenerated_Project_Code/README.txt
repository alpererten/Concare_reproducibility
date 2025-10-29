ConCare Training & Evaluation
=============================

This repository implements a modular PyTorch pipeline for training and evaluating the ConCare model,
with reproducibility and modern hardware support (Python 3.10 + PyTorch 2.5.1 + CUDA 12.4).

Usage
-----
Run training from the project root:

    python train.py --epochs 100 --batch_size 256 --amp

Features
--------
- Detects CUDA and prints GPU name
- Prints start and end timestamps
- Trains on data/train/ and validates on data/val/
- Evaluates on the test set automatically
- Saves:
    - Best model: trained_models/best_concare.pt
    - Training log: results/train_log.json
    - Test metrics: results/test_metrics.json
    - ROC/PR plots: results/roc_curve.png, results/pr_curve.png

AMP (Automatic Mixed Precision)
-------------------------------
AMP speeds up training and reduces GPU memory use by mixing FP16 and FP32 arithmetic.

Enabled with the --amp flag:
    - 40–60% faster training on modern GPUs (e.g. RTX 4090)
    - Uses torch.cuda.amp.autocast() and GradScaler
    - Minor ±0.01 variations in metrics (non-deterministic)

Recommendation:
    - Use --amp for speed
    - Omit for strict reproducibility

Reproducibility
---------------
Training uses the ConCare paper's original seed (42) with deterministic settings:

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

This ensures stable results on the same hardware. Minor differences may occur when using AMP
or on different GPUs.

Environment Tested
------------------
- Python: 3.10
- PyTorch: 2.5.1
- Torchvision: 0.20.1
- Torchaudio: 2.5.1
- CUDA: 12.4
- GPU: NVIDIA RTX 4090

Folder Structure
----------------
project_root/
├── data/
│   ├── train/
│   ├── val/
│   ├── test/
│   ├── demographic/
│   ├── *_listfile.csv
│   └── ihm_normalizer
│
├── model_codes/ConCare_Model_v1.py
├── helpers.py
├── metrics.py
├── data_preprocessing.py
├── train.py
│
├── results/ (metrics, logs, plots)
└── trained_models/ (best model)

File Roles
-----------
data_preprocessing.py - Handles dataset loading, normalization, and batching.
train.py - Full training/validation/test pipeline with timestamped output.
metrics.py - Computes AUROC, AUPRC, accuracy, precision, recall, F1.
helpers.py - Manages seeding (set_seed(42)) and JSON utilities.

Quick Recap
-----------
1. Ensure data folders are structured as above
2. Run:
       python train.py --epochs 100 --batch_size 256 --amp
3. Check results in 'results/' folder.
