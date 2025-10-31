"""
ConCare Trainer — RAM-only: materialize once -> train from memory
Loads normalized NPZs from data/normalized_data_cache/

PHASE 3 UPDATES:
- Batch size: 32 → 256
- Learning rate: 5e-4 → 1e-3
- Feed-forward dim: 128 → 256
- Epochs: 50 → 100 (default)
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import amp
from datetime import datetime
from helpers import set_seed

# ---- Metrics selection ----
from metrics import binary_metrics as ours_binary_metrics
try:
    from metrics import minpse_report as ours_minpse
except Exception:
    ours_minpse = None

try:
    from metrics_authors import binary_metrics as authors_binary_metrics
except Exception:
    authors_binary_metrics = None

# ---- Fixed cache locations inside the project ----
CACHE_DIR = os.path.join("data", "normalized_data_cache")
NORM_STATS = os.path.join(CACHE_DIR, "np_norm_stats.npz")


def build_argparser():
    p = argparse.ArgumentParser("ConCare Trainer (RAM-only)")
    # PHASE 3: Updated defaults to match authors
    p.add_argument("--epochs", type=int, default=100)  # Changed from 50
    p.add_argument("--batch_size", type=int, default=256)  # Changed from 32
    p.add_argument("--lr", type=float, default=1e-3)  # Changed from 1e-3 (was halved in old code)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--lambda_decov", type=float, default=1e-4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir", type=str, default="trained_models")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--timestep", type=float, default=1.0)  # PHASE 2: Changed from 0.8 to 1.0
    p.add_argument("--append_masks", action="store_true", default=True,  # PHASE 2: Default True
                   help="Append binary masks to values in discretizer (2F features).")
    p.add_argument("--diag", action="store_true", help="Run preflight diagnostics before training")
    p.add_argument("--compile", action="store_true", help="Enable torch.compile if Triton is available")
    p.add_argument("--papers_metrics_mode", action="store_true",
                   help="Use the authors' metric implementation for AUROC and AUPRC")
    return p


def _ensure_materialized(timestep: float, append_masks: bool):
    """
    Ensure normalized NPZs + stats exist in data/normalized_data_cache/.
    If not present, call materializer which writes to that directory.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    need = (
        not os.path.exists(os.path.join(CACHE_DIR, "train.npz")) or
        not os.path.exists(os.path.join(CACHE_DIR, "val.npz")) or
        not os.path.exists(os.path.join(CACHE_DIR, "test.npz")) or
        not os.path.exists(NORM_STATS)
    )
    if need:
        print("[INFO] Materialized cache not found. Running materialize_ram.py...")
        from materialize_ram import materialize_split
        for sp in ["train", "val", "test"]:
            materialize_split(sp, timestep=timestep, append_masks=append_masks, impute_strategy='previous')
    else:
        print(f"[INFO] Using existing normalized cache in {CACHE_DIR}")


def _choose_workers():
    cpu_cnt = os.cpu_count() or 4
    workers = min(8, max(2, cpu_cnt // 2))
    return workers, workers > 0


def make_model(input_dim, device, use_compile=False):
    """
    PHASE 1 & PHASE 3: Load the multi-channel GRU model with updated hyperparameters.
    """
    try:
        from model_codes.ConCare_Model_v1 import ConCare
    except ModuleNotFoundError:
        from ConCare_Model_v1 import ConCare
    
    print(f"[INFO] Creating ConCare model with input_dim={input_dim}")
    model = ConCare(
        input_dim=input_dim,
        hidden_dim=64,  # Same as authors
        d_model=64,  # Same as authors
        MHD_num_head=4,  # Same as authors
        d_ff=256,  # PHASE 3: Changed from 128 to 256
        output_dim=1,
        keep_prob=0.5,  # Same as authors
        demographic_dim=12,
    ).to(device)
    
    if use_compile:
        try:
            import triton  # noqa: F401
        except Exception:
            print("[WARN] Triton is not available. Skipping torch.compile()")
        else:
            try:
                model = torch.compile(model, dynamic=True)
                print("[INFO] torch.compile enabled")
            except Exception as e:
                print(f"[WARN] torch.compile disabled: {e}")
    return model


def tensor_stats(name, t):
    t_cpu = t.detach().float().cpu()
    finite = torch.isfinite(t_cpu)
    pct_finite = finite.float().mean().item() * 100.0
    msg = f"{name}: shape={tuple(t_cpu.shape)} finite={pct_finite:.2f}%"
    if pct_finite > 0:
        msg += f" min={t_cpu[finite].min().item():.4f} max={t_cpu[finite].max().item():.4f} mean={t_cpu[finite].mean().item():.4f}"
    print("[DIAG]", msg)


def diag_preflight(train_ds, device, collate_fn):
    print("\n[DIAG] ===== Preflight diagnostics =====")
    X0, D0, y0 = train_ds[0]
    print(f"[DIAG] First item shapes -> X:{tuple(X0.shape)} D:{tuple(D0.shape)} y:{tuple(y0.shape)}")
    loader = DataLoader(train_ds, batch_size=8, shuffle=False, collate_fn=collate_fn,
                        num_workers=0, pin_memory=True)
    Xb, Db, yb = next(iter(loader))
    tensor_stats("X batch", Xb); tensor_stats("D batch", Db); tensor_stats("y batch", yb)
    model = make_model(Xb.shape[-1], device, use_compile=False)
    model.eval()
    with torch.no_grad():
        Xb, Db, yb = Xb.to(device), Db.to(device), yb.to(device)
        logits, decov = model(Xb, Db)
        tensor_stats("logits", logits); tensor_stats("decov", decov)
        try:
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb)
            print(f"[DIAG] BCE on first batch: {float(bce):.6f}")
        except Exception as e:
            print(f"[DIAG] BCE could not be computed. Reason: {e}")
    print("[DIAG] ===== End preflight =====\n")


def _sanitize_decov(decov_tensor: torch.Tensor):
    """Replace NaN or Inf and clamp to prevent numerical blow ups."""
    d = torch.nan_to_num(decov_tensor, nan=0.0, posinf=1e6, neginf=-1e6)
    return d.clamp_max(1e4)


def best_threshold_from_probs(y_true, y_prob):
    """Grid search thresholds to maximize F1 for the positive class."""
    y_true = y_true.astype(np.float32)
    best = (0.5, 0.0, 0.0, 0.0)
    for t in np.linspace(0.01, 0.99, 99):
        y_pred = (y_prob >= t).astype(np.int32)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1 = 0.0 if (prec + rec) == 0.0 else 2 * prec * rec / (prec + rec)
        if f1 > best[1]:
            best = (float(t), float(f1), float(prec), float(rec))
    return best


def minpse_from_pr(precision: float, recall: float) -> float:
    """Compute MINPSE at a specific operating point if we only know P and R."""
    return min(precision, recall)


# Choose which metric function to use
def select_metric_fn(papers_mode: bool):
    if papers_mode:
        if authors_binary_metrics is None:
            print("[WARN] papers_metrics_mode is requested but metrics_authors.py is not available. Falling back to local metrics")
            return ours_binary_metrics, "local"
        return authors_binary_metrics, "authors"
    return ours_binary_metrics, "local"


def train_one_epoch(model, loader, optimizer, scaler, device, lambda_decov, epoch, criterion, metric_fn):
    model.train()
    total_loss = 0.0
    probs, labels = [], []
    for batch_idx, (X, D, y) in enumerate(loader):
        X, D, y = X.to(device, non_blocking=True), D.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler:
            with amp.autocast("cuda"):
                logits, decov = model(X, D)
                decov = _sanitize_decov(decov)
                bce_loss = criterion(logits, y)
                loss = bce_loss + lambda_decov * decov
            scaler.scale(loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, decov = model(X, D)
            decov = _sanitize_decov(decov)
            bce_loss = criterion(logits, y)
            loss = bce_loss + lambda_decov * decov
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item() * X.size(0)
        probs.append(torch.sigmoid(logits).to(torch.float32).cpu().numpy())
        labels.append(y.to(torch.float32).cpu().numpy())
    
    y_true = np.concatenate(labels).ravel()
    y_prob = np.concatenate(probs).ravel()
    m = metric_fn(y_true, y_prob)
    m["loss"] = total_loss / len(loader.dataset)
    return m


@torch.no_grad()
def evaluate(model, loader, device, criterion, metric_fn):
    model.eval()
    probs, labels = [], []
    total_loss = 0.0
    for X, D, y in loader:
        X, D, y = X.to(device, non_blocking=True), D.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits, decov = model(X, D)
        decov = _sanitize_decov(decov)
        loss = criterion(logits, y)
        total_loss += loss.item() * X.size(0)
        probs.append(torch.sigmoid(logits).to(torch.float32).cpu().numpy())
        labels.append(y.to(torch.float32).cpu().numpy())
    y_true = np.concatenate(labels).ravel()
    y_prob = np.concatenate(probs).ravel()
    m = metric_fn(y_true, y_prob)
    m["loss"] = total_loss / len(loader.dataset)
    return m, y_true, y_prob


def print_thresholded_report(yt_true, yt_prob, thr, header="📊 Test"):
    yhat = (yt_prob >= thr).astype(np.int32)
    tp = int(((yhat == 1) & (yt_true == 1)).sum())
    fp = int(((yhat == 1) & (yt_true == 0)).sum())
    tn = int(((yhat == 0) & (yt_true == 0)).sum())
    fn = int(((yhat == 0) & (yt_true == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 0.0 if (prec + rec) == 0.0 else 2 * prec * rec / (prec + rec)
    acc  = (tp + tn) / max(tp + tn + fp + fn, 1)
    print(f"\n{header} @thr={thr:.2f}  acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")
    return acc, prec, rec, f1


def main():
    args = build_argparser().parse_args()
    set_seed(42)

    metric_fn, metric_source = select_metric_fn(args.papers_metrics_mode)

    if torch.cuda.is_available() and "cuda" in args.device:
        print(f"✅ Using CUDA on device: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CUDA not available. Using CPU")

    print("\n" + "="*60)
    print("PHASE 1, 2, 3 UPDATES ACTIVE")
    print("="*60)
    print("Phase 1: Multi-channel GRU architecture")
    print("Phase 2: timestep=1.0, masks=True, impute='previous'")
    print("Phase 3: batch=256, lr=1e-3, d_ff=256, epochs=100")
    print("="*60 + "\n")

    print("\n[INFO] Preparing RAM datasets")
    _ensure_materialized(args.timestep, args.append_masks)

    from ram_dataset import RAMDataset, pad_collate as ram_pad_collate
    train_ds = RAMDataset("train", cache_dir=CACHE_DIR)
    val_ds   = RAMDataset("val",   cache_dir=CACHE_DIR)
    test_ds  = RAMDataset("test",  cache_dir=CACHE_DIR)

    X0, _, _ = train_ds[0]; input_dim = X0.shape[1]
    print(f"[INFO] Detected input_dim={input_dim} (should be 2*F if masks appended)")

    workers, use_workers = _choose_workers()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=ram_pad_collate,
                              num_workers=workers, pin_memory=True, persistent_workers=use_workers,
                              prefetch_factor=2 if use_workers else None)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=ram_pad_collate,
                              num_workers=workers, pin_memory=True, persistent_workers=use_workers,
                              prefetch_factor=2 if use_workers else None)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, collate_fn=ram_pad_collate,
                              num_workers=workers, pin_memory=True, persistent_workers=use_workers,
                              prefetch_factor=2 if use_workers else None)

    if args.diag:
        diag_preflight(train_ds, args.device, ram_pad_collate)

    model = make_model(input_dim, args.device, use_compile=args.compile)

    # ---- Class-weighted BCE ----
    tmp_loader = DataLoader(train_ds, batch_size=1024, shuffle=False, collate_fn=ram_pad_collate,
                            num_workers=0, pin_memory=True)
    train_labels = []
    for _, _, y in tmp_loader:
        train_labels.append(y.cpu().numpy())
    train_labels = np.concatenate(train_labels).ravel()
    n_pos = float((train_labels > 0.5).sum())
    n_neg = float((train_labels <= 0.5).sum())
    pos_weight_value = n_neg / max(n_pos, 1.0)
    print(f"[INFO] Class counts (train): pos={int(n_pos)} neg={int(n_neg)} pos_weight={pos_weight_value:.2f}")
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], device=args.device, dtype=torch.float32)
    )

    # PHASE 3: Full learning rate (no halving)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = amp.GradScaler("cuda") if args.amp and "cuda" in args.device else None

    os.makedirs(args.save_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, "best_concare.pt")
    best_auprc = -1.0

    # decov warmup
    target_lambda = args.lambda_decov
    warmup_epochs = 10

    print(f"\n🚀 Starting training at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Input dimension: {input_dim}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Feed-forward dim: 256")
    print(f"   Using AMP: {args.amp}")
    print(f"   Using torch.compile: {args.compile}")
    print(f"   Metrics source: {metric_source}{' (papers_metrics_mode ON)' if args.papers_metrics_mode else ''}\n")

    for epoch in range(1, args.epochs + 1):
        lambda_decov = target_lambda * (epoch / warmup_epochs) if epoch <= warmup_epochs else target_lambda

        tr = train_one_epoch(model, train_loader, optimizer, scaler, args.device, lambda_decov, epoch, criterion, metric_fn)
        va, yv_true, yv_prob = evaluate(model, val_loader, args.device, criterion, metric_fn)

        print(f"Epoch {epoch:03d} | Train loss {tr['loss']:.4f} AUPRC {tr['auprc']:.4f} AUROC {tr['auroc']:.4f} | "
              f"Val loss {va['loss']:.4f} AUPRC {va['auprc']:.4f} AUROC {va['auroc']:.4f}")

        # choose threshold that maximizes F1 on current val
        thr, f1, p, r = best_threshold_from_probs(yv_true, yv_prob)
        print(f"          Val@thr={thr:.2f}  F1={f1:.4f}  P={p:.4f}  R={r:.4f}")

        # track best by AUPRC
        if va["auprc"] > best_auprc:
            best_auprc = va["auprc"]
            torch.save({"model": model.state_dict(), "epoch": epoch}, best_path)

    print(f"\n📊 Evaluating best checkpoint on TEST set")
    ckpt = torch.load(best_path, map_location=args.device) if os.path.exists(best_path) else None
    if ckpt:
        model.load_state_dict(ckpt["model"])

    test_metrics, yt_true, yt_prob = evaluate(model, test_loader, args.device, criterion, metric_fn)

    # Enrich with MINPSE summary around the best precision and recall balance if available
    if ours_minpse is not None:
        try:
            mp = ours_minpse(yt_true, yt_prob)
            test_metrics = dict(test_metrics)
            test_metrics.update({
                "minpse": mp.get("minpse", float("nan")),
                "minpse_thr": mp.get("best_thr", float("nan")),
                "minpse_prec": mp.get("precision", float("nan")),
                "minpse_rec": mp.get("recall", float("nan")),
            })
        except Exception:
            pass

    print("\n📊 Test Set Results (threshold-free)")
    for k in sorted(test_metrics.keys()):
        v = test_metrics[k]
        try:
            print(f"{k:>8}: {float(v):.4f}")
        except Exception:
            print(f"{k:>8}: {v}")

    # Use best validation threshold for the final "authors-style" operating point
    va_full, yv_true_full, yv_prob_full = evaluate(model, val_loader, args.device, criterion, metric_fn)
    best_thr, _, _, _ = best_threshold_from_probs(yv_true_full, yv_prob_full)

    acc_best, prec_best, rec_best, f1_best = print_thresholded_report(
        yt_true, yt_prob, best_thr, header="📊 Test @best_val_thr"
    )
    minpse_best = minpse_from_pr(prec_best, rec_best)

    print("\n📊 Test Set Results (authors-style)")
    print(f"     acc: {acc_best:.4f}")
    print(f"      f1: {f1_best:.4f}")
    print(f"   auroc: {test_metrics.get('auroc', float('nan')):.4f}")
    print(f"   auprc: {test_metrics.get('auprc', float('nan')):.4f}")
    print(f"  minpse: {minpse_best:.4f}")
    print(f"   thr_used: {best_thr:.2f}")

    print("\n✅ Training completed successfully")


if __name__ == "__main__":
    main()
