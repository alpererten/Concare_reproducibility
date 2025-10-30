"""
ConCare Trainer with detailed diagnostics

This keeps your training flow intact and adds an optional --diag preflight.
No behavior changes unless --diag is provided.
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
from metrics import binary_metrics

from data_preprocessing import ConcareEpisodeDataset, Normalizer, pad_collate


def build_argparser():
    p = argparse.ArgumentParser("ConCare Trainer")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--lambda_decov", type=float, default=1e-3)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir", type=str, default="trained_models")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--normalizer_path", type=str, default="data/ihm_normalizer")
    p.add_argument("--timestep", type=float, default=0.8)
    p.add_argument("--diag", action="store_true", help="Run preflight diagnostics before training")
    return p


def make_datasets(args):
    normalizer = Normalizer(args.normalizer_path)
    expected_features = normalizer.feature_count if normalizer.feature_count else 76
    print(f"[INFO] Normalizer feature width: {expected_features}")

    train_ds = ConcareEpisodeDataset("train", timestep=args.timestep, normalizer=normalizer, expected_features=expected_features)
    val_ds   = ConcareEpisodeDataset("val",   timestep=args.timestep, normalizer=normalizer, expected_features=expected_features)
    test_ds  = ConcareEpisodeDataset("test",  timestep=args.timestep, normalizer=normalizer, expected_features=expected_features)
    return train_ds, val_ds, test_ds, int(expected_features), normalizer


def make_model(input_dim, device):
    from model_codes.ConCare_Model_v1 import ConCare
    print(f"[INFO] Creating model with input_dim={input_dim}")
    model = ConCare(
        input_dim=input_dim,
        hidden_dim=64,
        d_model=64,
        MHD_num_head=4,
        d_ff=128,
        output_dim=1,
        keep_prob=0.5,
        demographic_dim=12,
    ).to(device)
    return model


def tensor_stats(name, t):
    t_cpu = t.detach().float().cpu()
    finite = torch.isfinite(t_cpu)
    pct_finite = finite.float().mean().item() * 100.0
    msg = f"{name}: shape={tuple(t_cpu.shape)} finite={pct_finite:.2f}%"
    if pct_finite > 0:
        msg += f" min={t_cpu[finite].min().item():.4f} max={t_cpu[finite].max().item():.4f} mean={t_cpu[finite].mean().item():.4f}"
    print("[DIAG]", msg)


def diag_preflight(train_ds, normalizer, device):
    print("\n[DIAG] ===== Preflight diagnostics =====")

    # Normalizer stats
    if normalizer.means is not None and normalizer.stds is not None:
        m = normalizer.means
        s = normalizer.stds
        m = np.asarray(m, dtype=np.float32)
        s = np.asarray(s, dtype=np.float32)
        nF = len(m)
        zero_std = int(np.sum(~np.isfinite(s) | (s <= 0)))
        nan_std = int(np.sum(~np.isfinite(s)))
        print(f"[DIAG] Normalizer length={nF} zero_or_nonfinite_stds={zero_std} nonfinite_std_count={nan_std}")
        print(f"[DIAG] Means finite ratio={(np.isfinite(m).mean()*100):.2f}% Stds finite ratio={(np.isfinite(s).mean()*100):.2f}%")
        print(f"[DIAG] Means sample: {m[:6].round(4)} ... Stds sample: {s[:6].round(4)} ...")
    else:
        print("[DIAG] Normalizer has no loaded stats")

    # Single item inspection
    X0, D0, y0 = train_ds[0]
    print(f"[DIAG] First item shapes -> X:{tuple(X0.shape)} D:{tuple(D0.shape)} y:{tuple(y0.shape)}")

    # One batch inspection
    loader = DataLoader(train_ds, batch_size=8, shuffle=False, collate_fn=pad_collate, num_workers=0, pin_memory=True)
    Xb, Db, yb = next(iter(loader))
    tensor_stats("X batch", Xb)
    tensor_stats("D batch", Db)
    tensor_stats("y batch", yb)

    # Forward only to check model outputs
    model = make_model(Xb.shape[-1], device)
    model.eval()
    with torch.no_grad():
        Xb = Xb.to(device)
        Db = Db.to(device)
        yb = yb.to(device)
        logits, decov = model(Xb, Db)
        tensor_stats("logits", logits)
        tensor_stats("decov", decov)
        try:
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, yb)
            print(f"[DIAG] BCE on first batch: {float(bce):.6f}")
        except Exception as e:
            print(f"[DIAG] BCE could not be computed. Reason: {e}")

    print("[DIAG] ===== End preflight =====\n")


def train_one_epoch(model, loader, optimizer, scaler, device, lambda_decov, epoch=0):
    model.train()
    total_loss = 0.0
    probs, labels = [], []

    for batch_idx, (X, D, y) in enumerate(loader):
        # Runtime guards for inputs
        if not torch.isfinite(X).all():
            print(f"[WARN] Non finite X at epoch {epoch} batch {batch_idx}")
        if not torch.isfinite(D).all():
            print(f"[WARN] Non finite D at epoch {epoch} batch {batch_idx}")
        if not torch.isfinite(y).all():
            print(f"[WARN] Non finite y at epoch {epoch} batch {batch_idx}")

        X, D, y = X.to(device, non_blocking=True), D.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler:
            with amp.autocast("cuda", dtype=torch.float16):
                logits, decov = model(X, D)
                bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
                loss = bce + lambda_decov * decov
            if not torch.isfinite(loss):
                print(f"[WARN] NaN or Inf loss at epoch {epoch} batch {batch_idx}. Skipping batch.")
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                print(f"[WARN] Non finite grad norm at epoch {epoch} batch {batch_idx}")
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, decov = model(X, D)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            loss = bce + lambda_decov * decov
            if not torch.isfinite(loss):
                print(f"[WARN] NaN or Inf loss at epoch {epoch} batch {batch_idx}. Skipping batch.")
                continue
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                print(f"[WARN] Non finite grad norm at epoch {epoch} batch {batch_idx}")
            optimizer.step()

        total_loss += loss.item() * X.size(0)
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())

        if batch_idx == 0:
            # Extra detail for the very first batch
            with torch.no_grad():
                print(f"[DIAG] Epoch {epoch} batch {batch_idx} decov={float(decov):.6f} bce={float(bce):.6f}")

    if not probs:
        print("[ERROR] No valid batches in epoch")
        return {"loss": float("inf"), "auprc": 0.0, "auroc": 0.5}

    y_true = np.concatenate(labels).ravel()
    y_prob = np.concatenate(probs).ravel()
    m = binary_metrics(y_true, y_prob)
    m["loss"] = total_loss / len(loader.dataset)
    return m


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, labels = [], []
    total_loss = 0.0

    for X, D, y in loader:
        X, D, y = X.to(device, non_blocking=True), D.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits, decov = model(X, D)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        if torch.isfinite(loss):
            total_loss += loss.item() * X.size(0)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(y.cpu().numpy())

    if not probs:
        return {"loss": float("inf"), "auprc": 0.0, "auroc": 0.5}, np.array([]), np.array([])

    y_true = np.concatenate(labels).ravel()
    y_prob = np.concatenate(probs).ravel()
    m = binary_metrics(y_true, y_prob)
    m["loss"] = total_loss / len(loader.dataset)
    return m, y_true, y_prob


def _save_curves(y_true, y_prob, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    try:
        from sklearn.metrics import roc_curve, precision_recall_curve, auc
        import matplotlib.pyplot as plt

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(rec, prec)

        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "roc_curve.png"))
        plt.close()

        plt.figure()
        plt.plot(rec, prec, label=f"PR AUC = {pr_auc:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision Recall Curve")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "pr_curve.png"))
        plt.close()
    except Exception as e:
        print(f"[WARN] Could not save curves. Reason: {e}")


def main():
    args = build_argparser().parse_args()
    set_seed(42)

    if torch.cuda.is_available() and "cuda" in args.device:
        print(f"✅ Using CUDA on device: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CUDA not available. Using CPU")

    print("\n[INFO] Loading datasets")
    train_ds, val_ds, test_ds, expected_features, normalizer = make_datasets(args)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate, num_workers=0, pin_memory=True)

    # Discover input dimension from a real item and assert consistency
    X0, D0, _ = train_ds[0]
    input_dim = X0.shape[1]
    assert input_dim == expected_features, f"X width {input_dim} does not match expected {expected_features}"

    if args.diag:
        diag_preflight(train_ds, normalizer, args.device)

    model = make_model(input_dim, args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.5, weight_decay=args.weight_decay)
    scaler = amp.GradScaler("cuda") if args.amp and "cuda" in args.device else None

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, "best_concare.pt")
    best_auprc = -1.0
    log = []

    print(f"\n🚀 Starting training at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Input dimension: {input_dim}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr * 0.5}")
    print(f"   Using AMP: {args.amp}\n")

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, scaler, args.device, args.lambda_decov, epoch)
        va, _, _ = evaluate(model, val_loader, args.device)
        log.append({"epoch": epoch, "train": tr, "val": va})

        print(
            f"Epoch {epoch:03d} | "
            f"Train loss {tr['loss']:.4f} AUPRC {tr['auprc']:.4f} AUROC {tr['auroc']:.4f} | "
            f"Val loss {va['loss']:.4f} AUPRC {va['auprc']:.4f} AUROC {va['auroc']:.4f}"
        )

        if va["auprc"] > best_auprc:
            best_auprc = va["auprc"]
            torch.save({"model_state": model.state_dict(), "best_auprc": best_auprc}, best_path)

    print(f"\n🏁 Training finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Final evaluation on test split
    print("📊 Evaluating best checkpoint on TEST set")
    checkpoint = torch.load(best_path, map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics, y_true_test, y_prob_test = evaluate(model, test_loader, args.device)

    print("\n📊 Test Set Results")
    for k, v in test_metrics.items():
        print(f"{k:>8}: {v:.4f}")

    if len(y_true_test) > 0:
        _save_curves(y_true_test, y_prob_test, args.results_dir)

    print("\n✅ Training completed successfully")


if __name__ == "__main__":
    main()
