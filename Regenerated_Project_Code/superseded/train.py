import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  ## Ensures deterministic behavior for CuBLAS (avoids "non-deterministic CuBLAS" warnings on CUDA >=10.2)


import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import amp
from datetime import datetime

from helpers import set_seed, save_json
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
    return p


def make_datasets():
    normalizer = Normalizer("data/ihm_normalizer")
    train_ds = ConcareEpisodeDataset("train", normalizer=normalizer)
    val_ds = ConcareEpisodeDataset("val", normalizer=normalizer)
    test_ds = ConcareEpisodeDataset("test", normalizer=normalizer)
    return train_ds, val_ds, test_ds, normalizer


def make_model(input_dim):
    from model_codes.ConCare_Model_v1 import ConCare
    return ConCare(
        input_dim=input_dim, hidden_dim=64, d_model=64,
        MHD_num_head=4, d_ff=128, output_dim=1,
        keep_prob=0.5, demographic_dim=12
    )


def train_one_epoch(model, loader, optimizer, scaler, device, lambda_decov):
    model.train()
    total_loss, all_probs, all_labels = 0.0, [], []
    for X, D, y in loader:
        X, D, y = X.to(device), D.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler:
            with amp.autocast('cuda', dtype=torch.float16):
                logits, decov = model(X, D)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y) + lambda_decov * decov
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, decov = model(X, D)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y) + lambda_decov * decov
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * X.size(0)
        y_prob = torch.sigmoid(logits)
        all_probs.append(y_prob.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())

    y_true = np.concatenate(all_labels).ravel()
    y_prob = np.concatenate(all_probs).ravel()
    metrics = binary_metrics(y_true, y_prob)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels, total_loss = [], [], 0.0
    for X, D, y in loader:
        X, D, y = X.to(device), D.to(device), y.to(device)
        logits, decov = model(X, D)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        total_loss += loss.item() * X.size(0)
        y_prob = torch.sigmoid(logits)
        all_probs.append(y_prob.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    y_true = np.concatenate(all_labels).ravel()
    y_prob = np.concatenate(all_probs).ravel()
    metrics = binary_metrics(y_true, y_prob)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics, y_true, y_prob


def _save_curves(y_true, y_prob, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    try:
        from sklearn.metrics import roc_curve, precision_recall_curve, auc
        import matplotlib.pyplot as plt

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(rec, prec)

        # ROC
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "roc_curve.png"))
        plt.close()

        # PR
        plt.figure()
        plt.plot(rec, prec, label=f"PR AUC = {pr_auc:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision–Recall Curve")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "pr_curve.png"))
        plt.close()

    except Exception:
        thresholds = np.linspace(0.0, 1.0, 200)
        prec, rec, tpr, fpr = [], [], [], []
        for th in thresholds:
            y_hat = (y_prob >= th).astype(int)
            tp = ((y_hat == 1) & (y_true == 1)).sum()
            fp = ((y_hat == 1) & (y_true == 0)).sum()
            fn = ((y_hat == 0) & (y_true == 1)).sum()
            tn = ((y_hat == 0) & (y_true == 0)).sum()
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            fpr.append(fp / max(fp + tn, 1))
            tpr.append(tp / max(tp + fn, 1))
            prec.append(p); rec.append(r)
        np.save(os.path.join(results_dir, "roc_fpr.npy"), np.array(fpr))
        np.save(os.path.join(results_dir, "roc_tpr.npy"), np.array(tpr))
        np.save(os.path.join(results_dir, "pr_prec.npy"), np.array(prec))
        np.save(os.path.join(results_dir, "pr_rec.npy"), np.array(rec))


def main():
    args = build_argparser().parse_args()
    set_seed(42)

    if torch.cuda.is_available() and "cuda" in args.device:
        print(f"✅ Using CUDA on device: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CUDA not available — using CPU.")

    train_ds, val_ds, test_ds, normalizer = make_datasets()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate)

    model = make_model(train_ds[0][0].shape[1]).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = amp.GradScaler('cuda') if args.amp and "cuda" in args.device else None

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    best_auprc, best_path, log = -1.0, os.path.join(args.save_dir, "best_concare.pt"), []

    print(f"\n🚀 Starting training at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, scaler, args.device, args.lambda_decov)
        va, _, _ = evaluate(model, val_loader, args.device)
        log.append({"epoch": epoch, "train": tr, "val": va})
        print(
            f"Epoch {epoch:03d} | "
            f"Train: loss {tr['loss']:.4f} AUPRC {tr['auprc']:.4f} AUROC {tr['auroc']:.4f} | "
            f"Val:   loss {va['loss']:.4f} AUPRC {va['auprc']:.4f} AUROC {va['auroc']:.4f}"
        )

        if va["auprc"] > best_auprc:
            best_auprc = va["auprc"]
            torch.save({"model_state": model.state_dict(), "best_auprc": best_auprc}, best_path)

    print(f"\n🏁 Training finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    save_json({"log": log, "best_auprc": best_auprc, "best_path": best_path},
              os.path.join(args.results_dir, "train_log.json"))

    print("🔎 Evaluating best checkpoint on TEST set...")
    checkpoint = torch.load(best_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics, y_true_test, y_prob_test = evaluate(model, test_loader, args.device)
    print("\n📊 Test Set Results:")
    for k, v in test_metrics.items():
        print(f"{k:>8}: {v:.4f}")

    save_json(test_metrics, os.path.join(args.results_dir, "test_metrics.json"))
    _save_curves(y_true_test, y_prob_test, args.results_dir)


if __name__ == "__main__":
    main()
