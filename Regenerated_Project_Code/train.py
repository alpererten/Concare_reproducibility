"""
ConCare Trainer — RAM-only: materialize once -> train from memory
Loads normalized NPZs from data/normalized_data_cache/
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

# ---- Fixed cache locations inside the project ----
CACHE_DIR = os.path.join("data", "normalized_data_cache")
NORM_STATS = os.path.join(CACHE_DIR, "np_norm_stats.npz")


def build_argparser():
    p = argparse.ArgumentParser("ConCare Trainer (RAM-only)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--lambda_decov", type=float, default=1e-3)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir", type=str, default="trained_models")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--timestep", type=float, default=0.8)
    p.add_argument("--append_masks", action="store_true",
                   help="Append binary masks to values in discretizer (2F features).")
    p.add_argument("--diag", action="store_true", help="Run preflight diagnostics before training")
    # torch.compile is opt-in (Windows often lacks Triton)
    p.add_argument("--compile", action="store_true", help="Enable torch.compile if Triton is available")
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
        from materialize_ram import materialize_split
        for sp in ["train", "val", "test"]:
            materialize_split(sp, timestep=timestep, append_masks=append_masks)
    else:
        print(f"[INFO] Using existing normalized cache in {CACHE_DIR}")


def _choose_workers():
    cpu_cnt = os.cpu_count() or 4
    workers = min(8, max(2, cpu_cnt // 2))
    return workers, workers > 0


def make_model(input_dim, device, use_compile=False):
    try:
        from model_codes.ConCare_Model_v1 import ConCare
    except ModuleNotFoundError:
        from ConCare_Model_v1 import ConCare
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


def train_one_epoch(model, loader, optimizer, scaler, device, lambda_decov, epoch=0):
    model.train()
    total_loss = 0.0
    probs, labels = [], []
    for batch_idx, (X, D, y) in enumerate(loader):
        X, D, y = X.to(device, non_blocking=True), D.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler:
            with amp.autocast("cuda", dtype=torch.float16):
                logits, decov = model(X, D)
                bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
                loss = bce + lambda_decov * decov
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer); scaler.update()
        else:
            logits, decov = model(X, D)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            loss = bce + lambda_decov * decov
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total_loss += loss.item() * X.size(0)
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())
        if batch_idx == 0:
            print(f"[DIAG] Epoch {epoch} batch {batch_idx} decov={float(decov):.6f} bce={float(bce):.6f}")
    y_true = np.concatenate(labels).ravel()
    y_prob = np.concatenate(probs).ravel()
    m = binary_metrics(y_true, y_prob); m["loss"] = total_loss / len(loader.dataset)
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
        total_loss += loss.item() * X.size(0)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.cpu().numpy())
    y_true = np.concatenate(labels).ravel()
    y_prob = np.concatenate(probs).ravel()
    m = binary_metrics(y_true, y_prob); m["loss"] = total_loss / len(loader.dataset)
    return m, y_true, y_prob


def main():
    args = build_argparser().parse_args()
    set_seed(42)

    if torch.cuda.is_available() and "cuda" in args.device:
        print(f"✅ Using CUDA on device: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CUDA not available. Using CPU")

    print("\n[INFO] Preparing RAM datasets")
    _ensure_materialized(args.timestep, args.append_masks)

    from ram_dataset import RAMDataset, pad_collate as ram_pad_collate
    # Point RAMDataset at the fixed cache dir
    train_ds = RAMDataset("train", cache_dir=CACHE_DIR)
    val_ds   = RAMDataset("val",   cache_dir=CACHE_DIR)
    test_ds  = RAMDataset("test",  cache_dir=CACHE_DIR)

    X0, _, _ = train_ds[0]; input_dim = X0.shape[1]

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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.5, weight_decay=args.weight_decay)
    scaler = amp.GradScaler("cuda") if args.amp and "cuda" in args.device else None

    os.makedirs(args.save_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, "best_concare.pt")
    best_auprc = -1.0

    print(f"\n🚀 Starting training at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Input dimension: {input_dim}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr * 0.5}")
    print(f"   Using AMP: {args.amp}")
    print(f"   Using torch.compile: {args.compile}\n")

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, scaler, args.device, args.lambda_decov, epoch)
        va, _, _ = evaluate(model, val_loader, args.device)
        print(f"Epoch {epoch:03d} | Train loss {tr['loss']:.4f} AUPRC {tr['auprc']:.4f} AUROC {tr['auroc']:.4f} | "
              f"Val loss {va['loss']:.4f} AUPRC {va['auprc']:.4f} AUROC {va['auroc']:.4f}")

    print(f"\n📊 Evaluating best checkpoint on TEST set")
    test_metrics, _, _ = evaluate(model, test_loader, args.device)
    print("\n📊 Test Set Results")
    for k, v in test_metrics.items():
        print(f"{k:>8}: {v:.4f}")
    print("\n✅ Training completed successfully")


if __name__ == "__main__":
    main()
