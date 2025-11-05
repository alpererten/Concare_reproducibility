"""
ConCare Trainer — RAM-only: materialize once -> train from memory
Loads normalized NPZs from data/normalized_data_cache/
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import glob
import shutil
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import amp
from datetime import datetime
from helpers import set_seed

# ---- Metrics selection ----
# Our local metrics (threshold-free AUROC/AUPRC and extras)
from metrics import binary_metrics as ours_binary_metrics
# Optional MINPSE helper from our metrics, with a safe fallback if absent
try:
    from metrics import minpse_report as ours_minpse
except Exception:
    ours_minpse = None

# Authors' exact metrics for parity (AUROC, AUPRC, MinPSE, F1, etc.)
try:
    from metrics_authors import print_metrics_binary as authors_print_metrics_binary
except ModuleNotFoundError:
    authors_print_metrics_binary = None


# ---- Fixed cache locations inside the project ----
CACHE_DIR = os.path.join("data", "normalized_data_cache")
NORM_STATS = os.path.join(CACHE_DIR, "np_norm_stats.npz")


def build_argparser():
    p = argparse.ArgumentParser("ConCare Trainer (RAM-only)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--lambda_decov", type=float, default=1e-4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir", type=str, default="trained_models")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--cache_dir", type=str, default="data/normalized_data_cache_train",
                   help="Path to normalized NPZ cache (e.g., data/normalized_data_cache_train or _all)")
    p.add_argument("--timestep", type=float, default=0.8)
    p.add_argument("--append_masks", action="store_true",
                   help="Append binary masks to values in discretizer (2F features).")
    p.add_argument("--diag", action="store_true", help="Run preflight diagnostics before training")
    p.add_argument("--compile", action="store_true", help="Enable torch.compile if Triton is available")
    p.add_argument("--papers_metrics_mode", action="store_true",
                   help="Use the authors' metric implementation for AUROC and AUPRC")

    # -------- Parity mode options (authors' exact tensors) --------
    p.add_argument("--parity_mode", action="store_true",
                   help="Build cache using authors' Readers + Discretizer + Normalizer, then train")
    p.add_argument("--parity_train_dir", type=str, default=None,
                   help="Path to authors' TRAIN dataset dir containing timeseries CSVs")
    p.add_argument("--parity_train_listfile", type=str, default=None,
                   help="Path to authors' TRAIN listfile.csv")
    p.add_argument("--parity_val_dir", type=str, default=None,
                   help="Path to authors' VAL dataset dir")
    p.add_argument("--parity_val_listfile", type=str, default=None,
                   help="Path to authors' VAL listfile.csv")
    p.add_argument("--parity_test_dir", type=str, default=None,
                   help="Path to authors' TEST dataset dir")
    p.add_argument("--parity_test_listfile", type=str, default=None,
                   help="Path to authors' TEST listfile.csv")
    # --------------------------------------------------------------

    return p


def _assemble_from_cases(case_dir: Path, out_path: Path):
    """
    Assemble a split NPZ (X ragged, D 12-d rows or zeros, y) from per-stay .npz files
    that contain at least X and y. Demographics are zeros here by design, since authors'
    parity does not supply D.
    """
    files = sorted(case_dir.glob("*.npz"))
    if len(files) == 0:
        raise RuntimeError(f"No per-case NPZ files found in {case_dir}")
    X_list, y_list, D_list = [], [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        X = z["X"].astype(np.float32)
        y = float(z["y"])
        X_list.append(X)
        y_list.append(np.float32(y))
        D_list.append(np.zeros((12,), dtype=np.float32))  # parity does not add demographics
    np.savez_compressed(
        out_path,
        X=np.array(X_list, dtype=object),
        D=np.array(D_list, dtype=np.float32),
        y=np.array(y_list, dtype=np.float32),
    )
    print(f"[PARITY] Assembled {len(y_list)} cases → {out_path}")


def _materialize_authors_parity(args):
    """
    Use authors' pipeline via materialize_parity(), then gather its per-stay NPZs into
    train.npz/val.npz/test.npz that RAMDataset expects. We isolate each split into a
    temporary subfolder to avoid filename collisions across splits.
    """
    from materialize_ram import materialize_parity  # parity builder we added

    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp_train = Path(CACHE_DIR) / "cases_train"
    tmp_val   = Path(CACHE_DIR) / "cases_val"
    tmp_test  = Path(CACHE_DIR) / "cases_test"
    for p in [tmp_train, tmp_val, tmp_test]:
        if p.exists():
            # Clean any old cases for a fresh run
            for f in p.glob("*.npz"):
                try:
                    f.unlink()
                except Exception:
                    pass
        else:
            p.mkdir(parents=True, exist_ok=True)

    # Helper to sweep fresh per-stay NPZs into a split folder
    def _sweep_cases(target_dir: Path):
        fresh = sorted(Path(CACHE_DIR).glob("*.npz"))
        for f in fresh:
            # Skip previously assembled split npz names
            if f.name in ("train.npz", "val.npz", "test.npz", "np_norm_stats.npz"):
                continue
            shutil.move(str(f), str(target_dir))

    # Train split
    written_tr = materialize_parity(
        dataset_dir=args.parity_train_dir,
        listfile=args.parity_train_listfile,
        task="ihm",
        timestep=args.timestep,
        imputation="previous",
        store_masks=True,
        start_time="zero",
        normalizer_state="data/ihm_normalizer",
        small_part=False,
        limit=None,
    )
    print(f"[PARITY] Train wrote {written_tr} cases")
    _sweep_cases(tmp_train)

    # Val split
    written_va = materialize_parity(
        dataset_dir=args.parity_val_dir,
        listfile=args.parity_val_listfile,
        task="ihm",
        timestep=args.timestep,
        imputation="previous",
        store_masks=True,
        start_time="zero",
        normalizer_state="data/ihm_normalizer",
        small_part=False,
        limit=None,
    )
    print(f"[PARITY] Val wrote {written_va} cases")
    _sweep_cases(tmp_val)

    # Test split
    written_te = materialize_parity(
        dataset_dir=args.parity_test_dir,
        listfile=args.parity_test_listfile,
        task="ihm",
        timestep=args.timestep,
        imputation="previous",
        store_masks=True,
        start_time="zero",
        normalizer_state="data/ihm_normalizer",
        small_part=False,
        limit=None,
    )
    print(f"[PARITY] Test wrote {written_te} cases")
    _sweep_cases(tmp_test)

    # Assemble split files RAMDataset expects
    _assemble_from_cases(tmp_train, Path(CACHE_DIR) / "train.npz")
    _assemble_from_cases(tmp_val,   Path(CACHE_DIR) / "val.npz")
    _assemble_from_cases(tmp_test,  Path(CACHE_DIR) / "test.npz")

    # Leave a small marker stats file so _ensure_materialized() passes if called later
    if not os.path.exists(NORM_STATS):
        # Authors' normalizer is a pickle elsewhere. Here we only drop a marker.
        np.savez_compressed(NORM_STATS, means=np.array([0.0], dtype=np.float32), stds=np.array([1.0], dtype=np.float32))


# ---- NEW helper: infer scope name from --cache_dir ----
def _infer_scope_from_cache_dir(path: str) -> str:
    name = os.path.basename(os.path.normpath(path))
    # expected format: normalized_data_cache_{scope}
    if name.startswith("normalized_data_cache_") and len(name) > len("normalized_data_cache_"):
        return name.replace("normalized_data_cache_", "", 1)
    return "train"


def _ensure_materialized_default(timestep: float, append_masks: bool, cache_dir: str):
    """
    Ensure normalized NPZs + stats exist under the selected cache_dir.
    If not present, call our default materializer which writes to that directory/scope.
    """
    os.makedirs(cache_dir, exist_ok=True)
    scope = _infer_scope_from_cache_dir(cache_dir)

    # Accept either plain or scoped stats filenames
    stats_plain = os.path.join(cache_dir, "np_norm_stats.npz")
    stats_scoped = os.path.join(cache_dir, f"np_norm_stats_{scope}.npz")

    need = (
        not os.path.exists(os.path.join(cache_dir, "train.npz")) or
        not os.path.exists(os.path.join(cache_dir, "val.npz")) or
        not os.path.exists(os.path.join(cache_dir, "test.npz")) or
        not (os.path.exists(stats_plain) or os.path.exists(stats_scoped))
    )
    if need:
        from materialize_ram import materialize_split
        for sp in ["train", "val", "test"]:
            materialize_split(
                sp,
                timestep=timestep,
                append_masks=append_masks,
                norm_scope=scope,  # key: write/read in the scope matching --cache_dir
            )
    else:
        print(f"[INFO] Using existing normalized cache in {cache_dir}")


def _ensure_materialized(args):
    """
    Branch between parity mode and default materializer.
    """
    if args.parity_mode:
        required = [
            args.parity_train_dir, args.parity_train_listfile,
            args.parity_val_dir, args.parity_val_listfile,
            args.parity_test_dir, args.parity_test_listfile,
        ]
        if any(x is None for x in required):
            raise ValueError("parity_mode requires all of: "
                             "--parity_train_dir --parity_train_listfile "
                             "--parity_val_dir --parity_val_listfile "
                             "--parity_test_dir --parity_test_listfile")
        print("[INFO] Parity mode ON — building cache with authors' pipeline")
        _materialize_authors_parity(args)
    else:
        _ensure_materialized_default(args.timestep, args.append_masks, args.cache_dir)


def _choose_workers():
    cpu_cnt = os.cpu_count() or 4
    workers = min(8, max(2, cpu_cnt // 2))
    return workers, workers > 0


def make_model(input_dim, device, use_compile=False):
    try:
        from model_codes.ConCare_Model_v3 import ConCare
    except ModuleNotFoundError:
        from ConCare_Model_v3 import ConCare
    print(f"[INFO] Creating model with input_dim={input_dim}")
    model = ConCare(
        input_dim=input_dim,
        hidden_dim=64,
        d_model=64,
        MHD_num_head=4,
        d_ff=256,
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
            bce = torch.nn.functional.binary_cross_entropy(logits, yb)
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
    return 1.0 - 0.5 * (precision + recall)


# Choose which metric function to use (threshold-free set)
def select_metric_fn(papers_mode: bool):
    # Always use local threshold-free metrics for training curves
    if papers_mode:
        if authors_print_metrics_binary is None:
            print("[WARN] papers_metrics_mode requested but metrics_authors.py not found. Using local metrics only.")
        else:
            print("[INFO] papers_metrics_mode ON — authors' print_metrics_binary will be reported alongside local metrics.")
    return ours_binary_metrics, "local"


def train_one_epoch(model, loader, optimizer, scaler, device, lambda_decov, epoch, criterion, metric_fn):
    model.train()
    total_loss = 0.0
    probs, labels = [], []
    for batch_idx, (X, D, y) in enumerate(loader):
        X, D, y = X.to(device, non_blocking=True), D.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler:
            with amp.autocast("cuda", dtype=torch.bfloat16):
                logits, decov = model(X, D)
                decov = _sanitize_decov(decov)
                bce = criterion(logits, y)
                loss = bce + lambda_decov * decov
            if not torch.isfinite(loss):
                for g in optimizer.param_groups:
                    g['lr'] = max(g['lr'] * 0.5, 1e-6)
                if batch_idx % 50 == 0:
                    print(f"[WARN] Non-finite loss at epoch {epoch} batch {batch_idx}; "
                          f"decov={float(decov):.6f} bce={float(bce):.6f}; "
                          f"lowering LR to {optimizer.param_groups[0]['lr']:.2e} and skipping batch")
                scaler.update()
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer); scaler.update()
        else:
            logits, decov = model(X, D)
            decov = _sanitize_decov(decov)
            bce = criterion(logits, y)
            loss = bce + lambda_decov * decov
            if not torch.isfinite(loss):
                for g in optimizer.param_groups:
                    g['lr'] = max(g['lr'] * 0.5, 1e-6)
                if batch_idx % 50 == 0:
                    print(f"[WARN] Non-finite loss at epoch {epoch} batch {batch_idx}; "
                          f"decov={float(decov):.6f} bce={float(bce):.6f}; "
                          f"lowering LR to {optimizer.param_groups[0]['lr']:.2e} and skipping batch")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * X.size(0)
        probs.append(logits.detach().to(torch.float32).cpu().numpy())
        labels.append(y.detach().to(torch.float32).cpu().numpy())
        if batch_idx == 0:
            print(f"[DIAG] Epoch {epoch} batch {batch_idx} decov={float(decov):.6f} bce={float(bce):.6f}")

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
        probs.append(logits.to(torch.float32).cpu().numpy())
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

    # override cache folder via CLI
    global CACHE_DIR, NORM_STATS
    CACHE_DIR = args.cache_dir
    NORM_STATS = os.path.join(CACHE_DIR, "np_norm_stats.npz")

    set_seed(42)

    metric_fn, metric_source = select_metric_fn(args.papers_metrics_mode)

    if torch.cuda.is_available() and "cuda" in args.device:
        print(f"✅ Using CUDA on device: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CUDA not available. Using CPU")

    print("\n[INFO] Preparing RAM datasets")
    _ensure_materialized(args)

    from ram_dataset import RAMDataset, pad_collate as ram_pad_collate
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

    # ---- Unweighted BCE on probabilities ----
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = amp.GradScaler("cuda") if args.amp and "cuda" in args.device else None

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, "best_concare.pt")
    best_auprc = -1.0
    best_epoch = -1

    # Prepare timestamped results log for this run
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_path = os.path.join(args.results_dir, f"train_val_test_log_{run_timestamp}.txt")
    with open(results_path, "w") as f:
        f.write(f"=== ConCare Training Log Started ({run_timestamp}) ===\n")
        f.write(f"epochs={args.epochs}  batch_size={args.batch_size}  lr={args.lr}  weight_decay={args.weight_decay}  "
                f"lambda_decov={args.lambda_decov}  amp={args.amp}  compile={args.compile}\n")
        f.write(f"input_dim={input_dim}  timestep={args.timestep}  append_masks={args.append_masks}\n\n")

    # decov warmup
    target_lambda = args.lambda_decov
    warmup_epochs = 10

    print(f"\n🚀 Starting training at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Input dimension: {input_dim}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
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

        # --- Authors-style validation metrics on the SAME predictions ---
        if authors_print_metrics_binary is not None:
            # Flip so authors' column-1 corresponds to positive class score
            auth_prob = 1.0 - yv_prob
            authors_val = authors_print_metrics_binary(yv_true, auth_prob, verbose=0)
            print(f"[AUTHORS] Val acc={authors_val['acc']:.4f} "
                  f"AUROC={authors_val['auroc']:.4f} AUPRC={authors_val['auprc']:.4f} "
                  f"MinPSE={authors_val['minpse']:.4f} F1={authors_val['f1_score']:.4f}")

        # track best by AUPRC and log to file (include authors metrics if available)
        if va["auprc"] > best_auprc:
            best_auprc = va["auprc"]
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch}, best_path)
            with open(results_path, "a") as f:
                f.write(f"New best model at epoch {epoch}: val AUPRC={va['auprc']:.4f}, AUROC={va['auroc']:.4f}, "
                        f"loss={va['loss']:.4f}, thr={thr:.2f}, F1={f1:.4f}, P={p:.4f}, R={r:.4f}\n")
                if authors_print_metrics_binary is not None:
                    f.write(f"[AUTHORS] acc={authors_val['acc']:.4f} auroc={authors_val['auroc']:.4f} "
                            f"auprc={authors_val['auprc']:.4f} minpse={authors_val['minpse']:.4f} "
                            f"f1={authors_val['f1_score']:.4f}\n")

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

    # --- Authors-style full test metrics from authors' function ---
    if authors_print_metrics_binary is not None:
        auth_prob_test = 1.0 - yt_prob
        authors_test = authors_print_metrics_binary(yt_true, auth_prob_test, verbose=0)
        print(f"[AUTHORS] Test acc={authors_test['acc']:.4f} "
              f"AUROC={authors_test['auroc']:.4f} AUPRC={authors_test['auprc']:.4f} "
              f"MinPSE={authors_test['minpse']:.4f} F1={authors_test['f1_score']:.4f}")

    # Optional fixed operating point for easy comparison
    thr_fixed = 0.66
    acc66, prec66, rec66, f1_66 = print_thresholded_report(yt_true, yt_prob, thr_fixed, header="📊 Test @thr=0.66")
    minpse_66 = minpse_from_pr(prec66, rec66)
    print(f"      auroc={test_metrics.get('auroc', float('nan')):.4f} "
          f"auprc={test_metrics.get('auprc', float('nan')):.4f} "
          f"minpse={minpse_66:.4f}")

    # Save final test results to the same timestamped log (include authors metrics if available)
    with open(results_path, "a") as f:
        f.write("\n=== Final Test Results ===\n")
        f.write(f"best_epoch={best_epoch}  best_val_auprc={best_auprc:.4f}\n")
        f.write("\n-- threshold-free --\n")
        for k in sorted(test_metrics.keys()):
            v = test_metrics[k]
            try:
                f.write(f"{k:>8}: {float(v):.4f}\n")
            except Exception:
                f.write(f"{k:>8}: {v}\n")
        f.write("\n-- authors-style @best_val_thr --\n")
        f.write(f"acc={acc_best:.4f}  f1={f1_best:.4f}  auroc={test_metrics.get('auroc', float('nan')):.4f}  "
                f"auprc={test_metrics.get('auprc', float('nan')):.4f}  minpse={minpse_best:.4f}  thr_used={best_thr:.2f}\n")
        if authors_print_metrics_binary is not None:
            f.write("\n=== AUTHORS-STYLE TEST METRICS ===\n")
            for k in ["acc","auroc","auprc","minpse","f1_score","prec0","prec1","rec0","rec1"]:
                try:
                    f.write(f"{k}={authors_test[k]:.6f}\n")
                except Exception:
                    pass
        f.write("\n-- fixed @thr=0.66 --\n")
        f.write(f"acc={acc66:.4f}  f1={f1_66:.4f}  auroc={test_metrics.get('auroc', float('nan')):.4f}  "
                f"auprc={test_metrics.get('auprc', float('nan')):.4f}  minpse={minpse_66:.4f}\n")

    print(f"\n[INFO] Saved run log to {results_path}")
    print("\n✅ Training completed successfully")


if __name__ == "__main__":
    main()
