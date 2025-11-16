"""
ConCare Trainer — RAM-only: materialize once -> train from memory
Loads normalized NPZs from data/normalized_data_cache/
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import copy
import importlib
import glob
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import amp
from datetime import datetime
from typing import Dict, List, Optional
from train_helpers import RAMDataset, pad_collate as ram_pad_collate
from train_helpers import set_seed
try:
    from sklearn.model_selection import StratifiedKFold, train_test_split
except Exception:  # pragma: no cover - optional dependency
    StratifiedKFold = None
    train_test_split = None
try:
    from worker_utils import resolve_num_workers
except ImportError:  # pragma: no cover
    from .worker_utils import resolve_num_workers


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

_MODEL_VARIANTS = {
    "concare_full": {
        "class_name": "ConCare",
        "modules": ("model_codes.ConCare_Model_v3", "ConCare_Model_v3"),
        "description": "Full ConCare (multi-channel + DeCov)",
        "uses_decov": True,
    },
    "concare_mc_minus": {
        "class_name": "ConCareMCMinus",
        "modules": ("model_codes.ConCare_MC_minus", "ConCare_MC_minus"),
        "description": "ConCareMC- ablation (visit embedding only, no DeCov)",
        "uses_decov": False,
    },
    "concare_de_minus": {
        "class_name": "ConCareDEMinus",
        "modules": ("model_codes.ConCare_DE_minus", "ConCare_DE_minus"),
        "description": "ConCareDE- ablation (removes demographic channel)",
        "uses_decov": True,
    },
}


def _discretizer_config_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "discretizer_config.json"


def _load_discretizer_feature_dims():
    """
    Returns (value_dim, num_channels) from the discretizer config if available.
    """
    cfg_path = _discretizer_config_path()
    try:
        with cfg_path.open("r") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        return None

    value_dim = 0
    for ch in cfg["id_to_channel"]:
        if cfg["is_categorical_channel"][ch]:
            value_dim += len(cfg["possible_values"][ch])
        else:
            value_dim += 1
    num_channels = len(cfg["id_to_channel"])
    return value_dim, num_channels


def _infer_mask_dim(input_dim: int, append_masks: bool):
    dims = _load_discretizer_feature_dims()
    if dims is None:
        # Fall back to assuming no explicit mask columns if config is missing.
        return 0, input_dim

    value_dim, channel_count = dims
    if append_masks:
        expected = value_dim + channel_count
        if input_dim != expected:
            print(f"[WARN] Expected {expected} input dims from discretizer but received {input_dim}. "
                  f"Using residual value ({input_dim - value_dim}) as mask_dim.")
            mask_dim = max(input_dim - value_dim, 0)
        else:
            mask_dim = channel_count
    else:
        mask_dim = max(input_dim - value_dim, 0)
    return mask_dim, value_dim


def _clone_ema_model(model: torch.nn.Module):
    ema = copy.deepcopy(model)
    for param in ema.parameters():
        param.requires_grad_(False)
    return ema


@torch.no_grad()
def _update_ema_model(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)


def _freeze_concare_encoder(model: torch.nn.Module, freeze: bool):
    """
    Freeze all parameters except the final prediction head.
    """
    if not hasattr(model, "output0"):
        return
    for name, param in model.named_parameters():
        if name.startswith("output0") or name.startswith("output1"):
            param.requires_grad_(True)
        else:
            param.requires_grad_(not freeze)


def _mask_random_observations(
    X: torch.Tensor,
    value_dim: int,
    mask_dim: int,
    min_ratio: float,
    max_ratio: float,
    value_to_channel: List[int],
):
    """
    Randomly remove observed measurements with probability sampled
    from [min_ratio, max_ratio]. Returns augmented tensor and
    per-feature removal weights.
    """
    if mask_dim <= 0:
        raise ValueError("Missing-aware pretraining requires append_masks=True to supply observation masks.")
    augmented = X.clone()
    values = augmented[:, :, :value_dim]
    masks = augmented[:, :, value_dim:]
    observed = masks > 0.5
    ratio = torch.empty(X.size(0), 1, 1, device=X.device).uniform_(min_ratio, max_ratio)
    removed = observed & (torch.rand_like(masks) < ratio)
    idx = torch.tensor(value_to_channel, device=X.device, dtype=torch.long)
    removal_values = torch.index_select(removed, dim=2, index=idx)
    values = values.masked_fill(removal_values, 0.0)
    augmented[:, :, :value_dim] = values
    new_mask = masks * (~removed)
    augmented[:, :, value_dim:] = new_mask.float()
    removal_weights = removal_values.float().mean(dim=1)  # [B, value_dim]
    return augmented, removal_weights


def _latent_reconstruction_loss(student_ctx, teacher_ctx, removal_weights):
    """
    Compute normalized L1 loss where removal_weights selects features that were masked.
    """
    if removal_weights is None:
        return None
    demo_pad = torch.zeros(removal_weights.size(0), 1, device=removal_weights.device)
    weights = torch.cat([removal_weights, demo_pad], dim=1).unsqueeze(-1)  # [B, value_dim+1, 1]
    total_weight = weights.sum()
    if total_weight.item() <= 1e-6:
        return None
    diff = torch.abs(student_ctx.float() - teacher_ctx.detach().float())
    loss = (diff * weights).sum() / (total_weight * diff.size(-1) + 1e-6)
    return loss


def _run_missing_aware_pretraining(model, train_loader, args):
    if args.missing_aware_pretrain_epochs <= 0:
        return
    if not hasattr(model, "mask_dim") or model.mask_dim <= 0:
        raise ValueError("Missing-aware pretraining requires append_masks=True so that observation masks exist.")

    print(f"\n[SMART] Starting latent reconstruction pre-training for {args.missing_aware_pretrain_epochs} epochs")
    ema_model = _clone_ema_model(model).to(args.device)
    ema_model.eval()
    pretrain_lr = args.missing_aware_pretrain_lr or args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=pretrain_lr, weight_decay=args.weight_decay)
    scaler = amp.GradScaler("cuda") if args.amp and "cuda" in args.device else None
    device_type = "cuda" if isinstance(args.device, str) and "cuda" in args.device else "cpu"

    for epoch in range(1, args.missing_aware_pretrain_epochs + 1):
        model.train()
        running_loss = 0.0
        steps = 0
        for X, D, _ in train_loader:
            X = X.to(args.device, non_blocking=True)
            D = D.to(args.device, non_blocking=True)
            augmented, removal_weights = _mask_random_observations(
                X,
                model.value_dim,
                model.mask_dim,
                args.missing_aware_mask_ratio_min,
                args.missing_aware_mask_ratio_max,
                model.value_to_channel,
            )
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                with amp.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=(device_type == "cuda")):
                    student_ctx, _, _ = model(augmented, D, return_context=True)
                with torch.no_grad():
                    ema_model.eval()
                    teacher_ctx, _, _ = ema_model(X, D, return_context=True)
                loss = _latent_reconstruction_loss(student_ctx, teacher_ctx, removal_weights)
                if loss is None:
                    continue
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                student_ctx, _, _ = model(augmented, D, return_context=True)
                with torch.no_grad():
                    ema_model.eval()
                    teacher_ctx, _, _ = ema_model(X, D, return_context=True)
                loss = _latent_reconstruction_loss(student_ctx, teacher_ctx, removal_weights)
                if loss is None:
                    continue
                loss.backward()
                optimizer.step()

            _update_ema_model(ema_model, model, args.missing_aware_ema_decay)
            running_loss += loss.item()
            steps += 1
        avg_loss = running_loss / max(steps, 1)
        print(f"[SMART] Pre-train epoch {epoch}/{args.missing_aware_pretrain_epochs}  loss={avg_loss:.6f}")

    ema_model.cpu()
    del ema_model


def _load_model_class(variant_key: str):
    meta = _MODEL_VARIANTS.get(variant_key)
    if meta is None:
        raise ValueError(f"Unknown model_variant '{variant_key}'")
    last_exc = None
    for module_name in meta["modules"]:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, meta["class_name"])
        except (ModuleNotFoundError, AttributeError) as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Unable to import model class for variant '{variant_key}'")


def build_argparser():
    p = argparse.ArgumentParser("ConCare Trainer (RAM-only)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_scheduler", choices=["none", "cosine"], default="none",
                   help="Optional LR scheduler for fine-tuning stage")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--lambda_decov", type=float, default=1e-4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir", type=str, default="trained_models")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--cache_dir", type=str, default="data/normalized_data_cache_train",
                   help="Path to normalized NPZ cache (e.g., data/normalized_data_cache_train or _all)")
    p.add_argument("--timestep", type=float, default=0.8)
    p.add_argument("--append_masks", action="store_true", default=True,
                   help="Append binary masks to values in discretizer (2F features).")
    p.add_argument("--diag", action="store_true", help="Run preflight diagnostics before training")
    p.add_argument("--compile", action="store_true", help="Enable torch.compile if Triton is available")
    p.add_argument("--papers_metrics_mode", action="store_true",
                   help="Use the authors' metric implementation for AUROC and AUPRC")
    p.add_argument("--num_workers", type=int, default=-1,
                   help="DataLoader workers per process (-1 auto, 0 disables multiprocessing)")
    p.add_argument("--model_variant", choices=sorted(_MODEL_VARIANTS.keys()), default="concare_full",
                   help="Choose which ConCare variant to train (full model vs. ConCareMC- ablation)")
    p.add_argument("--missing_aware_extension", action="store_true",
                   help="Enable SMART-style missing-aware attention extension for ConCare")
    p.add_argument("--missing_aware_pretrain_epochs", type=int, default=0,
                   help="Number of latent reconstruction epochs before fine-tuning (SMART pre-training)")
    p.add_argument("--missing_aware_pretrain_lr", type=float, default=None,
                   help="Learning rate for the missing-aware pre-training stage (defaults to --lr)")
    p.add_argument("--missing_aware_mask_ratio_min", type=float, default=0.2,
                   help="Lower bound for random removal probability during pre-training")
    p.add_argument("--missing_aware_mask_ratio_max", type=float, default=0.8,
                   help="Upper bound for random removal probability during pre-training")
    p.add_argument("--missing_aware_ema_decay", type=float, default=0.99,
                   help="EMA decay used to update the teacher encoder during pre-training")
    p.add_argument("--missing_aware_freeze_epochs", type=int, default=0,
                   help="Freeze encoder parameters for this many fine-tuning epochs (decoder only learns)")
    p.add_argument("--missing_aware_aux_weight", type=float, default=0.0,
                   help="Weight of latent reconstruction auxiliary loss during fine-tuning (0 disables)")
    p.add_argument("--keep_prob", type=float, default=0.5,
                   help="Dropout keep probability for ConCare encoders/heads (default 0.5)")
    p.add_argument("--missing_aware_disable_mask_bias", action="store_true",
                   help="Disable mask-biased feature attention when SMART extension is on")
    p.add_argument("--missing_aware_disable_temporal_attention", action="store_true",
                   help="Use ConCare's original per-feature attention even when SMART extension is enabled")
    p.add_argument("--cv_folds", type=int, default=0,
                   help="Number of folds for repeated cross-validation (0 disables CV mode)")
    p.add_argument("--cv_repeats", type=int, default=1,
                   help="How many random repetitions of the cross-validation to run")
    p.add_argument("--cv_pool_splits", type=str, default="train,val",
                   help="Comma-separated cache splits to pool together for CV (e.g., 'train,val,test')")
    p.add_argument("--cv_val_ratio", type=float, default=0.1,
                   help="Portion of the training portion inside each fold that is used for validation")
    p.add_argument("--cv_seed", type=int, default=42,
                   help="Seed used to shuffle folds/repetitions for CV mode")
    p.add_argument("--early_stop_patience", type=int, default=0,
                   help="Stop training if val AUPRC fails to improve after this many epochs (0 disables)")
    p.add_argument("--early_stop_min_delta", type=float, default=0.0,
                   help="Minimum AUPRC improvement to reset patience when early stopping is enabled")

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


def make_model(input_dim, device, args, use_compile=False):
    meta = _MODEL_VARIANTS[args.model_variant]
    ModelClass = _load_model_class(args.model_variant)
    extra_kwargs = {}
    mask_dim = base_value_dim = None
    if meta["class_name"] == "ConCare" and args.missing_aware_extension:
        mask_dim, base_value_dim = _infer_mask_dim(input_dim, args.append_masks)
        extra_kwargs["mask_dim"] = mask_dim
        extra_kwargs["base_value_dim"] = base_value_dim
        extra_kwargs["enable_missing_aware"] = True
        extra_kwargs["enable_mask_bias"] = not args.missing_aware_disable_mask_bias
        extra_kwargs["use_missing_temporal_attention"] = not args.missing_aware_disable_temporal_attention
    elif meta["class_name"] == "ConCare":
        extra_kwargs["enable_missing_aware"] = False
    print(f"[INFO] Creating model variant='{args.model_variant}' ({meta['description']}) "
          f"with input_dim={input_dim}"
          + (f" (value_dim≈{base_value_dim}, mask_dim={mask_dim})" if mask_dim is not None else ""))
    model = ModelClass(
        input_dim=input_dim,
        hidden_dim=64,
        d_model=64,
        MHD_num_head=4,
        d_ff=256,
        output_dim=1,
        keep_prob=args.keep_prob,
        demographic_dim=12,
        **extra_kwargs,
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


def diag_preflight(train_ds, device, collate_fn, args):
    print("\n[DIAG] ===== Preflight diagnostics =====")
    X0, D0, y0 = train_ds[0]
    print(f"[DIAG] First item shapes -> X:{tuple(X0.shape)} D:{tuple(D0.shape)} y:{tuple(y0.shape)}")
    loader = DataLoader(train_ds, batch_size=8, shuffle=False, collate_fn=collate_fn,
                        num_workers=0, pin_memory=True)
    Xb, Db, yb = next(iter(loader))
    tensor_stats("X batch", Xb); tensor_stats("D batch", Db); tensor_stats("y batch", yb)
    model = make_model(Xb.shape[-1], device, args, use_compile=False)
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


def _sample_auxiliary_mask(X, value_dim, mask_dim, ratio, value_to_channel):
    if mask_dim <= 0 or ratio <= 0:
        return X, None
    idx = torch.tensor(value_to_channel, device=X.device, dtype=torch.long)
    augmented = X.clone()
    values = augmented[:, :, :value_dim]
    masks = augmented[:, :, value_dim:]
    observed = masks > 0.5
    removal = observed & (torch.rand_like(masks) < ratio)
    removal_values = torch.index_select(removal, dim=2, index=idx)
    values = values.masked_fill(removal_values, 0.0)
    augmented[:, :, :value_dim] = values
    augmented[:, :, value_dim:] = masks * (~removal)
    removal_weights = removal_values.float().mean(dim=1)
    return augmented, removal_weights


# Choose which metric function to use (threshold-free set)
def select_metric_fn(papers_mode: bool):
    # Always use local threshold-free metrics for training curves
    if papers_mode:
        if authors_print_metrics_binary is None:
            print("[WARN] papers_metrics_mode requested but metrics_authors.py not found. Using local metrics only.")
        else:
            print("[INFO] papers_metrics_mode ON — authors' print_metrics_binary will be reported alongside local metrics.")
    return ours_binary_metrics, "local"


def train_one_epoch(model, loader, optimizer, scaler, device, lambda_decov, epoch, criterion, metric_fn, aux_cfg=None):
    model.train()
    total_loss = 0.0
    probs, labels = [], []
    device_type = "cuda" if isinstance(device, str) and "cuda" in device else "cpu"
    aux_weight = aux_cfg["weight"] if aux_cfg else 0.0
    aux_ratio = aux_cfg["ratio"] if aux_cfg else 0.0
    for batch_idx, (X, D, y) in enumerate(loader):
        X, D, y = X.to(device, non_blocking=True), D.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler:
            with amp.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=True):
                logits, decov = model(X, D)
                decov = _sanitize_decov(decov)
                aux_loss = None
                if aux_weight > 0.0 and hasattr(model, "value_dim") and model.mask_dim > 0:
                    aug_X, removal_weights = _sample_auxiliary_mask(X, model.value_dim, model.mask_dim, aux_ratio, model.value_to_channel)
                    if removal_weights is not None:
                        student_ctx, _, _ = model(aug_X, D, return_context=True)
                        with torch.no_grad():
                            teacher_ctx, _, _ = model(X, D, return_context=True)
                        aux_loss = _latent_reconstruction_loss(student_ctx, teacher_ctx, removal_weights)
            with amp.autocast(device_type=device_type, enabled=False):
                bce = criterion(logits.float(), y.float())
            loss = bce + lambda_decov * decov.float()
            if aux_loss is not None:
                loss = loss + aux_weight * aux_loss
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
            aux_loss = None
            if aux_weight > 0.0 and hasattr(model, "value_dim") and model.mask_dim > 0:
                aug_X, removal_weights = _sample_auxiliary_mask(X, model.value_dim, model.mask_dim, aux_ratio, model.value_to_channel)
                if removal_weights is not None:
                    student_ctx, _, _ = model(aug_X, D, return_context=True)
                    with torch.no_grad():
                        teacher_ctx, _, _ = model(X, D, return_context=True)
                    aux_loss = _latent_reconstruction_loss(student_ctx, teacher_ctx, removal_weights)
            loss = bce + lambda_decov * decov
            if aux_loss is not None:
                loss = loss + aux_weight * aux_loss
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
    variant_meta = _MODEL_VARIANTS[args.model_variant]

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

    if args.cv_folds and args.cv_folds > 1:
        _run_cross_validation(args, variant_meta, metric_fn, metric_source)
        return

    train_ds = RAMDataset("train", cache_dir=CACHE_DIR)
    val_ds   = RAMDataset("val",   cache_dir=CACHE_DIR)
    test_ds  = RAMDataset("test",  cache_dir=CACHE_DIR)
    run_training_cycle(args, variant_meta, metric_fn, metric_source, train_ds, val_ds, test_ds)


def run_training_cycle(args, variant_meta, metric_fn, metric_source,
                       train_ds, val_ds, test_ds, fold_tag: Optional[str] = None,
                       run_diag: bool = True):
    if fold_tag:
        print(f"\n===== Fold {fold_tag} =====")
    X0, _, _ = train_ds[0]; input_dim = X0.shape[1]

    workers, use_workers = resolve_num_workers(args.num_workers)
    if args.num_workers >= 0:
        print(f"[INFO] Using user-provided num_workers={args.num_workers}")
    else:
        print(f"[INFO] Auto-selected num_workers={workers}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=ram_pad_collate,
                              num_workers=workers, pin_memory=True, persistent_workers=use_workers,
                              prefetch_factor=2 if use_workers else None)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=ram_pad_collate,
                              num_workers=workers, pin_memory=True, persistent_workers=use_workers,
                              prefetch_factor=2 if use_workers else None)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, collate_fn=ram_pad_collate,
                              num_workers=workers, pin_memory=True, persistent_workers=use_workers,
                              prefetch_factor=2 if use_workers else None)

    if args.diag and run_diag:
        diag_preflight(train_ds, args.device, ram_pad_collate, args)

    model = make_model(input_dim, args.device, args, use_compile=args.compile)

    if args.missing_aware_extension and args.missing_aware_pretrain_epochs > 0:
        _run_missing_aware_pretraining(model, train_loader, args)
        model.train()

    if not variant_meta["uses_decov"] and args.lambda_decov != 0.0:
        print(f"[WARN] Model variant '{args.model_variant}' disables DeCov. "
              f"Overriding lambda_decov={args.lambda_decov} → 0.0")
    target_lambda = args.lambda_decov if variant_meta["uses_decov"] else 0.0
    warmup_epochs = 10

    # ---- Unweighted BCE on probabilities ----
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)
    else:
        scheduler = None
    scaler = amp.GradScaler("cuda") if args.amp and "cuda" in args.device else None
    freeze_epochs = args.missing_aware_freeze_epochs if args.missing_aware_extension else 0
    if freeze_epochs > 0:
        print(f"[SMART] Freezing encoder for first {freeze_epochs} fine-tuning epochs")
        _freeze_concare_encoder(model, True)
    else:
        _freeze_concare_encoder(model, False)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    fold_suffix = f"_{fold_tag}" if fold_tag else ""
    best_path = os.path.join(args.save_dir, f"best_concare{fold_suffix}.pt")
    best_auprc = -1.0
    best_epoch = -1
    epochs_no_improve = 0

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_path = os.path.join(args.results_dir, f"train_val_test_log_{run_timestamp}{fold_suffix}.txt")
    with open(results_path, "w") as f:
        f.write(f"=== ConCare Training Log Started ({run_timestamp}) ===\n")
        if fold_tag:
            f.write(f"cv_fold={fold_tag}\n")
        f.write(f"epochs={args.epochs}  batch_size={args.batch_size}  lr={args.lr}  weight_decay={args.weight_decay}  "
                f"lambda_decov={args.lambda_decov}  effective_lambda={target_lambda}  amp={args.amp}  compile={args.compile}\n")
        f.write(f"model_variant={args.model_variant} ({variant_meta['description']})  uses_decov={variant_meta['uses_decov']}\n")
        f.write(f"input_dim={input_dim}  timestep={args.timestep}  append_masks={args.append_masks}  keep_prob={args.keep_prob}\n")
        f.write(f"lr_scheduler={args.lr_scheduler}\n\n")
        if args.missing_aware_extension:
            f.write(f"missing_aware_extension=1  pretrain_epochs={args.missing_aware_pretrain_epochs}  "
                    f"mask_ratio=[{args.missing_aware_mask_ratio_min},{args.missing_aware_mask_ratio_max}]  "
                    f"freeze_epochs={args.missing_aware_freeze_epochs}  "
                    f"aux_weight={args.missing_aware_aux_weight}  "
                    f"mask_bias={'on' if not args.missing_aware_disable_mask_bias else 'off'}  "
                    f"temporal_attn={'mask' if not args.missing_aware_disable_temporal_attention else 'original'}\n\n")

    print(f"\n🚀 Starting training at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if fold_tag:
        print(f"   Fold tag: {fold_tag}")
    print(f"   Input dimension: {input_dim}")
    print(f"   Model variant: {args.model_variant} ({variant_meta['description']})")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Using AMP: {args.amp}")
    print(f"   Using torch.compile: {args.compile}")
    print(f"   Metrics source: {metric_source}{' (papers_metrics_mode ON)' if args.papers_metrics_mode else ''}\n")

    for epoch in range(1, args.epochs + 1):
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            _freeze_concare_encoder(model, False)
            print("[SMART] Encoder unfrozen; joint fine-tuning begins.")
        lambda_decov = target_lambda * (epoch / warmup_epochs) if epoch <= warmup_epochs else target_lambda

        aux_cfg = None
        if args.missing_aware_extension and args.missing_aware_aux_weight > 0.0:
            aux_cfg = {"weight": args.missing_aware_aux_weight, "ratio": args.missing_aware_mask_ratio_min}

        tr = train_one_epoch(model, train_loader, optimizer, scaler, args.device, lambda_decov, epoch, criterion, metric_fn, aux_cfg=aux_cfg)
        va, yv_true, yv_prob = evaluate(model, val_loader, args.device, criterion, metric_fn)

        print(f"Epoch {epoch:03d} | Train loss {tr['loss']:.4f} AUPRC {tr['auprc']:.4f} AUROC {tr['auroc']:.4f} | "
              f"Val loss {va['loss']:.4f} AUPRC {va['auprc']:.4f} AUROC {va['auroc']:.4f}")

        thr, f1, p, r = best_threshold_from_probs(yv_true, yv_prob)
        print(f"          Val@thr={thr:.2f}  F1={f1:.4f}  P={p:.4f}  R={r:.4f}")

        authors_val = None
        if authors_print_metrics_binary is not None:
            auth_prob = 1.0 - yv_prob
            authors_val = authors_print_metrics_binary(yv_true, auth_prob, verbose=0)
            print(f"[AUTHORS] Val acc={authors_val['acc']:.4f} "
                  f"AUROC={authors_val['auroc']:.4f} AUPRC={authors_val['auprc']:.4f} "
                  f"MinPSE={authors_val['minpse']:.4f} F1={authors_val['f1_score']:.4f}")

        improved = va["auprc"] > (best_auprc + args.early_stop_min_delta)
        if improved:
            best_auprc = va["auprc"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({"model": model.state_dict(), "epoch": epoch}, best_path)
            with open(results_path, "a") as f:
                f.write(f"New best model at epoch {epoch}: val AUPRC={va['auprc']:.4f}, AUROC={va['auroc']:.4f}, "
                        f"loss={va['loss']:.4f}, thr={thr:.2f}, F1={f1:.4f}, P={p:.4f}, R={r:.4f}\n")
                if authors_val is not None:
                    f.write(f"[AUTHORS] acc={authors_val['acc']:.4f} auroc={authors_val['auroc']:.4f} "
                            f"auprc={authors_val['auprc']:.4f} minpse={authors_val['minpse']:.4f} "
                            f"f1={authors_val['f1_score']:.4f}\n")
        else:
            epochs_no_improve += 1

        if scheduler is not None:
            scheduler.step()

        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
            print(f"[EARLY STOP] No val AUPRC improvement for {args.early_stop_patience} epochs. Stopping at epoch {epoch}.")
            with open(results_path, "a") as f:
                f.write(f"[EARLY STOP] Triggered at epoch {epoch} "
                        f"(best_epoch={best_epoch}, best_val_auprc={best_auprc:.4f})\n")
            break

    print(f"\n📊 Evaluating best checkpoint on TEST set")
    ckpt = torch.load(best_path, map_location=args.device) if os.path.exists(best_path) else None
    if ckpt:
        model.load_state_dict(ckpt["model"])

    test_metrics, yt_true, yt_prob = evaluate(model, test_loader, args.device, criterion, metric_fn)

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

    authors_test = None
    if authors_print_metrics_binary is not None:
        auth_prob_test = 1.0 - yt_prob
        authors_test = authors_print_metrics_binary(yt_true, auth_prob_test, verbose=0)
        print(f"[AUTHORS] Test acc={authors_test['acc']:.4f} "
              f"AUROC={authors_test['auroc']:.4f} AUPRC={authors_test['auprc']:.4f} "
              f"MinPSE={authors_test['minpse']:.4f} F1={authors_test['f1_score']:.4f}")

    thr_fixed = 0.66
    acc66, prec66, rec66, f1_66 = print_thresholded_report(yt_true, yt_prob, thr_fixed, header="📊 Test @thr=0.66")
    minpse_66 = minpse_from_pr(prec66, rec66)
    print(f"      auroc={test_metrics.get('auroc', float('nan')):.4f} "
          f"auprc={test_metrics.get('auprc', float('nan')):.4f} "
          f"minpse={minpse_66:.4f}")

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
        if authors_test is not None:
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

    summary = {
        "fold_tag": fold_tag,
        "results_path": results_path,
        "best_epoch": best_epoch,
        "best_val_auprc": best_auprc,
        "best_thr": best_thr,
        "acc_best": acc_best,
        "f1_best": f1_best,
        "minpse_best": minpse_best,
        "test_metrics": test_metrics,
    }
    if authors_test is not None:
        summary["authors_test"] = authors_test
    return summary


def _load_cv_pool(cache_dir: str, splits: List[str]) -> Dict[str, np.ndarray]:
    if not splits:
        raise ValueError("cv_pool_splits must specify at least one split")
    Xs, ys, Ds = [], [], []
    ds_present = True
    for split in splits:
        split = split.strip()
        if not split:
            continue
        path = Path(cache_dir) / f"{split}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Cache split '{split}' not found at {path}")
        arr = np.load(path, allow_pickle=True)
        X_chunk = list(arr["X"])
        y_chunk = list(arr["y"].astype(np.float32))
        Xs.extend(X_chunk)
        ys.extend(y_chunk)
        if ds_present and "D" in arr.files:
            Ds.extend(list(arr["D"]))
        elif "D" not in arr.files:
            ds_present = False
            Ds = []
    bundle = {
        "X": np.array(Xs, dtype=object),
        "y": np.array(ys, dtype=np.float32),
    }
    if ds_present and Ds:
        bundle["D"] = np.array(Ds, dtype=np.float32)
    print(f"[CV] Loaded {len(ys)} stays from splits {splits} (cache_dir={cache_dir})")
    return bundle


def _mean_std(values: List[float]) -> Optional[tuple]:
    if not values:
        return None
    arr = np.array(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def _summarize_cv_results(summaries: List[Dict], args):
    if not summaries:
        print("[CV] No runs to summarize.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    summary_path = os.path.join(args.results_dir, f"cv_summary_{timestamp}.txt")
    metric_lists: Dict[str, List[float]] = {}
    thresh_lists: Dict[str, List[float]] = {}
    author_lists: Dict[str, List[float]] = {}

    def _push(target, key, value):
        if value is None:
            return
        try:
            val = float(value)
        except Exception:
            return
        if np.isnan(val):
            return
        target.setdefault(key, []).append(val)

    def _fmt(value):
        try:
            return f"{float(value):.4f}"
        except Exception:
            return "nan"

    with open(summary_path, "w") as f:
        f.write(f"=== Repeated CV Summary ({timestamp}) ===\n")
        for idx, summary in enumerate(summaries, 1):
            fold_tag = summary.get("fold_tag") or f"run{idx}"
            test_metrics = summary.get("test_metrics", {})
            acc_best = summary.get("acc_best")
            f1_best = summary.get("f1_best")
            minpse_best = summary.get("minpse_best")
            auroc = test_metrics.get("auroc")
            auprc = test_metrics.get("auprc")
            loss = test_metrics.get("loss")
            minpse = test_metrics.get("minpse")
            f.write(f"{fold_tag}: auroc={_fmt(auroc)} "
                    f"auprc={_fmt(auprc)} "
                    f"loss={_fmt(loss)} "
                    f"minpse={_fmt(minpse)} "
                    f"acc={_fmt(acc_best)} "
                    f"f1={_fmt(f1_best)} "
                    f"thr={_fmt(summary.get('best_thr'))}\n")

            _push(metric_lists, "auroc", auroc)
            _push(metric_lists, "auprc", auprc)
            _push(metric_lists, "loss", loss)
            _push(metric_lists, "minpse", minpse)
            _push(thresh_lists, "acc", acc_best)
            _push(thresh_lists, "f1", f1_best)
            _push(thresh_lists, "minpse", minpse_best)

            authors_test = summary.get("authors_test")
            if authors_test:
                for key in ["acc", "auroc", "auprc", "minpse", "f1_score"]:
                    _push(author_lists, key, authors_test.get(key))

        f.write("\n=== Aggregate (threshold-free) ===\n")
        for key in sorted(metric_lists.keys()):
            stats = _mean_std(metric_lists[key])
            if stats:
                f.write(f"{key}: mean={stats[0]:.4f} std={stats[1]:.4f}\n")

        f.write("\n=== Aggregate (thresholded @best_val) ===\n")
        for key in sorted(thresh_lists.keys()):
            stats = _mean_std(thresh_lists[key])
            if stats:
                f.write(f"{key}: mean={stats[0]:.4f} std={stats[1]:.4f}\n")

        if author_lists:
            f.write("\n=== Aggregate (authors metrics) ===\n")
            for key in sorted(author_lists.keys()):
                stats = _mean_std(author_lists[key])
                if stats:
                    f.write(f"{key}: mean={stats[0]:.4f} std={stats[1]:.4f}\n")

    print(f"[CV] Saved summary to {summary_path}")


def _run_cross_validation(args, variant_meta, metric_fn, metric_source):
    if StratifiedKFold is None or train_test_split is None:
        raise ImportError("scikit-learn is required for cross-validation mode. Please install scikit-learn.")
    if args.cv_folds <= 1:
        raise ValueError("--cv_folds must be > 1 to enable cross-validation")
    if not (0.0 < args.cv_val_ratio < 1.0):
        raise ValueError("--cv_val_ratio must be between 0 and 1")
    split_names = [s.strip() for s in args.cv_pool_splits.split(",") if s.strip()]
    pool_bundle = _load_cv_pool(args.cache_dir, split_names)
    y_all = pool_bundle["y"]
    total = len(y_all)
    print(f"[CV] Starting repeated CV with {args.cv_folds} folds x {args.cv_repeats} repeats ({total} stays)")
    summaries = []
    for repeat in range(args.cv_repeats):
        skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True,
                              random_state=args.cv_seed + repeat)
        for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(np.zeros(total), y_all)):
            fold_tag = f"rep{repeat+1}_fold{fold_idx+1}"
            print(f"[CV] Preparing {fold_tag}: train+val={len(trainval_idx)} test={len(test_idx)}")
            train_idx, val_idx = train_test_split(
                trainval_idx,
                test_size=args.cv_val_ratio,
                stratify=y_all[trainval_idx],
                random_state=args.cv_seed + (repeat * args.cv_folds) + fold_idx,
            )
            set_seed(args.cv_seed + (repeat * args.cv_folds) + fold_idx)
            train_ds = RAMDataset(split=None, cache_dir=args.cache_dir, data_bundle=pool_bundle, indices=train_idx)
            val_ds   = RAMDataset(split=None, cache_dir=args.cache_dir, data_bundle=pool_bundle, indices=val_idx)
            test_ds  = RAMDataset(split=None, cache_dir=args.cache_dir, data_bundle=pool_bundle, indices=test_idx)
            summary = run_training_cycle(
                args, variant_meta, metric_fn, metric_source,
                train_ds, val_ds, test_ds,
                fold_tag=fold_tag,
                run_diag=(repeat == 0 and fold_idx == 0),
            )
            summaries.append(summary)
    _summarize_cv_results(summaries, args)


if __name__ == "__main__":
    main()
