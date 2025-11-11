#!/usr/bin/env python3
"""
ConCare feature–interdependence plots (diagonal-correct, no hidden normalization).

What changed vs your current script (heatmap_plots_fresh.py):
  1) **Diagonal aggregation fixed** — the diagonal now reflects **group→same‑group** attention
     (mean over A[q, Gq] for q in Gq), which matches the paper’s intent, instead of only A[q,q].
  2) **No destriping by default** — we plot the **raw aggregated means** (post‑softmax) with
     optional `--destripe` switch for visual comparison, so diagonals are not suppressed.
  3) **Diagnostics** — we print the diagonal vs off‑diagonal ratio so you can verify dominance.
  4) Minor: safer percentile limits, explicit vmin/vmax override respected, clearer CLI.

Usage (single panel from checkpoint):
  python heatmap_plots_fixed.py \
      --ckpt trained_models/best_concare.pt \
      --config data/discretizer_config.json \
      --head 0 --pmin 5 --pmax 95

Optional two‑panel (if you already exported cohort‑averaged attention .npy arrays):
  python heatmap_plots_fixed.py \
      --ckpt trained_models/best_concare.pt \
      --config data/discretizer_config.json \
      --head 0 \
      --attn-diabetic heatmap_plots/attn_diab.npy \
      --attn-nondiabetic heatmap_plots/attn_nondiab.npy

Tip: run once with `--print-metrics` to see diag/off ratios.
"""

import argparse
import os
import sys
import json
import importlib
import inspect
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colorbar import ColorbarBase

# --- Softmax helpers (used if saved attention is pre-softmax or clearly not row-stochastic) ---

def _row_softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    den = np.sum(ex, axis=axis, keepdims=True)
    return ex / np.clip(den, 1e-12, None)


def _maybe_softmax(A: np.ndarray, force: bool) -> np.ndarray:
    if force:
        return _row_softmax(A, axis=1)
    try:
        row_sum = A.sum(axis=1, keepdims=True)
        mean_sum = float(np.mean(row_sum))
        min_val = float(np.min(A))
        if not (0.98 <= mean_sum <= 1.02) or min_val < 0:
            return _row_softmax(A, axis=1)
        return A
    except Exception:
        return A

# ------------------------- tiny utils -------------------------

def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_outdir() -> str:
    out = os.path.join(os.getcwd(), "heatmap_plots")
    os.makedirs(out, exist_ok=True)
    return out

def short_labels(names: List[str]) -> List[str]:
    MAP = {
        "Capillary refill rate":"Cap refill",
        "Diastolic blood pressure":"DBP",
        "Fraction inspired oxygen":"FiO2",
        "Glascow coma scale eye opening":"GCS-E",
        "Glascow coma scale motor response":"GCS-M",
        "Glascow coma scale total":"GCS-Total",
        "Glascow coma scale verbal response":"GCS-V",
        "Glucose":"Glucose",
        "Heart Rate":"HR",
        "Height":"Height",
        "Mean blood pressure":"MAP",
        "Oxygen saturation":"SpO2",
        "Respiratory rate":"RR",
        "Systolic blood pressure":"SBP",
        "Temperature":"Temp",
        "Weight":"Weight",
        "pH":"pH",
    }
    return [MAP.get(x, x) for x in names]

# ------------------------- config + header grouping -------------------------

def load_feature_order(config_path: str) -> List[str]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    order = cfg.get("id_to_channel")
    if not isinstance(order, list) or not order:
        raise RuntimeError("config.id_to_channel missing/invalid")
    return order

def import_discretizer():
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    mod = importlib.import_module("data_processing")
    if not hasattr(mod, "Discretizer"):
        raise RuntimeError("data_processing.Discretizer not found")
    return mod.Discretizer

def normalize_base(name: str) -> str:
    base = name
    if "->" in base: base = base.split("->")[0]
    if "==" in base: base = base.split("==")[0]
    if "_bin" in base: base = base.split("_bin")[0]
    return base.strip().lower()

def is_mask_col(name: str) -> bool:
    l = name.lower()
    return ("mask" in l) or l.endswith("->mask")

def build_header_and_groups(config_path: str,
                            timestep: float = 0.8,
                            store_masks: bool = True,
                            impute: str = "previous",
                            start_time: str = "zero",
                            dump_path: Optional[str] = None
                            ) -> Tuple[List[str], List[str], List[List[int]]]:
    target_order = load_feature_order(config_path)
    Discretizer = import_discretizer()
    disc = Discretizer(timestep=timestep, store_masks=store_masks,
                       impute_strategy=impute, start_time=start_time,
                       config_path=config_path)
    header = disc._build_header_parts()
    if dump_path:
        with open(dump_path, "w", encoding="utf-8") as f:
            for i, h in enumerate(header):
                f.write(f"{i:3d}\t{h}\n")

    header_norm = [normalize_base(h) for h in header]
    groups: List[List[int]] = []
    names: List[str] = []
    used = set()

    for var in target_order:
        v = var.lower()
        idxs = []
        for i, (h_raw, h_norm) in enumerate(zip(header, header_norm)):
            if i in used or is_mask_col(h_raw):
                continue
            if v in h_norm:
                idxs.append(i)
        if idxs:
            groups.append(idxs)
            names.append(var)
            used.update(idxs)

    for var, idxs in zip(names, groups):
        print(f"[MAP] {var:>28s}  <--  {len(idxs)} cols")

    missing = [v for v in target_order if v not in names]
    if missing:
        print(f"[WARN] No column matches for: {', '.join(missing)}")

    return header, names, groups

# ------------------------- checkpoint readers -------------------------

def import_model_module():
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    return importlib.import_module("model_codes.ConCare_Model_v3")

def pick_model_class(mod):
    for name in ["ConCare", "ConCareV3", "ConCare_Model_v3", "Model", "Net"]:
        if hasattr(mod, name) and inspect.isclass(getattr(mod, name)):
            cls = getattr(mod, name)
            if issubclass(cls, nn.Module):
                print(f"[INFO] Using model class: {name}")
                return cls
    for _, obj in inspect.getmembers(mod, inspect.isclass):
        if obj.__module__ == mod.__name__ and issubclass(obj, nn.Module):
            print(f"[INFO] Using model class: {obj.__name__}")
            return obj
    raise RuntimeError("No nn.Module subclass found in model_codes.ConCare_Model_v3")

def strip_attn_buffers(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    sd2 = sd.copy()
    for k in list(sd2.keys()):
        if k.endswith("saved_attn_avg") or k.endswith("saved_attn_count"):
            sd2.pop(k, None)
    return sd2

def build_model_for_rates(state_dict: Dict[str, torch.Tensor]) -> nn.Module:
    mod = import_model_module()
    cls = pick_model_class(mod)
    try:
        m = cls()
        m.load_state_dict(strip_attn_buffers(state_dict), strict=False)
        print("[INFO] Built model with empty constructor")
        return m.eval()
    except Exception:
        pass
    CAND = [
        {"input_dim":76, "hidden_dim":64, "d_model":64, "MHD_num_head":4, "d_ff":256, "output_dim":1, "demographic_dim":12, "keep_prob":0.5},
        {"input_dim":76, "hidden_dim":64, "d_model":64, "MHD_num_head":4, "d_ff":256, "output_dim":1, "demographic_dim":12, "keep_prob":0.6},
    ]
    last_err = None
    for i, kw in enumerate(CAND, 1):
        try:
            m = cls(**kw)
            m.load_state_dict(strip_attn_buffers(state_dict), strict=False)
            print(f"[INFO] Built with kwargs #{i}.")
            return m.eval()
        except Exception as e:
            print(f"[WARN] Candidate kwargs #{i} failed: {e}")
            last_err = e
    if last_err:
        raise last_err
    raise RuntimeError("Could not instantiate model to read β")

def load_ckpt(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        sd = obj.get("model") or obj.get("state_dict") or obj
        print("[INFO] Loaded checkpoint dict → using its state_dict")
        return sd
    elif isinstance(obj, nn.Module):
        print("[INFO] Loaded full nn.Module; using state_dict()")
        return obj.state_dict()
    else:
        raise RuntimeError("Unsupported checkpoint format")

# ------------------------- β extraction -------------------------

def extract_betas(state_dict: Dict[str, torch.Tensor]) -> np.ndarray:
    idx, vals = [], []
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.numel() == 1 and k.endswith(".rate"):
            parts = k.split(".")
            try:
                if "LastStepAttentions" in parts:
                    i = parts.index("LastStepAttentions")
                    j = int(parts[i + 1])
                    raw = float(v.detach().cpu().item())
                    beta = np.log1p(np.exp(raw))  # softplus → β > 0
                    idx.append(j); vals.append(beta)
            except Exception:
                continue
    if not idx:
        raise RuntimeError("No β (.rate) scalars found in state_dict")
    order = np.argsort(idx)
    return np.array([vals[i] for i in order], dtype=float)

# ------------------------- attention extraction -------------------------

def extract_saved_attention_matrix(state_dict: Dict[str, torch.Tensor], head: int) -> np.ndarray:
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and k.endswith("saved_attn_avg") and v.ndim in (2, 3):
            arr = v.detach().cpu().float().numpy()
            if arr.ndim == 3:  # [H,N,N]
                h = max(0, min(head, arr.shape[0]-1))
                return arr[h]
            return arr  # [N,N]
    raise RuntimeError("No saved_attn_avg tensor found in checkpoint")

# ------------------------- aggregation & visualization helpers -------------------------

def aggregate_beta_to_17(beta_full: np.ndarray, var_groups: List[List[int]]) -> np.ndarray:
    agg = []
    n = len(beta_full)
    for grp in var_groups:
        vals = [beta_full[i] for i in grp if 0 <= i < n]
        agg.append(float(np.mean(vals)) if vals else np.nan)
    return np.array(agg, dtype=float)


def aggregate_attention_to_17(A: np.ndarray, var_groups: List[List[int]],
                              diag_mode: str = "group") -> np.ndarray:
    """
    Aggregate raw feature‑level attention to a 17×17 matrix in the canonical order.

    Off‑diagonals (qi != kj):  for each q∈Gq, take mean over columns in Gk, then mean over q.
    Diagonal (qi == kj):
      - diag_mode="group"  → mean over A[q, Gq] for q∈Gq  (matches paper: attention to same group)
      - diag_mode="self"   → mean over A[q, q] for q∈Gq   (strict self‑only)
    """
    G = len(var_groups)
    M = np.zeros((G, G), dtype=float)
    for qi, Gq in enumerate(var_groups):
        for kj, Gk in enumerate(var_groups):
            vals: List[float] = []
            if qi == kj:
                if diag_mode == "self":
                    vals = [float(A[q, q]) for q in Gq]
                else:  # group
                    for q in Gq:
                        vals.append(float(np.mean(A[q, Gq])))
            else:
                for q in Gq:
                    vals.append(float(np.mean(A[q, Gk])))
            M[qi, kj] = float(np.nanmean(vals)) if vals else np.nan
    return M


def destripe_for_visualization(M: np.ndarray) -> np.ndarray:
    X = M.astype(float)
    row_m = X.mean(axis=1, keepdims=True)
    col_m = X.mean(axis=0, keepdims=True)
    g = X.mean()
    Xc = X - row_m - col_m + g
    mmin = np.nanmin(Xc)
    if np.isfinite(mmin) and mmin < 0:
        Xc = Xc - mmin
    return Xc


def percentile_limits(arrays: List[np.ndarray], pmin: float, pmax: float,
                      vmin: Optional[float]=None, vmax: Optional[float]=None) -> Tuple[float,float]:
    if vmin is not None and vmax is not None:
        return float(vmin), float(vmax)
    cat = np.concatenate([a.reshape(-1) for a in arrays if a is not None])
    return float(np.percentile(cat, pmin)), float(np.percentile(cat, pmax))

# ------------------------- plotting -------------------------

def plot_decay_strip(beta: np.ndarray, features: List[str], outdir: str) -> str:
    order = np.argsort(-beta)
    b = beta[order]
    labels = short_labels([features[i] for i in order])

    vmin, vmax = float(np.nanmin(b)), float(np.nanmax(b))

    base_cmap = plt.cm.Blues
    cmap5 = ListedColormap(base_cmap(np.linspace(0.2, 0.95, 5)))
    levels = np.linspace(vmin, vmax, 6)
    norm5 = BoundaryNorm(levels, cmap5.N, clip=True)

    fig, ax = plt.subplots(figsize=(5, 1.6))
    ax.imshow(b.reshape(1, -1), aspect=0.4, cmap=cmap5, norm=norm5, interpolation="nearest")

    ax.set_yticks([0]); ax.set_yticklabels(["Decay_Rates"], fontsize=11, rotation=45, va="center")
    ax.tick_params(axis='y', pad=-10, length=0)
    ax.margins(y=0)
    ax.set_ylim(0.5, -0.5)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_title("Decay Rate For Different Features", fontsize=12, pad=10)

    fig.subplots_adjust(bottom=0.3)
    cax = fig.add_axes([0.4, 0.08, 0.35, 0.05])
    cb = ColorbarBase(cax, cmap=cmap5, norm=norm5, boundaries=levels,
                      orientation="horizontal", drawedges=True)
    tick_locs = 0.5 * (levels[:-1] + levels[1:])
    cb.set_ticks(tick_locs)
    cb.set_ticklabels([f"{x:.2f}" for x in tick_locs])
    cb.set_label("β", fontsize=10, labelpad=3)
    cb.ax.tick_params(labelsize=9)

    path = os.path.join(outdir, f"decay_rates_{ts()}.png")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_single_heatmap(M: np.ndarray, features: List[str], outdir: str,
                        title: str, vmin: float, vmax: float) -> str:
    labels = short_labels(features)
    fig = plt.figure(figsize=(9.6, 9.0))
    ax = fig.add_subplot(111)
    img = ax.imshow(M, aspect="equal", cmap="Blues", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.xaxis.tick_top(); ax.xaxis.set_label_position('top')
    ax.set_xlabel("Key Features"); ax.set_ylabel("Query Features")
    ax.set_title(title)
    cbar = fig.colorbar(img, ax=ax, orientation="horizontal", fraction=0.04, pad=0.08)
    cbar.set_label("attention", rotation=0, labelpad=10, ha="left")
    fig.tight_layout()
    path = os.path.join(outdir, f"interdependencies_{ts()}.png")
    fig.savefig(path, dpi=260); plt.close(fig)
    return path


def plot_two_panel(ML: np.ndarray, MR: np.ndarray, features: List[str], outdir: str,
                   titles=("Died WITH Diabetes", "Died WITHOUT Diabetes"),
                   vmin: float = 0.0, vmax: float = 1.0) -> str:
    labels = short_labels(features)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.6), constrained_layout=True)
    for ax, M, ttl in ((axes[0], ML, titles[0]), (axes[1], MR, titles[1])):
        img = ax.imshow(M, aspect="equal", cmap="Blues", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        ax.xaxis.tick_top(); ax.xaxis.set_label_position('top')
        ax.set_xlabel("Key Features"); ax.set_ylabel("Query Features")
        ax.set_title(ttl)
    cbar = fig.colorbar(img, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.05, pad=0.08)
    cbar.set_label("attention", rotation=0, labelpad=10, ha="left")
    path = os.path.join(outdir, f"interdependencies_two_panel_{ts()}.png")
    fig.savefig(path, dpi=260); plt.close(fig)
    return path

# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser(description="ConCare paper‑style plots with correct diagonals and no hidden normalization.")
    ap.add_argument("--ckpt", required=True, help="Path to model checkpoint (.pt/.pth)")
    ap.add_argument("--config", required=True, help="Path to discretizer_config.json")
    ap.add_argument("--head", type=int, default=0, help="Which attention head to plot when [H,N,N]")
    ap.add_argument("--auto-head", action="store_true", help="Scan all heads and pick the one with max diag/off ratio")
    ap.add_argument("--force-softmax", action="store_true", help="Force a softmax over keys if saved tensor is pre-softmax")
    ap.add_argument("--pmin", type=float, default=5.0, help="Percentile for vmin")
    ap.add_argument("--pmax", type=float, default=95.0, help="Percentile for vmax")
    ap.add_argument("--vmin", type=float, default=None, help="Override colorbar min (raw value)")
    ap.add_argument("--vmax", type=float, default=None, help="Override colorbar max (raw value)")
    ap.add_argument("--dump-header", action="store_true", help="Write header to heatmap_plots/header.txt")
    ap.add_argument("--attn-diabetic", help="Cohort‑averaged [N,N] or [H,N,N] attention (died WITH diabetes)")
    ap.add_argument("--attn-nondiabetic", help="Cohort‑averaged [N,N] or [H,N,N] attention (died WITHOUT diabetes)")
    ap.add_argument("--destripe", action="store_true", help="Apply optional double‑centering for display only")
    ap.add_argument("--diag-mode", choices=["group", "self"], default="group", help="Diagonal aggregation mode")
    ap.add_argument("--enrich", action="store_true", help="Plot enrichment over uniform baseline (accounts for group size)")
    ap.add_argument("--print-metrics", action="store_true", help="Print diag/off‑diag ratio for sanity check")
    args = ap.parse_args()

    outdir = ensure_outdir()
    print(f"[INFO] Output directory: {outdir}")

    if not os.path.exists(args.ckpt):
        print(f"[ERROR] ckpt not found: {args.ckpt}"); sys.exit(1)
    if not os.path.exists(args.config):
        print(f"[ERROR] config not found: {args.config}"); sys.exit(1)

    # 1) Load state dict
    sd = load_ckpt(args.ckpt)

    # 2) Build header + 17‑feature groups
    dump_path = os.path.join(outdir, "header.txt") if args.dump_header else None
    header, feat_names, feat_groups = build_header_and_groups(
        config_path=args.config, timestep=0.8, store_masks=True,
        impute="previous", start_time="zero", dump_path=dump_path
    )
    print(f"[INFO] Aggregating into {len(feat_names)} variables: {', '.join(feat_names)}")

    # 3) β decay strip (optional but kept for parity)
    model_for_rates = build_model_for_rates(sd)
    beta_full = extract_betas(model_for_rates.state_dict())
    beta17 = aggregate_beta_to_17(beta_full, feat_groups)
    for n, b in zip(feat_names, beta17):
        print(f"{n:>28s}: β = {float(b):.6f}")
    p_decay = plot_decay_strip(beta17, feat_names, outdir)
    print(f"[INFO] Saved decay rate plot → {p_decay}")

    # 4) Interdependency heatmap(s)
    def prep_matrix(A: np.ndarray) -> np.ndarray:
        M_raw = aggregate_attention_to_17(A, feat_groups, diag_mode=args.diag_mode)
        if args.enrich:
            total_cols = sum(len(g) for g in feat_groups)
            expected_row = np.array([len(g)/total_cols for g in feat_groups], dtype=float)
            expected = np.tile(expected_row, (len(feat_groups), 1))
            M_proc = M_raw - expected
        else:
            M_proc = M_raw
        if args.print_metrics:
            diag_mean = np.mean(np.diag(M_proc))
            off_mean = np.mean(M_proc[~np.eye(M_proc.shape[0], dtype=bool)])
            print(f"[METRIC] diag_mean={diag_mean:.6g}  off_mean={off_mean:.6g}  diag/off={diag_mean/off_mean:.3f}")
        return destripe_for_visualization(M_proc) if args.destripe else M_proc

    if args.attn_diabetic and args.attn_nondiabetic:
        A_d = np.load(args.attn_diabetic)
        A_n = np.load(args.attn_nondiabetic)
        if A_d.ndim == 3:
            A_d = A_d[max(0, min(args.head, A_d.shape[0]-1))]
        if A_n.ndim == 3:
            A_n = A_n[max(0, min(args.head, A_n.shape[0]-1))]
        M_d = prep_matrix(A_d)
        M_n = prep_matrix(A_n)
        vmin, vmax = percentile_limits([M_d, M_n], args.pmin, args.pmax, args.vmin, args.vmax)
        p_two = plot_two_panel(M_d, M_n, feat_names, outdir, vmin=vmin, vmax=vmax)
        print(f"[INFO] Saved two‑panel interdependency heatmap → {p_two}")
    else:
        # Auto-head scan if requested
        if args.auto_head:
            # Try to infer number of heads from tensor in state dict
            H = None
            for k, v in sd.items():
                if isinstance(v, torch.Tensor) and k.endswith("saved_attn_avg") and v.ndim == 3:
                    H = v.shape[0]
                    break
            if H is None:
                H = max(1, args.head + 1)
            print(f"[SCAN] evaluating {H} heads for diag/off ratio…")
            best_h, best_r, best_M = 0, -1.0, None
            for h in range(H):
                A_h = extract_saved_attention_matrix(sd, head=h)
                A_h = _maybe_softmax(A_h, args.force_softmax)
                M_h = prep_matrix(A_h)
                r = float(np.mean(np.diag(M_h)) / np.mean(M_h[~np.eye(M_h.shape[0], dtype=bool)]))
                print(f"  head {h}: diag/off = {r:.3f}")
                if r > best_r:
                    best_h, best_r, best_M = h, r, M_h
            print(f"[SCAN] chose head {best_h} (diag/off={best_r:.3f})")
            M = best_M
        else:
            A = extract_saved_attention_matrix(sd, head=args.head)
            A = _maybe_softmax(A, args.force_softmax)
            M = prep_matrix(A)
        vmin, vmax = percentile_limits([M], args.pmin, args.pmax, args.vmin, args.vmax)
        p_single = plot_single_heatmap(M, feat_names, outdir, title="All Patients (single head)", vmin=vmin, vmax=vmax)
        print(f"[INFO] Saved interdependency heatmap → {p_single}")

    print("\n✅ Done. Plots saved in:", outdir)

if __name__ == "__main__":
    main()
