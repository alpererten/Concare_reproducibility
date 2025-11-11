#!/usr/bin/env python3
"""
concare_plots_fresh.py
======================

Minimal, clean ConCare paper-style plots with robust defaults.

Outputs:
  - heatmap_plots/decay_rates_<ts>.png
  - heatmap_plots/interdependencies_<ts>.png   (single)  OR
  - heatmap_plots/interdependencies_two_panel_<ts>.png   (if two .npy matrices provided)

Key design choices (fixing past issues):
  • Use 'id_to_channel' from discretizer_config.json as the canonical 17-feature order.
  • Aggregate attention with NO group-size bias and TRUE self-pair diagonals.
  • Remove vertical striping via double-centering (visual-only), NOT row/col normalization.
  • Use tight percentile color limits (pmin/pmax) to mimic paper contrast without altering values.

CLI:
  python concare_plots_fresh.py \
      --ckpt trained_models/best_concare.pt \
      --config data/discretizer_config.json \
      --head 0 \
      --pmin 5 --pmax 95 \
      [--dump-header] \
      [--attn-diabetic heatmap_plots/attn_diab.npy] \
      [--attn-nondiabetic heatmap_plots/attn_nondiab.npy]
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


# ------------------------- utils -------------------------

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
    return order  # 17 names in canonical order

def import_discretizer():
    # Your project already has data_processing.Discretizer
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
    """
    Build final input header via Discretizer, then group to the 17 channels from config.
    We match columns to each target channel by relaxed string rules on the normalized base.
    Mask columns are excluded.
    """
    target_order = load_feature_order(config_path)  # 17 paper features
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
            # loose containment match on normalized base
            if v in h_norm:
                idxs.append(i)
        if idxs:
            groups.append(idxs)
            names.append(var)
            used.update(idxs)

    # diagnostics
    for var, idxs in zip(names, groups):
        print(f"[MAP] {var:>28s}  <--  {len(idxs)} cols")

    missing = [v for v in target_order if v not in names]
    if missing:
        print(f"[WARN] These variables had no matches and will be omitted: {', '.join(missing)}")

    return header, names, groups


# ------------------------- checkpoint readers -------------------------

def import_model_module():
    # Your ConCare model lives under model_codes.ConCare_Model_v3
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
    """We only need the module to read β (.rate). Attention is read directly from state_dict."""
    mod = import_model_module()
    cls = pick_model_class(mod)
    # attempt empty init first
    try:
        m = cls()
        m.load_state_dict(strip_attn_buffers(state_dict), strict=False)
        print("[INFO] Built model with empty constructor")
        return m.eval()
    except Exception:
        pass
    # very small set of candidate kwargs (harmless if they don't match)
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

def extract_betas(state_dict: Dict[str, torch.Tensor]) -> np.ndarray:
    """
    β comes from LastStepAttentions[i].rate (scalar tensors).
    We collect them in index order.
    """
    idx, vals = [], []
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.numel() == 1 and k.endswith(".rate"):
            parts = k.split(".")
            # Expect ... LastStepAttentions.<idx>.rate
            try:
                if "LastStepAttentions" in parts:
                    i = parts.index("LastStepAttentions")
                    j = int(parts[i+1])
                    idx.append(j); vals.append(float(v.detach().cpu().item()))
            except Exception:
                continue
    if not idx:
        raise RuntimeError("No β (.rate) scalars found in state_dict")
    order = np.argsort(idx)
    return np.array([vals[i] for i in order], dtype=float)

def extract_saved_attention_matrix(state_dict: Dict[str, torch.Tensor], head: int) -> np.ndarray:
    """
    Returns a single-head [N,N] attention matrix.
    Prefers *.saved_attn_avg keys in the raw state_dict.
    """
    # Direct lookup in state_dict
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and k.endswith("saved_attn_avg") and v.ndim in (2, 3):
            arr = v.detach().cpu().float().numpy()
            if arr.ndim == 3:  # [H,N,N]
                h = max(0, min(head, arr.shape[0]-1))
                return arr[h]
            return arr  # [N,N]
    raise RuntimeError("No saved_attn_avg tensor found in checkpoint")


# ------------------------- aggregation & de-striping -------------------------

def aggregate_beta_to_17(beta_full: np.ndarray, var_groups: List[List[int]]) -> np.ndarray:
    agg = []
    n = len(beta_full)
    for grp in var_groups:
        vals = [beta_full[i] for i in grp if 0 <= i < n]
        agg.append(float(np.mean(vals)) if vals else np.nan)
    return np.array(agg, dtype=float)

def aggregate_attention_to_17(A: np.ndarray, var_groups: List[List[int]]) -> np.ndarray:
    """
    Raw feature-level aggregation without normalization:
      - Off-diagonals (Gq!=Gk): for each q in Gq, take mean over columns in Gk; then mean across q.
        (no group-size bias)
      - Diagonal (Gq==Gk): mean of true self-pairs A[q,q] for q in Gq.
    Output is [G,G] in the 17-feature order.
    """
    G = len(var_groups)
    M = np.zeros((G, G), dtype=float)
    for qi, Gq in enumerate(var_groups):
        for kj, Gk in enumerate(var_groups):
            if qi == kj:
                vals = [float(A[q, q]) for q in Gq] if Gq else [np.nan]
            else:
                vals = []
                for q in Gq:
                    vals.append(float(np.mean(A[q, Gk])))
            M[qi, kj] = float(np.nanmean(vals)) if len(vals) else np.nan
    return M  # raw magnitudes

def destripe_for_visualization(M: np.ndarray) -> np.ndarray:
    """
    Double-centering to remove row/column baseline effects (visual only),
    preserving interaction structure and strong diagonals:
      M' = M - row_mean - col_mean + global_mean
    Then shift so min >= 0 to get a nonnegative display range.
    """
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
    fig = plt.figure(figsize=(14, 3.0))
    ax = fig.add_subplot(111)
    img = ax.imshow(b.reshape(1, -1), aspect="auto", vmin=vmin, vmax=vmax, cmap="Blues")
    ax.set_yticks([0]); ax.set_yticklabels(["Decay_Rates"])
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title("Decay Rates For Different Features")
    cbar = fig.colorbar(img, ax=ax, orientation="horizontal", fraction=0.04, pad=0.15)
    cbar.set_label("β", rotation=0, labelpad=10, ha="left")
    fig.tight_layout()
    path = os.path.join(outdir, f"decay_rates_{ts()}.png")
    fig.savefig(path, dpi=220); plt.close(fig)
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
    ap = argparse.ArgumentParser(description="Fresh ConCare paper-style plots (clean + robust).")
    ap.add_argument("--ckpt", required=True, help="Path to model checkpoint (.pt/.pth)")
    ap.add_argument("--config", required=True, help="Path to discretizer_config.json")
    ap.add_argument("--head", type=int, default=0, help="Which attention head to plot when [H,N,N]")
    ap.add_argument("--pmin", type=float, default=5.0, help="Percentile for vmin")
    ap.add_argument("--pmax", type=float, default=95.0, help="Percentile for vmax")
    ap.add_argument("--vmin", type=float, default=None, help="Override colorbar min (raw value)")
    ap.add_argument("--vmax", type=float, default=None, help="Override colorbar max (raw value)")
    ap.add_argument("--dump-header", action="store_true", help="Write header to heatmap_plots/header.txt")
    ap.add_argument("--attn-diabetic", help="Cohort-averaged [N,N] or [H,N,N] attention (died WITH diabetes)")
    ap.add_argument("--attn-nondiabetic", help="Cohort-averaged [N,N] or [H,N,N] attention (died WITHOUT diabetes)")
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"[ERROR] ckpt not found: {args.ckpt}"); sys.exit(1)
    if not os.path.exists(args.config):
        print(f"[ERROR] config not found: {args.config}"); sys.exit(1)

    outdir = ensure_outdir()
    print(f"[INFO] Output directory: {outdir}")

    # 1) Load state dict from checkpoint
    sd = load_ckpt(args.ckpt)

    # 2) Build header + 17-feature groups (using discretizer to reproduce columns)
    dump_path = os.path.join(outdir, "header.txt") if args.dump_header else None
    header, feat_names, feat_groups = build_header_and_groups(
        config_path=args.config, timestep=0.8, store_masks=True,
        impute="previous", start_time="zero", dump_path=dump_path
    )
    print(f"[INFO] Aggregating into {len(feat_names)} variables: {', '.join(feat_names)}")

    # 3) β decay strip
    #    Build a small model instance only to ensure βs exist in a compatible state_dict
    model_for_rates = build_model_for_rates(sd)
    beta_full = extract_betas(model_for_rates.state_dict())
    beta17 = aggregate_beta_to_17(beta_full, feat_groups)
    for n, b in zip(feat_names, beta17):
        print(f"{n:>28s}: β = {float(b):.6f}")
    p_decay = plot_decay_strip(beta17, feat_names, outdir)
    print(f"[INFO] Saved decay rate plot → {p_decay}")

    # 4) Interdependency heatmap(s)
    #    Use provided cohort matrices if both given; else single-panel from ckpt
    if args.attn_diabetic and args.attn_nondiabetic:
        A_d = np.load(args.attn_diabetic)
        A_n = np.load(args.attn_nondiabetic)
        if A_d.ndim == 3:  # [H,N,N]
            A_d = A_d[max(0, min(args.head, A_d.shape[0]-1))]
        if A_n.ndim == 3:
            A_n = A_n[max(0, min(args.head, A_n.shape[0]-1))]
        M_d_raw = aggregate_attention_to_17(A_d, feat_groups)
        M_n_raw = aggregate_attention_to_17(A_n, feat_groups)
        # Visual de-striping (keeps diagonals strong)
        M_d_viz = destripe_for_visualization(M_d_raw)
        M_n_viz = destripe_for_visualization(M_n_raw)
        vmin, vmax = percentile_limits([M_d_viz, M_n_viz], args.pmin, args.pmax, args.vmin, args.vmax)
        p_two = plot_two_panel(M_d_viz, M_n_viz, feat_names, outdir, vmin=vmin, vmax=vmax)
        print(f"[INFO] Saved two-panel interdependency heatmap → {p_two}")
    else:
        A = extract_saved_attention_matrix(sd, head=args.head)  # [N,N] single head
        M_raw = aggregate_attention_to_17(A, feat_groups)
        M_viz = destripe_for_visualization(M_raw)
        vmin, vmax = percentile_limits([M_viz], args.pmin, args.pmax, args.vmin, args.vmax)
        p_single = plot_single_heatmap(M_viz, feat_names, outdir, title="All Patients (single head)", vmin=vmin, vmax=vmax)
        print(f"[INFO] Saved interdependency heatmap → {p_single}")

    print("\n✅ Done. Plots saved in:", outdir)


if __name__ == "__main__":
    main()
