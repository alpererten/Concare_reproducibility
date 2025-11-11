import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from matplotlib.ticker import FuncFormatter

def percentile_limits(M, pmin, pmax):
    lo = np.percentile(M, pmin)
    hi = np.percentile(M, pmax)
    return float(lo), float(hi)

def build_groups_from_config(config_path: str, N: int):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    order = [
        "Capillary refill rate", "Diastolic blood pressure", "Fraction inspired oxygen",
        "Glascow coma scale eye opening", "Glascow coma scale motor response",
        "Glascow coma scale total", "Glascow coma scale verbal response",
        "Glucose", "Heart Rate", "Height", "Mean blood pressure",
        "Oxygen saturation", "Respiratory rate", "Systolic blood pressure",
        "Temperature", "Weight", "pH"
    ]
    groups = [[] for _ in order]
    if isinstance(cfg.get("id_to_channel"), list):
        names = cfg["id_to_channel"][:N]
    else:
        names = [f"f{i+1}" for i in range(N)]
    for i, nm in enumerate(names):
        for gi, key in enumerate(order):
            if key in nm:
                groups[gi].append(i)
                break
    groups = [g for g in groups if len(g) > 0]
    names = order[:len(groups)]
    return groups, names

def aggregate_to_groups(M: np.ndarray, groups):
    G = len(groups)
    out = np.zeros((G, G))
    for i, gi in enumerate(groups):
        for j, gj in enumerate(groups):
            out[i, j] = M[np.ix_(gi, gj)].mean()
    return out

def _shorten_labels(names):
    mapping = {
        "Capillary refill rate": "Cap refill",
        "Diastolic blood pressure": "DBP",
        "Fraction inspired oxygen": "FiO2",
        "Glascow coma scale eye opening": "GCS-E",
        "Glascow coma scale motor response": "GCS-M",
        "Glascow coma scale total": "GCS-Total",
        "Glascow coma scale verbal response": "GCS-V",
        "Glucose": "Glucose",
        "Heart Rate": "HR",
        "Height": "Height",
        "Mean blood pressure": "MAP",
        "Oxygen saturation": "SpO2",
        "Respiratory rate": "RR",
        "Systolic blood pressure": "SBP",
        "Temperature": "Temp",
        "Weight": "Weight",
        "pH": "pH",
    }
    out = []
    for n in names:
        out.append(mapping.get(n, n))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--attn', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--head', type=int, default=0)
    ap.add_argument('--pmin', type=float, default=5)
    ap.add_argument('--pmax', type=float, default=95)
    ap.add_argument('--outdir', default='heatmap_plots')
    args = ap.parse_args()

    attn = np.load(args.attn)
    M = attn[args.head]

    groups, names = build_groups_from_config(args.config, M.shape[0])
    M = aggregate_to_groups(M, groups)

    # Double centering but keep positive (visual-only)
    rmean = M.mean(axis=1, keepdims=True)
    cmean = M.mean(axis=0, keepdims=True)
    gmean = float(M.mean())
    M = M - rmean - cmean + gmean
    M -= M.min()

    vmin, vmax = percentile_limits(M, args.pmin, args.pmax)

    short = _shorten_labels(names)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    outfile = os.path.join(args.outdir, f'head{args.head}_paperstyle_{ts}.png')

    # ---- Paper-style layout ----
    fig = plt.figure(figsize=(6.5, 6.7))
    ax = fig.add_axes([0.12, 0.20, 0.78, 0.68])

    im = ax.imshow(M, cmap='Blues', vmin=vmin, vmax=vmax, aspect='equal', interpolation='nearest')

    # Put x-axis ticks on top like the paper
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    ax.set_xticks(np.arange(len(short)))
    ax.set_yticks(np.arange(len(short)))
    ax.set_xticklabels(short, rotation=40, ha='left', fontsize=10)
    ax.set_yticklabels(short, fontsize=10)

    # Axis labels and title to match the reference figure
    ax.set_xlabel('Key Features', fontsize=13)
    ax.set_ylabel('Query Features', fontsize=13)
    ax.set_title('All Patients (single head)', pad=28, fontsize=14)

    # Neat frame and tick params
    ax.tick_params(axis='both', length=0)

    # Colorbar along the bottom
    cax = fig.add_axes([0.15, 0.11, 0.7, 0.035])

    # Formatter to show x * 1e3 (e.g., 0.00075 -> 0.75)
    scale = 1e3
    fmt = FuncFormatter(lambda x, pos: f"{x*scale:.2f}")

    cb = plt.colorbar(im, cax=cax, orientation='horizontal', format=fmt)
    cb.set_label(r'Attention ($\times 10^{-3}$)', fontsize=12)

    for t in cb.ax.get_xticklabels():
        t.set_fontsize(12)

    fig.savefig(outfile, dpi=220)
    print(f'[INFO] Saved {outfile}')

if __name__ == '__main__':
    main()
