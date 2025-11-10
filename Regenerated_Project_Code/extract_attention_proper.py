#!/usr/bin/env python3
"""
Extract attention from trained ConCare model and generate plots.

IMPORTANT: Uses model with attention capture hooks.
Extracts attention in EVAL MODE (dropout disabled).
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.getcwd())
from model_codes.ConCare_Model_v3B import ConCare
from train_helpers import RAMDataset, pad_collate
from torch.utils.data import DataLoader


def extract_attention_eval_mode(model, dataloader, device, head_idx=0, max_samples=None):
    """
    Extract attention in EVAL MODE.
    
    Key points:
    - model.eval() disables dropout
    - Captures actual attention weights from model's forward pass
    - Returns attention BEFORE dropout is applied
    """
    model.eval()  # CRITICAL: disables dropout
    
    all_attention = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (X, D, y) in enumerate(dataloader):
            if max_samples and len(all_labels) >= max_samples:
                break
            
            X = X.to(device)
            D = D.to(device)
            
            # Forward pass (attention gets captured in model.MultiHeadedAttention.last_attention_weights)
            _ = model(X, D)
            
            # Extract captured attention [B, num_heads, N, N]
            attn = model.MultiHeadedAttention.last_attention_weights
            
            if attn is None:
                raise RuntimeError("Attention not captured! Make sure you're using ConCare_Model_v3_with_attention.py")
            
            # Get specified head
            attn_head = attn[:, head_idx, :, :].cpu().numpy()  # [B, N, N]
            
            for i in range(attn_head.shape[0]):
                all_attention.append(attn_head[i])
                all_labels.append(y[i].item())
            
            if batch_idx % 10 == 0:
                print(f"  Processed {len(all_labels)} samples...")
    
    return np.array(all_attention), np.array(all_labels)


def normalize_attention_for_viz(M, method='row_col'):
    """
    Normalize attention for visualization.
    
    Methods:
    - 'row_col': Row + column normalization (removes vertical banding)
    - 'row': Row normalization only (softmax already applied)
    - 'none': No normalization (raw attention weights)
    """
    if method == 'none':
        return M.copy()
    
    M = M.copy().astype(float)
    
    if method in ['row', 'row_col']:
        rs = M.sum(axis=1, keepdims=True) + 1e-12
        M = M / rs
    
    if method == 'row_col':
        cs = M.sum(axis=0, keepdims=True) + 1e-12
        M = M / cs
    
    return M


def plot_two_panel(M_left, M_right, feature_labels, save_path, 
                   titles=("Died WITH Diabetes", "Died WITHOUT Diabetes"),
                   normalization='row_col', pmin=5, pmax=95):
    """Create two-panel heatmap."""
    
    # Apply normalization
    M_left = normalize_attention_for_viz(M_left, method=normalization)
    M_right = normalize_attention_for_viz(M_right, method=normalization)
    
    # Compute shared color limits
    cat = np.concatenate([M_left.reshape(-1), M_right.reshape(-1)])
    vmin = np.percentile(cat, pmin)
    vmax = np.percentile(cat, pmax)
    
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.3), constrained_layout=True)
    
    for ax, M, title in [(axes[0], M_left, titles[0]), (axes[1], M_right, titles[1])]:
        img = ax.imshow(M, aspect="equal", cmap="Blues", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(feature_labels)))
        ax.set_xticklabels(feature_labels, rotation=45, ha="left")
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.set_yticks(range(len(feature_labels)))
        ax.set_yticklabels(feature_labels)
        ax.set_xlabel("Key Features")
        ax.set_ylabel("Query Features")
        ax.set_title(title)
    
    cbar = fig.colorbar(img, ax=axes.ravel().tolist(), 
                        orientation="horizontal", fraction=0.05, pad=0.08)
    cbar.set_label("attention", rotation=0, labelpad=10, ha="left")
    cbar.set_ticks([float(x) for x in np.linspace(vmin, vmax, 6)])
    
    fig.savefig(save_path, dpi=260, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_single_panel(M, feature_labels, save_path, title="Cross-Feature Attention",
                      normalization='row_col', pmin=5, pmax=95):
    """Create single-panel heatmap."""
    
    # Apply normalization
    M = normalize_attention_for_viz(M, method=normalization)
    
    # Compute color limits
    vmin = np.percentile(M.reshape(-1), pmin)
    vmax = np.percentile(M.reshape(-1), pmax)
    
    fig = plt.figure(figsize=(9.5, 8.8))
    ax = fig.add_subplot(111)
    
    img = ax.imshow(M, aspect="equal", cmap="Blues", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(feature_labels)))
    ax.set_xticklabels(feature_labels, rotation=45, ha="left")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_yticks(range(len(feature_labels)))
    ax.set_yticklabels(feature_labels)
    ax.set_xlabel("Key Features")
    ax.set_ylabel("Query Features")
    ax.set_title(title)
    
    cbar = fig.colorbar(img, ax=ax, orientation="horizontal", 
                        fraction=0.035, pad=0.08)
    cbar.set_label("attention", rotation=0, labelpad=10, ha="left")
    cbar.set_ticks([float(x) for x in np.linspace(vmin, vmax, 6)])
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=260, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def compute_diagonal_ratio(M):
    """Compute ratio of diagonal to off-diagonal attention."""
    N = M.shape[0]
    diag = np.diag(M).mean()
    off_diag = (M.sum() - np.diag(M).sum()) / (N * (N - 1))
    return diag / (off_diag + 1e-9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default='data/normalized_data_cache_train')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--input_dim', type=int, default=None,
                       help='Input dimension (auto-detected if not specified)')
    parser.add_argument('--head', type=int, default=0)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='heatmap_plots')
    parser.add_argument('--normalization', type=str, default='row_col',
                       choices=['none', 'row', 'row_col'],
                       help='Normalization for visualization: none, row, or row_col')
    parser.add_argument('--pmin', type=float, default=5.0,
                       help='Percentile for vmin (default 5)')
    parser.add_argument('--pmax', type=float, default=95.0,
                       help='Percentile for vmax (default 95)')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("ATTENTION EXTRACTION (EVAL MODE)")
    print("="*60)
    print(f"Normalization: {args.normalization}")
    print(f"Color scale: {args.pmin}th to {args.pmax}th percentile")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    
    # Load checkpoint first to infer input_dim if needed
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    
    # Auto-detect input_dim from checkpoint if not specified
    if args.input_dim is None:
        # Count number of GRU modules in state dict
        gru_count = sum(1 for k in state_dict.keys() if k.startswith('GRUs.') and '.weight_ih_l0' in k)
        if gru_count > 0:
            args.input_dim = gru_count
            print(f"Auto-detected input_dim: {args.input_dim} (from {gru_count} GRU modules)")
        else:
            raise ValueError("Could not auto-detect input_dim. Please specify --input_dim")
    
    # Auto-detect hidden_dim from checkpoint
    # GRU weight_hh_l0 has shape [3*hidden_dim, hidden_dim]
    gru0_key = 'GRUs.0.weight_hh_l0'
    if gru0_key in state_dict:
        hidden_dim = state_dict[gru0_key].shape[1]
        print(f"Auto-detected hidden_dim: {hidden_dim}")
    else:
        hidden_dim = 128
        print(f"Could not auto-detect hidden_dim, using default: {hidden_dim}")
    
    # d_model should equal hidden_dim
    d_model = hidden_dim
    
    model = ConCare(
        input_dim=args.input_dim,
        hidden_dim=hidden_dim,
        d_model=d_model,
        MHD_num_head=4,
        d_ff=256,
        output_dim=1,
        keep_prob=0.5,
        demographic_dim=12
    )
    
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
        print(f"✓ Loaded checkpoint from epoch {ckpt.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(ckpt)
        print(f"✓ Loaded checkpoint")
    
    model.to(args.device)
    model.eval()  # CRITICAL: eval mode
    
    # Verify dropout is disabled
    print("\nVerifying dropout state:")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            status = "✓ DISABLED" if not module.training else "✗ ACTIVE"
            print(f"  {name}: {status}")
    
    # Load data
    print(f"\nLoading {args.split} data...")
    dataset = RAMDataset(args.split, cache_dir=args.cache_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                           shuffle=False, collate_fn=pad_collate)
    
    print(f"Dataset: {len(dataset)} samples")
    
    # Extract attention
    print(f"\nExtracting attention (head {args.head}, eval mode)...")
    attention_weights, labels = extract_attention_eval_mode(
        model, dataloader, args.device, 
        head_idx=args.head,
        max_samples=args.max_samples
    )
    
    print(f"\n✓ Extracted attention from {len(labels)} samples")
    print(f"  Shape: {attention_weights.shape}")
    print(f"  Died: {(labels == 1).sum()}")
    print(f"  Survived: {(labels == 0).sum()}")
    
    # Feature labels
    feature_labels = [
        'Cl', 'CO2CP', 'WBC', 'Hb', 'Urea', 'Ca', 'K', 'Na', 
        'Scr', 'P', 'Albumin', 'hs-CRP', 'Glucose', 'Appetite',
        'Weight', 'SBP', 'DBP', 'Base'
    ]
    
    n_features = attention_weights.shape[1]
    if len(feature_labels) > n_features:
        feature_labels = feature_labels[:n_features]
    elif len(feature_labels) < n_features:
        for i in range(len(feature_labels), n_features):
            feature_labels.append(f'F{i+1}')
    
    # Compute diagonal ratios
    print(f"\nDiagonal ratios (raw attention):")
    avg_all = attention_weights.mean(axis=0)
    ratio_all = compute_diagonal_ratio(avg_all)
    print(f"  All patients: {ratio_all:.3f}×")
    
    # Save raw data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    np_save_path = os.path.join(args.output_dir, f'attention_raw_head{args.head}_{args.split}.npz')
    np.savez_compressed(
        np_save_path,
        attention_weights=attention_weights,
        labels=labels,
        feature_labels=np.array(feature_labels)
    )
    print(f"\n✓ Saved: {np_save_path}")
    
    # Generate plots
    print(f"\n{'='*60}")
    print("GENERATING PLOTS")
    print(f"{'='*60}")
    
    # 1. All patients
    print("\n1. All patients...")
    plot_single_panel(
        avg_all, feature_labels,
        os.path.join(args.output_dir, f'attention_all_head{args.head}_{timestamp}.png'),
        title=f"All Patients (Head {args.head})",
        normalization=args.normalization,
        pmin=args.pmin, pmax=args.pmax
    )
    
    # 2. Died vs Survived
    died_mask = labels == 1
    survived_mask = labels == 0
    
    if died_mask.sum() > 0 and survived_mask.sum() > 0:
        print("\n2. Died vs Survived...")
        avg_died = attention_weights[died_mask].mean(axis=0)
        avg_survived = attention_weights[survived_mask].mean(axis=0)
        
        ratio_died = compute_diagonal_ratio(avg_died)
        ratio_survived = compute_diagonal_ratio(avg_survived)
        print(f"  Died diagonal ratio: {ratio_died:.3f}×")
        print(f"  Survived diagonal ratio: {ratio_survived:.3f}×")
        
        plot_two_panel(
            avg_died, avg_survived, feature_labels,
            os.path.join(args.output_dir, f'attention_died_vs_survived_head{args.head}_{timestamp}.png'),
            titles=("Died", "Survived"),
            normalization=args.normalization,
            pmin=args.pmin, pmax=args.pmax
        )
        
        # Save for reuse
        np.save(os.path.join(args.output_dir, f'attn_died_head{args.head}.npy'), avg_died)
        np.save(os.path.join(args.output_dir, f'attn_survived_head{args.head}.npy'), avg_survived)
    
    # 3. Random split demo
    if died_mask.sum() > 10:
        print("\n3. Random split of died patients (demo)...")
        died_indices = np.where(died_mask)[0]
        np.random.shuffle(died_indices)
        
        mid = len(died_indices) // 2
        avg_g1 = attention_weights[died_indices[:mid]].mean(axis=0)
        avg_g2 = attention_weights[died_indices[mid:]].mean(axis=0)
        
        ratio_g1 = compute_diagonal_ratio(avg_g1)
        ratio_g2 = compute_diagonal_ratio(avg_g2)
        print(f"  Group 1 diagonal ratio: {ratio_g1:.3f}×")
        print(f"  Group 2 diagonal ratio: {ratio_g2:.3f}×")
        
        plot_two_panel(
            avg_g1, avg_g2, feature_labels,
            os.path.join(args.output_dir, f'attention_random_split_head{args.head}_{timestamp}.png'),
            titles=(f"Group 1 (n={mid})", f"Group 2 (n={len(died_indices)-mid})"),
            normalization=args.normalization,
            pmin=args.pmin, pmax=args.pmax
        )
    
    print(f"\n{'='*60}")
    print("✓ DONE")
    print(f"{'='*60}")
    print(f"\nOutputs in: {args.output_dir}/")


if __name__ == '__main__':
    main()