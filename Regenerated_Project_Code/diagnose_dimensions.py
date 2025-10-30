"""
Diagnostic script to identify dimension mismatch issues in ConCare training
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import torch
from datetime import datetime

def check_normalizer():
    """Check normalizer dimensions"""
    print("=" * 60)
    print("NORMALIZER CHECK")
    print("=" * 60)
    
    normalizer_path = "data/ihm_normalizer"
    if os.path.exists(normalizer_path):
        with open(normalizer_path, "rb") as f:
            data = f.read()
        try:
            state = pickle.loads(data, encoding="latin1")
        except TypeError:
            state = pickle.loads(data)
        
        means = np.asarray(state["means"]).astype(np.float32)
        stds = np.asarray(state["stds"]).astype(np.float32)
        
        print(f"Normalizer expects {means.shape[0]} features")
        print(f"Means shape: {means.shape}")
        print(f"Stds shape: {stds.shape}")
        return means.shape[0]
    else:
        print("✗ Normalizer file not found")
        return None

def check_raw_timeseries():
    """Check raw timeseries files"""
    print("\n" + "=" * 60)
    print("RAW TIMESERIES CHECK")
    print("=" * 60)
    
    # Check a sample timeseries file
    listfile = pd.read_csv("data/train_listfile.csv")
    first_stay = listfile.iloc[0]["stay"]
    ts_path = f"data/train/{first_stay}"
    
    if os.path.exists(ts_path):
        ts_df = pd.read_csv(ts_path)
        print(f"Sample file: {first_stay}")
        print(f"  Raw shape: {ts_df.shape}")
        print(f"  Raw columns ({len(ts_df.columns)}): {list(ts_df.columns)}")
        
        # Check for NaN values
        nan_counts = ts_df.isna().sum()
        print(f"\n  NaN counts per column:")
        for col, count in nan_counts.items():
            if count > 0:
                print(f"    {col}: {count}/{len(ts_df)} ({count/len(ts_df)*100:.1f}%)")
        
        return ts_df.columns.tolist()
    else:
        print(f"✗ File not found: {ts_path}")
        return None

def test_discretization():
    """Test the discretization process"""
    print("\n" + "=" * 60)
    print("DISCRETIZATION TEST")
    print("=" * 60)
    
    from data_preprocessing import discretize_timeseries
    
    # Load a sample file
    listfile = pd.read_csv("data/train_listfile.csv")
    first_stay = listfile.iloc[0]["stay"]
    ts_df = pd.read_csv(f"data/train/{first_stay}")
    
    print(f"Before discretization:")
    print(f"  Shape: {ts_df.shape}")
    print(f"  Columns: {len(ts_df.columns)}")
    
    # Apply discretization
    X, cols = discretize_timeseries(ts_df, timestep=1.0)
    
    print(f"\nAfter discretization:")
    print(f"  Shape: {X.shape}")
    print(f"  Features: {len(cols)}")
    print(f"  Feature names: {cols}")
    
    # Check for NaN values in output
    nan_mask = np.isnan(X)
    if nan_mask.any():
        print(f"\n  ⚠️ WARNING: Output contains {nan_mask.sum()} NaN values!")
        print(f"  NaN percentage: {nan_mask.sum() / X.size * 100:.2f}%")
    
    return X, cols

def test_full_pipeline():
    """Test the complete data processing pipeline"""
    print("\n" + "=" * 60)
    print("FULL PIPELINE TEST")
    print("=" * 60)
    
    from data_preprocessing import ConcareEpisodeDataset, Normalizer, pad_collate
    from torch.utils.data import DataLoader
    
    # Create dataset
    normalizer = Normalizer("data/ihm_normalizer")
    train_ds = ConcareEpisodeDataset("train", normalizer=normalizer)
    
    print(f"Dataset created with {len(train_ds)} samples")
    print(f"Expected features: {len(train_ds.feature_names)}")
    print(f"Feature names: {train_ds.feature_names}")
    
    # Get a single sample
    X, D, y = train_ds[0]
    print(f"\nSingle sample shapes:")
    print(f"  X (timeseries): {X.shape}")
    print(f"  D (demographics): {D.shape}")
    print(f"  y (label): {y.shape}")
    
    # Check for NaN/Inf values
    if torch.isnan(X).any():
        print(f"  ⚠️ X contains NaN values!")
    if torch.isinf(X).any():
        print(f"  ⚠️ X contains Inf values!")
    if torch.isnan(D).any():
        print(f"  ⚠️ D contains NaN values!")
        
    # Create a small batch
    loader = DataLoader(train_ds, batch_size=4, collate_fn=pad_collate)
    batch = next(iter(loader))
    X_batch, D_batch, y_batch = batch
    
    print(f"\nBatch shapes (batch_size=4):")
    print(f"  X: {X_batch.shape}")
    print(f"  D: {D_batch.shape}")
    print(f"  y: {y_batch.shape}")
    
    # Check batch statistics
    print(f"\nBatch statistics:")
    print(f"  X min: {X_batch.min():.4f}, max: {X_batch.max():.4f}, mean: {X_batch.mean():.4f}")
    print(f"  D min: {D_batch.min():.4f}, max: {D_batch.max():.4f}, mean: {D_batch.mean():.4f}")
    
    return X_batch, D_batch, y_batch

def test_model_forward():
    """Test model forward pass with actual data"""
    print("\n" + "=" * 60)
    print("MODEL FORWARD PASS TEST")
    print("=" * 60)
    
    from data_preprocessing import ConcareEpisodeDataset, Normalizer, pad_collate
    from model_codes.ConCare_Model_v1 import ConCare
    from torch.utils.data import DataLoader
    import torch
    
    # Get data
    normalizer = Normalizer("data/ihm_normalizer")
    train_ds = ConcareEpisodeDataset("train", normalizer=normalizer)
    loader = DataLoader(train_ds, batch_size=32, collate_fn=pad_collate)
    
    X_batch, D_batch, y_batch = next(iter(loader))
    
    print(f"Input shapes:")
    print(f"  X: {X_batch.shape}")
    print(f"  D: {D_batch.shape}")
    
    # Create model with correct input dim
    input_dim = X_batch.shape[2]
    model = ConCare(
        input_dim=input_dim, 
        hidden_dim=64, 
        d_model=64,
        MHD_num_head=4, 
        d_ff=128, 
        output_dim=1,
        keep_prob=0.5, 
        demographic_dim=12
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
        X_batch = X_batch.cuda()
        D_batch = D_batch.cuda()
        y_batch = y_batch.cuda()
    
    # Forward pass
    print(f"\nTesting forward pass...")
    try:
        model.eval()
        with torch.no_grad():
            logits, decov = model(X_batch, D_batch)
        print(f"✓ Forward pass successful")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Decov loss: {decov.item():.4f}")
        
        # Compute loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_batch)
        print(f"  BCE loss: {loss.item():.4f}")
        
        if torch.isnan(loss):
            print(f"  ⚠️ WARNING: Loss is NaN!")
            print(f"  Logits stats - min: {logits.min():.4f}, max: {logits.max():.4f}, mean: {logits.mean():.4f}")
            
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    print(f"\nConCare Data Dimension Diagnostic")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check normalizer expectations
    expected_features = check_normalizer()
    
    # Check raw data
    raw_columns = check_raw_timeseries()
    
    # Test discretization
    X_disc, disc_features = test_discretization()
    
    # Test full pipeline
    test_full_pipeline()
    
    # Test model
    test_model_forward()
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    if expected_features and X_disc is not None:
        actual_features = X_disc.shape[1]
        if expected_features != actual_features:
            print(f"⚠️ DIMENSION MISMATCH DETECTED!")
            print(f"  Normalizer expects: {expected_features} features")
            print(f"  Discretization produces: {actual_features} features")
            print(f"\n  This is likely causing the NaN losses and slow training.")
            print(f"\n  SOLUTION: The normalizer needs to match the data processing pipeline.")
        else:
            print(f"✓ Dimensions match: {expected_features} features")

if __name__ == "__main__":
    main()
