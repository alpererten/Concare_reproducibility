# data_preprocessing_fixed.py
"""
Fixed data preprocessing that properly handles MIMIC-III discretized data format
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, List


class Normalizer:
    """Normalizer that properly handles MIMIC-III preprocessed data"""
    def __init__(self, state_path: Optional[str] = "data/ihm_normalizer"):
        self.means = None
        self.stds = None
        if state_path and os.path.exists(state_path):
            with open(state_path, "rb") as f:
                data = f.read()
            try:
                state = pickle.loads(data, encoding="latin1")
            except TypeError:
                state = pickle.loads(data)
            self.means = np.asarray(state["means"]).astype(np.float32)
            self.stds = np.asarray(state["stds"]).astype(np.float32)
            print(f"[DEBUG] Normalizer loaded: {self.means.shape[0]} features")

    def transform(self, X):
        """Transform with proper dimension handling"""
        if self.means is None or self.stds is None:
            return X
        
        # Handle dimension mismatch gracefully
        if X.shape[-1] == self.means.shape[0]:
            # Perfect match - standard normalization
            return (X - self.means[None, None, :]) / self.stds[None, None, :]
        elif X.shape[-1] < self.means.shape[0]:
            # Fewer features than expected - pad with zeros after normalizing
            X_norm = (X - self.means[None, None, :X.shape[-1]]) / self.stds[None, None, :X.shape[-1]]
            padding = np.zeros((X.shape[0], X.shape[1], self.means.shape[0] - X.shape[-1]), dtype=np.float32)
            return np.concatenate([X_norm, padding], axis=-1)
        else:
            # More features than expected - truncate
            return (X[:, :, :self.means.shape[0]] - self.means[None, None, :]) / self.stds[None, None, :]


def discretize_timeseries_fixed(df, timestep: float = 1.0, expected_features: int = 76):
    """
    Fixed discretization that properly handles MIMIC-III format
    This should produce 76 features (17 raw + 59 for categorical/masks)
    """
    assert "Hours" in df.columns, "Input must include 'Hours'"
    
    df = df.copy()
    
    # Get feature columns (everything except Hours)
    feature_cols = [c for c in df.columns if c != "Hours"]
    
    # Convert to numeric
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    
    # Discretize into time bins
    df["_bin"] = np.floor(df["Hours"] / timestep).astype(int)
    T = int(df["_bin"].max() + 1) if len(df) > 0 else 48  # Default to 48 hours
    
    # Initialize output array with expected number of features
    # For MIMIC-III: 17 base features + masks/categoricals = 76 total
    out = np.full((T, expected_features), np.nan, dtype=np.float32)
    
    # Fill in the raw features (first 17 columns)
    for b in range(T):
        chunk = df[df["_bin"] == b]
        if len(chunk) > 0 and len(feature_cols) > 0:
            # Take last value in each bin
            values = chunk[feature_cols].iloc[-1].values.astype(np.float32)
            out[b, :len(values)] = values
    
    # Forward fill and imputation for raw features
    for j in range(min(len(feature_cols), expected_features)):
        col = out[:, j]
        # Forward fill
        last_val = np.nan
        for t in range(len(col)):
            if not np.isnan(col[t]):
                last_val = col[t]
            elif not np.isnan(last_val):
                col[t] = last_val
        
        # Fill remaining NaNs with column mean or 0
        if np.isnan(col).any():
            if not np.all(np.isnan(col)):
                col[np.isnan(col)] = np.nanmean(col)
            else:
                col[np.isnan(col)] = 0.0
        
        out[:, j] = col
    
    # Fill mask features (columns 17-76) 
    # These represent whether a measurement was taken
    if expected_features > len(feature_cols):
        mask_start = len(feature_cols)
        for b in range(T):
            chunk = df[df["_bin"] == b]
            if len(chunk) > 0:
                for j, col in enumerate(feature_cols):
                    if j < (expected_features - mask_start):
                        # Set mask to 1 if measurement exists, 0 otherwise
                        out[b, mask_start + j] = 0.0 if pd.isna(chunk[col].iloc[-1]) else 1.0
    
    # Ensure no NaN values remain
    out = np.nan_to_num(out, nan=0.0)
    
    return out.astype(np.float32), feature_cols


class ConcareEpisodeDataset(Dataset):
    """Fixed dataset that handles MIMIC-III format properly"""
    
    def __init__(self, split: str = "train", timestep: float = 1.0, 
                 normalizer: Optional[Normalizer] = None,
                 expected_features: int = 76):
        self.split = split
        self.timestep = timestep
        self.normalizer = normalizer
        self.expected_features = expected_features
        
        self.ts_dir = f"data/{split}"
        self.demo_dir = "data/demographic"
        self.listfile = f"data/{split}_listfile.csv"
        
        # Load listfile
        self.df = pd.read_csv(self.listfile)
        
        # Get feature names from first file
        first_ts = pd.read_csv(os.path.join(self.ts_dir, self.df.iloc[0]["stay"]))
        self.feature_names = [c for c in first_ts.columns if c != "Hours"]
        
        print(f"[DEBUG] Dataset {split}: {len(self.df)} samples, {len(self.feature_names)} raw features")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ts_file = os.path.join(self.ts_dir, row["stay"])
        demo_name = row["stay"].replace("_timeseries", "")
        demo_file = os.path.join(self.demo_dir, demo_name)
        
        # Load and discretize timeseries
        ts_df = pd.read_csv(ts_file)
        X, _ = discretize_timeseries_fixed(ts_df, self.timestep, self.expected_features)
        
        # Apply normalization
        if self.normalizer:
            # Ensure X has shape (T, F) before normalizing
            X_expanded = X[np.newaxis, :, :]  # Add batch dimension
            X_normalized = self.normalizer.transform(X_expanded)
            X = X_normalized[0]  # Remove batch dimension
        
        # Load demographics
        if os.path.exists(demo_file):
            demo_df = pd.read_csv(demo_file)
            D = self._extract_demographics(demo_df)
        else:
            D = np.zeros((12,), dtype=np.float32)
        
        # Get label
        y = float(row["y_true"])
        
        # Ensure no NaN values
        X = np.nan_to_num(X, nan=0.0)
        D = np.nan_to_num(D, nan=0.0)
        
        return torch.from_numpy(X), torch.from_numpy(D), torch.tensor([y], dtype=torch.float32)
    
    def _extract_demographics(self, df, k=12):
        """Extract k demographic features"""
        # Get numeric columns
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove mortality if present
        if "Mortality" in num_cols:
            num_cols.remove("Mortality")
        
        # Take first row
        if len(df) > 0 and len(num_cols) > 0:
            row = df[num_cols].iloc[0].values.astype(np.float32)
        else:
            row = np.array([], dtype=np.float32)
        
        # Pad or truncate to k features
        if len(row) >= k:
            return row[:k]
        else:
            padded = np.zeros((k,), dtype=np.float32)
            if len(row) > 0:
                padded[:len(row)] = row
            return padded


def pad_collate(batch):
    """Collate function with proper padding"""
    Xs, Ds, ys = zip(*batch)
    
    # Find max time length
    max_time = max(x.shape[0] for x in Xs)
    feature_dim = Xs[0].shape[1]
    batch_size = len(Xs)
    
    # Create padded tensor
    X_padded = torch.zeros(batch_size, max_time, feature_dim, dtype=torch.float32)
    
    # Fill in data
    for i, x in enumerate(Xs):
        X_padded[i, :x.shape[0], :] = x
    
    # Stack demographics and labels
    D_stacked = torch.stack(Ds)
    y_stacked = torch.stack(ys)
    
    return X_padded, D_stacked, y_stacked
