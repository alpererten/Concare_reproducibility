
import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, List


class Normalizer:
    """
    Normalizer that loads means and stds for the expanded design.
    It transforms arrays of shape [T, F] or [B, T, F] with strict width checks.
    """
    def __init__(self, state_path: Optional[str] = "data/ihm_normalizer"):
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None
        self.state_path = state_path
        if state_path and os.path.exists(state_path):
            with open(state_path, "rb") as f:
                data = f.read()
            try:
                state = pickle.loads(data, encoding="latin1")
            except TypeError:
                state = pickle.loads(data)
            self.means = np.asarray(state["means"]).astype(np.float32)
            self.stds = np.asarray(state["stds"]).astype(np.float32)
            print(f"[DEBUG] Normalizer loaded with {self.means.shape[0]} features from {state_path}")

    @property
    def feature_count(self) -> Optional[int]:
        return None if self.means is None else int(self.means.shape[0])

    def _transform_2d(self, X2d: np.ndarray) -> np.ndarray:
        # X2d is [T, F]
        if self.means is None or self.stds is None:
            return X2d
        F = X2d.shape[-1]
        expF = self.means.shape[0]
        if F != expF:
            raise ValueError(
                f"Normalizer width mismatch. X has {F} features but normalizer expects {expF}. "
                f"Ensure discretization expected_features equals the normalizer length."
            )
        return (X2d - self.means[None, :]) / self.stds[None, :]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        X can be [T, F] or [B, T, F]. Returns the same shape.
        """
        if X.ndim == 2:
            return self._transform_2d(X)
        if X.ndim == 3:
            out = np.empty_like(X, dtype=np.float32)
            for i in range(X.shape[0]):
                out[i] = self._transform_2d(X[i])
            return out
        return X


def discretize_timeseries_fixed(
    df: pd.DataFrame,
    timestep: float,
    value_feature_count: Optional[int],
    expected_features: int
) -> Tuple[np.ndarray, List[str]]:
    """
    Discretize into bins of size `timestep`, expand with a mask per original channel,
    then align to `expected_features` exactly.

    value_feature_count controls how many leftmost columns are value channels.
    For MIMIC-III IHM the raw set is typically 17. The mask block is the same count.
    The final width equals expected_features, which should match the loaded normalizer.
    """
    assert "Hours" in df.columns, "Input must include 'Hours'"
    df = df.copy()

    # Feature columns in source csv, ignoring Hours
    feature_cols = [c for c in df.columns if c != "Hours"]

    # Convert to numeric
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Decide how many raw value channels to place before masks
    V = value_feature_count if value_feature_count is not None else len(feature_cols)
    V = min(V, len(feature_cols))  # guard
    if V == 0:
        # Degenerate input, create at least one step with zeros
        T = 1
        out = np.zeros((T, expected_features), dtype=np.float32)
        return out, feature_cols

    # Time binning
    df["_bin"] = np.floor(df["Hours"] / float(timestep)).astype(int)
    T = int(df["_bin"].max() + 1) if len(df) > 0 else 48  # default 48 bins if empty

    # Build base [T, V + V] = [values, masks]
    base_width = V + V
    if expected_features < base_width:
        raise ValueError(
            f"expected_features={expected_features} is smaller than value+mask base width={base_width}. "
            f"Increase expected_features or reduce value_feature_count."
        )

    X = np.full((T, base_width), np.nan, dtype=np.float32)

    # Fill values block with last observation per bin
    for b in range(T):
        chunk = df[df["_bin"] == b]
        if len(chunk) > 0:
            vals = chunk[feature_cols[:V]].iloc[-1].to_numpy(dtype=np.float32, copy=False)
            X[b, :V] = vals

    # Forward fill within the values block and impute remaining NaNs
    for j in range(V):
        col = X[:, j]
        # forward fill
        last_val = np.nan
        for t in range(T):
            if not np.isnan(col[t]):
                last_val = col[t]
            elif not np.isnan(last_val):
                col[t] = last_val
        # impute remaining NaNs
        if np.isnan(col).any():
            if np.all(np.isnan(col)):
                col[:] = 0.0
            else:
                m = np.nanmean(col)
                col[np.isnan(col)] = m
        X[:, j] = col

    # Build mask block: 1 if a measurement exists in bin for that variable, else 0
    mask_block = np.zeros((T, V), dtype=np.float32)
    for b in range(T):
        chunk = df[df["_bin"] == b]
        if len(chunk) > 0:
            last = chunk[feature_cols[:V]].iloc[-1]
            mask_block[b] = (~last.isna()).to_numpy(dtype=np.float32, copy=False)
    X[:, V:V+V] = mask_block

    # Pad with zeros to match expected_features exactly
    if base_width < expected_features:
        pad = np.zeros((T, expected_features - base_width), dtype=np.float32)
        X = np.concatenate([X, pad], axis=-1)

    # Replace any remaining NaNs
    X = np.nan_to_num(X, nan=0.0)
    return X.astype(np.float32), feature_cols


class ConcareEpisodeDataset(Dataset):
    """
    Dataset that:
      1) loads per-stay timeseries and demographics
      2) discretizes + expands with masks
      3) normalizes with the loaded normalizer, enforcing exact width
    """

    def __init__(
        self,
        split: str = "train",
        timestep: float = 0.8,
        normalizer: Optional[Normalizer] = None,
        expected_features: int = 76,
        value_feature_count: Optional[int] = 17
    ):
        self.split = split
        self.timestep = float(timestep)
        self.normalizer = normalizer
        # If a normalizer is provided and loaded, trust its width as single source of truth
        self.expected_features = int(normalizer.feature_count) if (normalizer and normalizer.feature_count) else int(expected_features)
        self.value_feature_count = value_feature_count

        self.ts_dir = f"data/{split}"
        self.demo_dir = "data/demographic"
        self.listfile = f"data/{split}_listfile.csv"

        # Load listfile
        self.df = pd.read_csv(self.listfile)

        # Inspect first file for column sanity
        first_ts = pd.read_csv(os.path.join(self.ts_dir, self.df.iloc[0]["stay"]))
        self.feature_names = [c for c in first_ts.columns if c != "Hours"]
        print(
            f"[DEBUG] Dataset {split}: {len(self.df)} samples, "
            f"{len(self.feature_names)} raw features, "
            f"expected_features={self.expected_features}, timestep={self.timestep}"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ts_file = os.path.join(self.ts_dir, row["stay"])
        demo_name = row["stay"].replace("_timeseries", "")
        demo_file = os.path.join(self.demo_dir, demo_name)

        # Load and discretize timeseries to exactly expected_features
        ts_df = pd.read_csv(ts_file)
        X, _ = discretize_timeseries_fixed(
            ts_df,
            timestep=self.timestep,
            value_feature_count=self.value_feature_count,
            expected_features=self.expected_features,
        )

        # Apply normalization with strict width equality
        if self.normalizer and self.normalizer.feature_count:
            # transform supports [T, F] directly
            X = self.normalizer.transform(X)

        # Load demographics vector of length 12
        if os.path.exists(demo_file):
            demo_df = pd.read_csv(demo_file)
            D = self._extract_demographics(demo_df)
        else:
            D = np.zeros((12,), dtype=np.float32)

        y = float(row["y_true"])

        # Final sanitation
        X = np.nan_to_num(X, nan=0.0)
        D = np.nan_to_num(D, nan=0.0)

        return torch.from_numpy(X), torch.from_numpy(D), torch.tensor([y], dtype=torch.float32)

    def _extract_demographics(self, df: pd.DataFrame, k: int = 12) -> np.ndarray:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if "Mortality" in num_cols:
            num_cols.remove("Mortality")
        if len(df) > 0 and len(num_cols) > 0:
            row = df[num_cols].iloc[0].to_numpy(dtype=np.float32, copy=False)
        else:
            row = np.array([], dtype=np.float32)
        if row.shape[0] >= k:
            return row[:k]
        out = np.zeros((k,), dtype=np.float32)
        if row.shape[0] > 0:
            out[: row.shape[0]] = row
        return out


def pad_collate(batch):
    """
    Collate that right-pads variable-length time axes to the max length in the batch.
    """
    Xs, Ds, ys = zip(*batch)
    max_time = max(x.shape[0] for x in Xs)
    feature_dim = Xs[0].shape[1]
    B = len(Xs)

    X_padded = torch.zeros(B, max_time, feature_dim, dtype=torch.float32)
    for i, x in enumerate(Xs):
        T = x.shape[0]
        X_padded[i, :T, :] = x

    D_stacked = torch.stack(Ds)
    y_stacked = torch.stack(ys)
    return X_padded, D_stacked, y_stacked
