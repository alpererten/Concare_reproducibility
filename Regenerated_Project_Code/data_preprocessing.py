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
    It transforms arrays of shape [T, F] or [B, T, F] with strict width checks
    and adds numerical safety to avoid NaNs/Infs during training (especially with AMP).
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
            self.stds  = np.asarray(state["stds"]).astype(np.float32)
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
        # Numerical safety: guard stds and clip Z to avoid fp16 overflow under AMP
        safe_stds = np.where(np.isfinite(self.stds) & (self.stds > 0.0), self.stds, 1.0).astype(np.float32)
        Z = (X2d - self.means[None, :]) / safe_stds[None, :]
        # Clip to a moderate range so downstream logits do not explode in fp16
        Z = np.clip(Z, -20.0, 20.0).astype(np.float32)
        return Z

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


# ---------------- Demographics scaling (new) ----------------

class DemoScaler:
    """Simple z-scoring scaler for 12-d demographics, with lazy fit+save."""
    def __init__(self, path: str = "artifacts/demo_norm.npz"):
        self.path = path
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray]  = None

    def load(self) -> bool:
        if not os.path.exists(self.path):
            return False
        data = np.load(self.path)
        self.means = data["means"].astype(np.float32)
        self.stds  = data["stds"].astype(np.float32)
        return True

    def fit_from_rows(self, rows: List[np.ndarray]) -> None:
        X = np.stack(rows).astype(np.float32)  # [N, 12]
        m = np.nanmean(X, axis=0)
        v = np.nanvar(X, axis=0)
        s = np.sqrt(np.clip(v, 1e-8, None))
        self.means, self.stds = m, s

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        np.savez(self.path, means=self.means, stds=self.stds)

    def transform(self, d: np.ndarray) -> np.ndarray:
        if self.means is None or self.stds is None:
            return d
        safe = np.where(np.isfinite(self.stds) & (self.stds > 0), self.stds, 1.0).astype(np.float32)
        z = (d - self.means) / safe
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# ---------------- Timeseries discretization (unchanged semantics) ----------------

def discretize_timeseries_fixed(
    df: pd.DataFrame,
    timestep: float,
    value_feature_count: Optional[int],
    expected_features: int
) -> Tuple[np.ndarray, List[str]]:
    """
    Discretize into bins of size `timestep`, expand with a mask per original channel,
    then pad with zeros to `expected_features` exactly.

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

    # Replace any remaining NaNs and ensure finiteness
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return X, feature_cols





class ConcareEpisodeDataset(Dataset):
    """
    Dataset that:
      1) loads per-stay timeseries and demographics
      2) discretizes + expands with masks
      3) normalizes with the loaded normalizer, enforcing exact width
      4) standardizes 12-d demographics with a saved scaler
      5) (NEW) caches per-sample tensors to disk for fast subsequent epochs
    """

    def __init__(
        self,
        split: str = "train",
        timestep: float = 0.8,
        normalizer: Optional[Normalizer] = None,
        expected_features: int = 76,
        value_feature_count: Optional[int] = 17,
        cache_tensors: bool = False,                  # NEW
        cache_dir: str = "cache",                     # NEW
    ):
        self.split = split
        self.timestep = float(timestep)
        self.normalizer = normalizer
        # Trust normalizer width if available
        self.expected_features = int(normalizer.feature_count) if (normalizer and normalizer.feature_count) else int(expected_features)
        self.value_feature_count = value_feature_count

        # Env toggles
        self.zero_demo = os.getenv("CONCARE_ZERO_DEMO", "0") == "1"
        self.demo_debug = os.getenv("CONCARE_DEMO_DEBUG", "0") == "1"

        # NEW: tensor cache setup (separate per split/feature width/timestep)
        self.cache_tensors = bool(cache_tensors)
        self.cache_dir = os.path.join(cache_dir, f"{split}_F{self.expected_features}_ts{str(self.timestep).replace('.','p')}")
        if self.cache_tensors:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Data roots
        self.ts_dir = f"data/{split}"
        self.demo_dir = "data/demographic"
        self.listfile = f"data/{split}_listfile.csv"

        # Listfile
        self.df = pd.read_csv(self.listfile)

        # Inspect first file for columns
        first_ts = pd.read_csv(os.path.join(self.ts_dir, self.df.iloc[0]["stay"]))
        self.feature_names = [c for c in first_ts.columns if c != "Hours"]
        print(
            f"[DEBUG] Dataset {split}: {len(self.df)} samples, "
            f"{len(self.feature_names)} raw features, expected_features={self.expected_features}, "
            f"timestep={self.timestep}"
        )

        # Demo scaler: load if exists; fit on a subset of train otherwise
        self.demo_scaler = DemoScaler("artifacts/demo_norm.npz")
        if self.demo_scaler.load():
            print("[DEBUG] Loaded demo scaler from artifacts/demo_norm.npz")
        elif self.split == "train":
            rows = []
            cap = min(len(self.df), 2000)
            for i in range(cap):
                demo_name = self.df.iloc[i]["stay"].replace("_timeseries", "")
                demo_file = os.path.join(self.demo_dir, demo_name)
                if os.path.exists(demo_file):
                    demo_df = pd.read_csv(demo_file)
                    rows.append(self._extract_demographics_raw(demo_df))
            if rows:
                self.demo_scaler.fit_from_rows(rows)
                self.demo_scaler.save()
                print("[DEBUG] Fitted and saved demo scaler to artifacts/demo_norm.npz")
            else:
                print("[WARN] Could not fit demo scaler — no demographic files found")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        stay_fname = row["stay"]                          # e.g., "12797_episode1_timeseries.csv"
        stay_name = stay_fname.replace("_timeseries.csv", "")
        ts_file = os.path.join(self.ts_dir, stay_fname)
        demo_file = os.path.join(self.demo_dir, stay_name)

        # -------- NEW: fast path via on-disk tensor cache --------
        if self.cache_tensors:
            base = stay_name.replace("/", "_")
            x_path = os.path.join(self.cache_dir, f"{base}_X.pt")
            d_path = os.path.join(self.cache_dir, f"{base}_D.pt")
            y_path = os.path.join(self.cache_dir, f"{base}_y.pt")
            if os.path.exists(x_path) and os.path.exists(d_path) and os.path.exists(y_path):
                try:
                    return torch.load(x_path), torch.load(d_path), torch.load(y_path)
                except Exception:
                    # cache might be stale/partial; rebuild below
                    pass
        # ---------------------------------------------------------

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
            X = self.normalizer.transform(X)

        # Sanitize NaNs/Infs
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Build demographics vector
        if self.zero_demo:
            D = np.zeros((12,), dtype=np.float32)
        else:
            if os.path.exists(demo_file):
                demo_df = pd.read_csv(demo_file)
                D = self._extract_demographics(demo_df)  # scaled if scaler is available
            else:
                D = np.zeros((12,), dtype=np.float32)
        D = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        y = np.array([float(row["y_true"])], dtype=np.float32)

        # Convert to tensors
        X_t = torch.from_numpy(X)
        D_t = torch.from_numpy(D)
        y_t = torch.from_numpy(y)

        # -------- NEW: write-through cache for future epochs/runs --------
        if self.cache_tensors:
            try:
                torch.save(X_t, x_path)
                torch.save(D_t, d_path)
                torch.save(y_t, y_path)
            except Exception as e:
                print(f"[WARN] Could not cache tensors for {stay_name}: {e}")
        # -----------------------------------------------------------------

        if self.demo_debug:
            dmin, dmax, dmean = float(np.min(D)), float(np.max(D)), float(np.mean(D))
            print(f"[DEBUG] D stats idx={idx} -> min={dmin:.4f} max={dmax:.4f} mean={dmean:.4f}")

        return X_t, D_t, y_t

    # --------- Demographics helpers ---------

    def _select_demo_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Choose up to 12 sensible demographics. Adjust the names to your CSVs.
        Falls back to first 12 numeric non-ID columns if expected names are missing.
        """
        preferred = [
            "Age", "Gender", "Height", "Weight",
            "Mean blood pressure", "Systolic blood pressure", "Diastolic blood pressure",
            "Heart Rate", "Respiratory rate", "Temperature",
            "Oxygen saturation", "Glascow coma scale total",
        ]
        cols = [c for c in preferred if c in df.columns]
        if len(cols) >= 12:
            return cols[:12]

        # fallback: numeric non-ID columns
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # drop likely ID-ish columns
        drop_keys = ("ID", "Id", "id", "ICUSTAY", "HADM", "SUBJECT")
        num_cols = [c for c in num_cols if not any(k in c for k in drop_keys)]
        # keep deterministic order
        extra = [c for c in num_cols if c not in cols]
        cols = (cols + extra)[:12]
        return cols

    def _extract_demographics_raw(self, df: pd.DataFrame, k: int = 12) -> np.ndarray:
        cols = self._select_demo_columns(df)
        if len(df) == 0 or not cols:
            return np.zeros((k,), dtype=np.float32)
        row = df.iloc[0][cols].to_numpy(dtype=np.float32, copy=False)
        # clamp extreme magnitudes (kill IDs or totals that slipped in)
        row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
        row[np.abs(row) > 1e6] = 0.0
        # pad/truncate to k
        if row.shape[0] >= k:
            return row[:k].astype(np.float32)
        out = np.zeros((k,), dtype=np.float32)
        out[: row.shape[0]] = row
        return out

    def _extract_demographics(self, df: pd.DataFrame, k: int = 12) -> np.ndarray:
        row = self._extract_demographics_raw(df, k=k)
        # scale if scaler is ready
        if self.demo_scaler is not None and self.demo_scaler.means is not None:
            row = self.demo_scaler.transform(row)
        return row


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
    return X_padded, D_stacked, y_stacked\n\n
# Parity helper for authors' transformers
# This does not change your existing classes, it only exposes the exact objects used by authors

import os
from pathlib import Path

def get_authors_transformers(
    timestep: float = 0.8,
    imputation: str = "previous",
    store_masks: bool = True,
    start_time: str = "zero",
    normalizer_state: str = "data/ihm_normalizer",
    resources_dir: str = None,
):
    """
    Return (Discretizer, Normalizer) that match the authors' implementations and configs.
    This assumes 'preprocessing.py' and 'resources/discretizer_config.json' are present.
    """
    if resources_dir is None:
        resources_dir = os.path.join(os.path.dirname(__file__), "resources")

    from preprocessing import Discretizer, Normalizer  # authors' modules
    disc = Discretizer(
        timestep=timestep,
        store_masks=store_masks,
        impute_strategy=imputation,
        start_time=start_time,
        config_path=os.path.join(resources_dir, "discretizer_config.json"),
    )
    norm = Normalizer()
    norm.load_params(normalizer_state)
    return disc, norm
