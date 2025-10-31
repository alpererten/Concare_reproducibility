import os, sys, csv, math
import numpy as np
from typing import List, Tuple, Optional


# Always clean existing cache before re-materializing
CACHE_DIR = os.path.join("data", "normalized_data_cache")
if os.path.exists(CACHE_DIR):
    for f in os.listdir(CACHE_DIR):
        if f.endswith(".npz"):
            try:
                os.remove(os.path.join(CACHE_DIR, f))
            except Exception as e:
                print(f"[WARN] Could not remove {f}: {e}")
    print(f"[INFO] Cleared old cache files in {CACHE_DIR}")
os.makedirs(CACHE_DIR, exist_ok=True)


# ---------- Lightweight reader (no pandas) ----------
class SimpleTimeseriesReader:
    def __init__(self, dataset_dir: str, listfile: str):
        self.dataset_dir = dataset_dir
        self.samples: List[Tuple[str, float]] = []
        with open(listfile, "r", newline="") as f:
            rdr = csv.DictReader(f)
            if "stay" not in rdr.fieldnames or "y_true" not in rdr.fieldnames:
                raise ValueError(f"Listfile {listfile} must contain 'stay' and 'y_true' columns, got {rdr.fieldnames}")
            for row in rdr:
                self.samples.append((row["stay"], float(row["y_true"])))

    def __len__(self) -> int:
        return len(self.samples)

    def iter_samples(self):
        for stay, y in self.samples:
            ts_path = os.path.join(self.dataset_dir, stay)
            yield ts_path, y

    def read_timeseries(self, ts_path: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
        with open(ts_path, "r", newline="") as f:
            rdr = csv.reader(f)
            hdr = next(rdr)
            if len(hdr) < 2 or hdr[0] != "Hours":
                raise ValueError(f"{ts_path}: first column must be 'Hours'")
            feat_names = hdr[1:]
            hours, rows = [], []
            for row in rdr:
                if not row:
                    continue
                try:
                    h = float(row[0])
                except:
                    continue
                vals = []
                for x in row[1:]:
                    if x == "" or x is None:
                        vals.append(np.nan)
                    else:
                        try:
                            vals.append(float(x))
                        except:
                            vals.append(np.nan)
                hours.append(h)
                rows.append(vals)
        if len(rows) == 0:
            return feat_names, np.zeros((0,), dtype=np.float32), np.zeros((0, len(feat_names)), dtype=np.float32)
        return feat_names, np.asarray(hours, dtype=np.float32), np.asarray(rows, dtype=np.float32)


# ---------- Fast NumPy discretizer with authors' settings ----------
class DiscretizerNP:
    """
    Settings aligned to authors:
      timestep = 1.0
      append_masks = True
      impute_strategy = 'previous' (forward-fill)
    """
    def __init__(self, timestep: float = 1.0, append_masks: bool = True, impute_strategy: str = 'previous'):
        if timestep <= 0:
            raise ValueError("timestep must be positive")
        self.dt = float(timestep)
        self.append_masks = bool(append_masks)
        self.impute_strategy = impute_strategy

        if self.impute_strategy not in ['zero', 'previous', 'normal']:
            raise ValueError(f"Invalid impute_strategy: {self.impute_strategy}. Must be 'zero', 'previous', or 'normal'")

    def transform(self, hours: np.ndarray, values: np.ndarray) -> np.ndarray:
        """
        Discretize time series with forward-fill imputation.
        Returns [T, F] or [T, 2F] if masks are appended.
        """
        if values.size == 0:
            return np.zeros((0, 0), dtype=np.float32)

        h = np.maximum(0.0, hours.astype(np.float32))
        F = values.shape[1]
        t_max = float(h.max()) if h.size > 0 else 0.0
        T_bins = 1 + int(math.floor(t_max / self.dt)) if t_max > 0 else 1

        X = np.zeros((T_bins, F), dtype=np.float32)
        M = np.zeros((T_bins, F), dtype=np.float32)

        bin_idx = np.minimum((h / self.dt).astype(np.int64), T_bins - 1)

        for i, b in enumerate(bin_idx):
            row = values[i]
            finite = np.isfinite(row)
            if not finite.any():
                continue
            X[b, finite] = row[finite].astype(np.float32)
            M[b, finite] = 1.0

        if self.impute_strategy == 'previous':
            for f in range(F):
                last_val = 0.0
                for t in range(T_bins):
                    if M[t, f] == 1.0:
                        last_val = X[t, f]
                    else:
                        X[t, f] = last_val
        elif self.impute_strategy == 'normal':
            for f in range(F):
                for t in range(T_bins):
                    if M[t, f] == 0.0:
                        X[t, f] = 0.0
        # else 'zero' is already zeros

        if self.append_masks:
            X = np.concatenate([X, M], axis=1)  # [T, 2F]

        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return X.astype(np.float32)


# ---------- Normalizer: values only, never masks ----------
class NormalizerNP:
    """
    Normalize value columns only.
    If masks are present, the first F columns are values and are standardized.
    Mask columns remain 0 or 1.
    """
    def __init__(self, stats_path: Optional[str] = None):
        self.stats_path = stats_path
        self.means = None
        self.stds = None
        self.n_value_feats: Optional[int] = None
        if stats_path and os.path.exists(stats_path):
            z = np.load(stats_path)
            self.means = z["means"].astype(np.float32)
            self.stds = z["stds"].astype(np.float32)
            self.n_value_feats = int(z["n_value_feats"])
            print(f"[INFO] Loaded normalizer: {self.n_value_feats} value features")

    def fit(self, matrices: List[np.ndarray], n_value_feats: int):
        self.n_value_feats = int(n_value_feats)
        Fv = self.n_value_feats

        s = np.zeros(Fv, dtype=np.float64)
        ss = np.zeros(Fv, dtype=np.float64)
        n = np.zeros(Fv, dtype=np.int64)

        for X in matrices:
            Xv = X[:, :Fv]
            finite = np.isfinite(Xv)
            vals = np.where(finite, Xv, 0.0)
            s += vals.sum(axis=0)
            ss += (vals * vals).sum(axis=0)
            n += finite.sum(axis=0)

        n = np.maximum(n, 1)
        means = s / n
        vars_ = np.maximum(ss / n - means * means, 1e-6)
        stds = np.sqrt(vars_).astype(np.float32)

        self.means = means.astype(np.float32)
        self.stds = stds
        print(f"[INFO] Fitted normalizer on {self.n_value_feats} value features")

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.means is None or self.stds is None or self.n_value_feats is None:
            return X
        Fv = min(self.n_value_feats, X.shape[1])
        X_out = X.copy()
        X_out[:, :Fv] = (X[:, :Fv] - self.means[:Fv]) / self.stds[:Fv]
        return X_out

    def save(self):
        if self.stats_path and self.means is not None and self.stds is not None and self.n_value_feats is not None:
            os.makedirs(os.path.dirname(self.stats_path), exist_ok=True)
            np.savez_compressed(
                self.stats_path,
                means=self.means,
                stds=self.stds,
                n_value_feats=np.array(self.n_value_feats, dtype=np.int32),
            )
            print(f"[INFO] Saved normalizer to {self.stats_path}")


# ---------- Fixed-width helper ----------
TARGET_WIDTH = 76  # authors’ expanded width

def _to_fixed_width(X: np.ndarray, append_masks: bool, n_value_feats: int) -> np.ndarray:
    """
    Enforce exact width = TARGET_WIDTH.
    If current width is smaller, pad zeros at the end.
    If it is larger, slice to TARGET_WIDTH keeping the leading columns.
    """
    cur = X.shape[1]
    if cur == TARGET_WIDTH:
        return X
    if cur < TARGET_WIDTH:
        pad = np.zeros((X.shape[0], TARGET_WIDTH - cur), dtype=X.dtype)
        return np.concatenate([X, pad], axis=1)
    # cur > TARGET_WIDTH → slice
    return X[:, :TARGET_WIDTH]


# ---------- Materialization ----------
def materialize_split(
    split: str,
    timestep: float = 1.0,
    append_masks: bool = True,
    impute_strategy: str = 'previous',
):
    """
    Save all normalized .npz files and stats to:
    data/normalized_data_cache/

    This version enforces fixed input_dim = 76 to match the authors.
    """
    output_dir = os.path.join("data", "normalized_data_cache")
    normalizer_stats = os.path.join(output_dir, "np_norm_stats.npz")

    os.makedirs(output_dir, exist_ok=True)
    listfile = f"data/{split}_listfile.csv"

    print(f"\n[INFO] Materializing {split} split with:")
    print(f"       timestep={timestep}")
    print(f"       append_masks={append_masks}")
    print(f"       impute_strategy={impute_strategy}")
    print(f"       target_width={TARGET_WIDTH}")

    reader = SimpleTimeseriesReader(dataset_dir=f"data/{split}", listfile=listfile)
    discret = DiscretizerNP(timestep=timestep, append_masks=append_masks, impute_strategy=impute_strategy)

    norm = NormalizerNP(normalizer_stats)
    need_fit = (split == "train") and (norm.means is None or norm.stds is None)

    X_list, y_list = [], []
    tmp = []

    n_value_feats: Optional[int] = None

    for i, (ts_path, y) in enumerate(reader.iter_samples()):
        feat_names, hours, values = reader.read_timeseries(ts_path)
        X = discret.transform(hours, values)  # [T, F] or [T, 2F] if masks

        # Value feature count before padding or slicing
        if n_value_feats is None:
            n_value_feats = X.shape[1] // 2 if append_masks else X.shape[1]
            print(f"[INFO] Detected {n_value_feats} value features, raw total width={X.shape[1]}")

        # Enforce fixed width = 76
        X = _to_fixed_width(X, append_masks, n_value_feats)

        if need_fit:
            tmp.append(X.copy())
        else:
            X = norm.transform(X)

        X_list.append(X.astype(np.float32))
        y_list.append(np.float32(y))

        if (i + 1) % 1000 == 0:
            print(f"[INFO] {split}: processed {i+1}/{len(reader)}")

    # Fit normalizer on training data
    if need_fit:
        print(f"[INFO] Fitting normalizer on {len(tmp)} training samples...")
        norm.fit(tmp, n_value_feats=n_value_feats)
        norm.save()
        # Transform all training data
        X_list = [norm.transform(x) for x in X_list]

    # Save to disk
    out_path = os.path.join(output_dir, f"{split}.npz")
    np.savez_compressed(out_path, X=np.array(X_list, dtype=object), y=np.array(y_list, dtype=np.float32))

    print(f"[OK] Materialized {split}: {len(y_list)} samples → {out_path}")
    print(f"     Enforced input_dim = {X_list[0].shape[1]} (expected {TARGET_WIDTH})\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("PHASE 2: DATA PREPROCESSING FIXES WITH FIXED WIDTH")
    print("="*60)
    print("Changes applied:")
    print("  1. timestep set to 1.0")
    print("  2. masks appended to values")
    print("  3. forward-fill imputation")
    print("  4. input width forced to 76")
    print("="*60 + "\n")

    for sp in ["train", "val", "test"]:
        materialize_split(sp, timestep=1.0, append_masks=True, impute_strategy='previous')
