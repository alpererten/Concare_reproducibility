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


# ---------- Fast NumPy discretizer ----------
class DiscretizerNP:
    def __init__(self, timestep: float = 0.8, append_masks: bool = False):
        if timestep <= 0:
            raise ValueError("timestep must be positive")
        self.dt = float(timestep)
        self.append_masks = bool(append_masks)

    def transform(self, hours: np.ndarray, values: np.ndarray) -> np.ndarray:
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

        if self.append_masks:
            X = np.concatenate([X, M], axis=1)

        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return X.astype(np.float32)


# ---------- Normalizer: values only, never masks ----------
class NormalizerNP:
    """
    Standardize only VALUE columns. Mask columns remain 0 or 1.
    Stores n_value_feats so we always skip masks correctly.
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

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.means is None or self.stds is None or self.n_value_feats is None:
            return X
        Fv = min(self.n_value_feats, X.shape[1])
        X[:, :Fv] = (X[:, :Fv] - self.means[:Fv]) / self.stds[:Fv]
        return X

    def save(self):
        if self.stats_path and self.means is not None and self.stds is not None and self.n_value_feats is not None:
            os.makedirs(os.path.dirname(self.stats_path), exist_ok=True)
            np.savez_compressed(
                self.stats_path,
                means=self.means,
                stds=self.stds,
                n_value_feats=np.array(self.n_value_feats, dtype=np.int32),
            )


# ---------- Materialization ----------
def materialize_split(
    split: str,
    timestep: float = 0.8,
    append_masks: bool = True,
):
    """
    Save all normalized .npz files and stats to:
    data/normalized_data_cache/
    """
    output_dir = os.path.join("data", "normalized_data_cache")
    normalizer_stats = os.path.join(output_dir, "np_norm_stats.npz")

    os.makedirs(output_dir, exist_ok=True)
    listfile = f"data/{split}_listfile.csv"
    reader = SimpleTimeseriesReader(dataset_dir=f"data/{split}", listfile=listfile)
    discret = DiscretizerNP(timestep=timestep, append_masks=append_masks)

    norm = NormalizerNP(normalizer_stats)
    need_fit = (split == "train") and (norm.means is None or norm.stds is None)

    X_list, y_list = [], []
    tmp = []

    n_value_feats: Optional[int] = None

    for i, (ts_path, y) in enumerate(reader.iter_samples()):
        feat_names, hours, values = reader.read_timeseries(ts_path)
        X = discret.transform(hours, values)  # [T, Fv] or [T, 2Fv] if masks
        if n_value_feats is None:
            n_value_feats = X.shape[1] // 2 if discret.append_masks else X.shape[1]
        if need_fit:
            tmp.append(X.copy())
        else:
            X = norm.transform(X)
        X_list.append(X.astype(np.float32))
        y_list.append(np.float32(y))
        if (i + 1) % 1000 == 0:
            print(f"[INFO] {split}: processed {i+1}/{len(reader)}")

    if need_fit:
        norm.fit(tmp, n_value_feats=n_value_feats)
        norm.save()
        X_list = [norm.transform(x) for x in X_list]

    out_path = os.path.join(output_dir, f"{split}.npz")
    np.savez_compressed(out_path, X=np.array(X_list, dtype=object), y=np.array(y_list, dtype=np.float32))
    print(f"[OK] materialized {split}: {len(y_list)} samples → {out_path}")


if __name__ == "__main__":
    for sp in ["train", "val", "test"]:
        materialize_split(sp)
