
import numpy as np
import torch
import random
from torch.utils.data import Dataset



def set_seed(seed: int = 42, deterministic: bool = True):

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def pad_collate(batch):
    Xs, Ds, ys = zip(*batch)
    max_time = max(x.shape[0] for x in Xs)
    feat = Xs[0].shape[1]
    B = len(Xs)
    Xp = torch.zeros(B, max_time, feat, dtype=torch.float32)
    for i, x in enumerate(Xs):
        T = x.shape[0]
        Xp[i, :T, :] = x
    return Xp, torch.stack(Ds), torch.stack(ys)


class RAMDataset(Dataset):
    """
    Loads discretized + normalized arrays from NPZ into memory.
    If demographics 'D' is present in the NPZ, uses it; otherwise falls back to zeros.
    Optional `data_bundle` allows reusing pre-loaded arrays (for CV) together with `indices`.
    """
    def __init__(self, split, cache_dir="data/normalized_data_cache", demographic_dim=12,
                 data_bundle=None, indices=None):
        if data_bundle is not None:
            self.Xs = data_bundle["X"]
            self.ys = data_bundle["y"].astype(np.float32)
            self.Ds = data_bundle.get("D")
        else:
            if split is None:
                raise ValueError("split must be provided when data_bundle is None")
            z = np.load(f"{cache_dir}/{split}.npz", allow_pickle=True)
            self.Xs = z["X"]  # ragged [T,F]
            self.ys = z["y"].astype(np.float32)
            self.Ds = z["D"] if "D" in z.files else None
        self.demo_dim = demographic_dim
        if indices is None:
            self.indices = np.arange(len(self.ys))
        else:
            self.indices = np.array(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        X = self.Xs[idx].astype(np.float32)
        if self.Ds is not None:
            D = self.Ds[idx].astype(np.float32)
        else:
            D = np.zeros((self.demo_dim,), dtype=np.float32)
        y = np.array([self.ys[idx]], dtype=np.float32)
        return torch.from_numpy(X), torch.from_numpy(D), torch.from_numpy(y)
