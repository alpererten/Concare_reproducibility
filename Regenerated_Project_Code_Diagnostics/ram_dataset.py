# ram_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset

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
    Demographics are zeros by default. Wire your own scaler here later if needed.
    """
    def __init__(self, split, cache_dir="npz_cache", demographic_dim=12):
        z = np.load(f"{cache_dir}/{split}.npz", allow_pickle=True)
        self.Xs = z["X"]  # ragged [T,F]
        self.ys = z["y"].astype(np.float32)
        self.demo_dim = demographic_dim

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, i):
        X = self.Xs[i].astype(np.float32)
        D = np.zeros((self.demo_dim,), dtype=np.float32)
        y = np.array([self.ys[i]], dtype=np.float32)
        return torch.from_numpy(X), torch.from_numpy(D), torch.from_numpy(y)
