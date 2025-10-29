# data_preprocessing.py

import os
import math
import pickle
from typing import Tuple, Optional, List, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class Normalizer:
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
            self.stds  = np.asarray(state["stds"]).astype(np.float32)

    def fit(self, X):
        self.means = np.nanmean(X, axis=(0, 1))
        self.stds = np.nanstd(X, axis=(0, 1)) + 1e-8

    def transform(self, X):
        if self.means is None or self.stds is None:
            return X
        if X.shape[-1] != self.means.shape[0]:
            return X
        return (X - self.means[None, None, :]) / self.stds[None, None, :]


def discretize_timeseries(df, timestep: float = 1.0):
    assert "Hours" in df.columns, "Input must include 'Hours'"
    df = df.copy()
    cols = [c for c in df.columns if c != "Hours"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["_bin"] = np.floor(df["Hours"] / timestep).astype(int)
    T = int(df["_bin"].max() + 1)
    out = np.full((T, len(cols)), np.nan, dtype=np.float32)
    for b in range(T):
        chunk = df[df["_bin"] == b]
        if len(chunk) > 0:
            out[b] = chunk[cols].iloc[-1].values.astype(np.float32)
    for j in range(out.shape[1]):
        col = out[:, j]
        last = np.nan
        for t in range(len(col)):
            if not np.isnan(col[t]):
                last = col[t]
            else:
                col[t] = last
        if np.isnan(col).any():
            m = np.nanmean(col)
            col[np.isnan(col)] = 0 if np.isnan(m) else m
        out[:, j] = col
    return out.astype(np.float32), cols


def _pick_demographics(df, k=12):
    num = df.select_dtypes(include=[np.number])
    if "Mortality" in num.columns:
        num = num.drop(columns=["Mortality"], errors="ignore")
    row = num.iloc[0].values.astype(np.float32)
    if len(row) >= k:
        return row[:k]
    padded = np.zeros((k,), dtype=np.float32)
    padded[: len(row)] = row
    return padded


class ConcareEpisodeDataset(Dataset):
    """
    Assumes fixed structure:
      data/
        train/   → *_timeseries.csv
        val/     → *_timeseries.csv
        test/    → optional
        demographic/ → *.csv
        train_listfile.csv
        val_listfile.csv
        ihm_normalizer
    """
    def __init__(self, split: str = "train", timestep: float = 1.0, normalizer: Optional[Normalizer] = None):
        self.split = split
        self.timestep = timestep
        self.normalizer = normalizer
        self.ts_dir = f"data/{split}"
        self.demo_dir = "data/demographic"
        self.listfile = f"data/{split}_listfile.csv"
        self.df = pd.read_csv(self.listfile)
        first_ts = pd.read_csv(os.path.join(self.ts_dir, self.df.iloc[0]["stay"]))
        _, self.feature_names = discretize_timeseries(first_ts)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ts_file = os.path.join(self.ts_dir, row["stay"])
        demo_name = row["stay"].replace("_timeseries", "")
        demo_file = os.path.join(self.demo_dir, demo_name)

        ts_df = pd.read_csv(ts_file)
        X, cols = discretize_timeseries(ts_df)
        if cols != self.feature_names:
            dfx = ts_df.copy()
            for c in self.feature_names:
                if c not in dfx.columns:
                    dfx[c] = np.nan
            X, _ = discretize_timeseries(dfx[["Hours"] + self.feature_names])
        if self.normalizer:
            X = self.normalizer.transform(X)
        D = _pick_demographics(pd.read_csv(demo_file)) if os.path.exists(demo_file) else np.zeros((12,), dtype=np.float32)
        y = float(row["y_true"])
        return torch.from_numpy(X), torch.from_numpy(D), torch.tensor([y], dtype=torch.float32)


def pad_collate(batch):
    Xs, Ds, ys = zip(*batch)
    B, T, N = len(Xs), max(x.shape[0] for x in Xs), Xs[0].shape[1]
    Xp = torch.zeros(B, T, N)
    for i, x in enumerate(Xs):
        Xp[i, : x.shape[0], :] = x
    return Xp, torch.stack(Ds), torch.stack(ys)
