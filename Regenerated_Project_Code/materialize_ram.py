# materialize_ram.py — revised to match "good" behavior (F=76, masks included, all columns normalized)
import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Ensure local imports (preprocessing.py, readers.py)
sys.path.insert(0, os.path.dirname(__file__))
from authors_modules.preprocessing import Discretizer, Normalizer
from authors_modules.readers import InHospitalMortalityReader as IHMReader

CACHE_DIR = Path("data/normalized_data_cache")
NORM_STATS = CACHE_DIR / "np_norm_stats.npz"


def _ensure_dirs():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _fit_normalizer_over_reader(reader, disc):
    """
    Fit means/stds over the FULL discretized feature space (including masks and one-hots),
    to mirror the "good" cache (76 columns where masks were also normalized).
    Returns a Normalizer whose _means/_stds are set in-memory, and persists stats to NORM_STATS.
    """
    # First example to get dimensionality
    ex0 = reader.read_example(0)
    X0, _ = disc.transform(ex0["X"], header=ex0["header"], end=ex0.get("t", None))
    X0 = X0.astype(np.float32)
    F = X0.shape[1]

    # Running sums to avoid holding everything in RAM
    sum_x = np.sum(X0, axis=0, dtype=np.float64)
    sum_sq = np.sum(X0.astype(np.float64) ** 2, axis=0, dtype=np.float64)
    N = X0.shape[0]

    # Remaining examples
    N_examples = reader.get_number_of_examples()
    for idx in range(1, N_examples):
        ex = reader.read_example(idx)
        Xd, _ = disc.transform(ex["X"], header=ex["header"], end=ex.get("t", None))
        Xd = Xd.astype(np.float32)
        # Safety: assert feature width consistent
        if Xd.shape[1] != F:
            raise ValueError(f"Feature width changed across examples: {Xd.shape[1]} vs {F}")
        sum_x += np.sum(Xd, axis=0, dtype=np.float64)
        sum_sq += np.sum(Xd.astype(np.float64) ** 2, axis=0, dtype=np.float64)
        N += Xd.shape[0]

    means = (sum_x / max(N, 1)).astype(np.float32)
    # Sample std (ddof=1) with numeric stability
    # std^2 = 1/(N-1) * (sum_sq - 2*sum_x*mean + N*mean^2)
    num = sum_sq - 2.0 * sum_x * means + N * (means.astype(np.float64) ** 2)
    denom = max(N - 1, 1)
    stds = np.sqrt(np.maximum(num / denom, 1e-14)).astype(np.float32)
    stds[stds < 1e-7] = 1e-7

    # Build a Normalizer that will normalize ALL columns (matches "good" cache)
    norm = Normalizer()   # fields=None → transform() applies to all columns
    norm._means = means
    norm._stds = stds

    # Persist stats for reuse
    _ensure_dirs()
    np.savez_compressed(NORM_STATS, means=means, stds=stds)

    return norm


def _load_or_fit_normalizer(train_dir, train_listfile, timestep=0.8, imputation="previous",
                            store_masks=True, start_time="zero"):
    """
    Ensure Discretizer (with masks as requested) and a Normalizer with stats.
    If stats exist, load them; else fit from TRAIN and save.
    """
    _ensure_dirs()
    disc = Discretizer(
        timestep=timestep,
        store_masks=store_masks,        # IMPORTANT: include masks to reach F=76
        impute_strategy=imputation,
        start_time=start_time,
        config_path=os.path.join("data", "discretizer_config.json"),
    )

    if NORM_STATS.exists():
        arr = np.load(NORM_STATS, allow_pickle=True)
        means, stds = arr["means"].astype(np.float32), arr["stds"].astype(np.float32)
        norm = Normalizer()             # normalize all columns
        norm._means = means
        norm._stds = stds
        return disc, norm

    # Fit from TRAIN
    train_reader = IHMReader(dataset_dir=train_dir, listfile=train_listfile)
    norm = _fit_normalizer_over_reader(train_reader, disc)
    return disc, norm


def _write_split(reader, disc, norm, out_path: Path):
    """
    Discretize + normalize every stay in the split reader and pack into split NPZ:
      X: object array of [T_i, F] float32 (already normalized)
      D: demographics (zeros placeholder unless you populate real vectors)
      y: labels
    """
    Xs, Ds, ys = [], [], []
    N = reader.get_number_of_examples()
    for idx in range(N):
        ex = reader.read_example(idx)
        Xd, _ = disc.transform(ex["X"], header=ex["header"], end=ex.get("t", None))
        Xn = norm.transform(Xd.astype(np.float32))  # normalize ALL columns (matches "good")
        Xs.append(Xn.astype(np.float32))
        ys.append(np.float32(ex["y"]))
        Ds.append(np.zeros((12,), dtype=np.float32))  # keep placeholder unless you have real D

    np.savez_compressed(out_path,
                        X=np.array(Xs, dtype=object),
                        D=np.array(Ds, dtype=np.float32),
                        y=np.array(ys, dtype=np.float32))


def materialize_split(split: str, timestep: float = 0.8, append_masks: bool = True,
                      imputation: str = "previous", start_time: str = "zero"):
    """
    Default path used by training: fits/loads stats from TRAIN (with masks included),
    then writes {split}.npz with already-normalized arrays.
    """
    _ensure_dirs()

    # Readers for {split}
    dataset_dir = f"data/{split}"
    listfile = f"data/{split}_listfile.csv"

    # Always load/fit stats using TRAIN with the same discretizer config
    disc, norm = _load_or_fit_normalizer(
        train_dir="data/train",
        train_listfile="data/train_listfile.csv",
        timestep=timestep,
        imputation=imputation,
        store_masks=append_masks,   # append_masks → include masks to hit F=76
        start_time=start_time,
    )

    reader = IHMReader(dataset_dir=dataset_dir, listfile=listfile)
    out_path = CACHE_DIR / f"{split}.npz"
    _write_split(reader, disc, norm, out_path)
    print(f"[MATERIALIZE] Wrote {out_path} with {reader.get_number_of_examples()} cases")


# ------------------ Authors parity path (unchanged) ------------------
# Kept for side-by-side checks using authors' ihm_normalizer if needed.
from authors_modules.preprocessing import Discretizer as ADisc, Normalizer as ANorm
from authors_modules.readers import (
    InHospitalMortalityReader, DecompensationReader,
    LengthOfStayReader, PhenotypingReader, MultitaskReader
)

TASK_READERS = {
    "ihm": InHospitalMortalityReader,
    "decomp": DecompensationReader,
    "los": LengthOfStayReader,
    "pheno": PhenotypingReader,
    "multitask": MultitaskReader,
}

def materialize_parity(
    dataset_dir: str,
    listfile: str,
    task: str = "ihm",
    timestep: float = 0.8,
    imputation: str = "previous",
    store_masks: bool = True,
    start_time: str = "zero",
    normalizer_state: str = "data/ihm_normalizer",
    small_part: bool = False,
    limit: int = None,
):
    """
    Authors-compatible materializer using supplied ihm_normalizer for parity experiments.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    task = task.lower()
    if task not in TASK_READERS:
        raise ValueError(f"Unknown task {task}. Expected one of {list(TASK_READERS.keys())}")
    ReaderCls = TASK_READERS[task]
    reader = ReaderCls(dataset_dir=dataset_dir, listfile=listfile)

    disc = ADisc(
        timestep=timestep,
        store_masks=store_masks,
        impute_strategy=imputation,
        start_time=start_time,
        config_path=os.path.join("data", "discretizer_config.json"),
    )
    norm = ANorm(); norm.load_params(normalizer_state)

    N = reader.get_number_of_examples()
    if small_part:
        N = min(N, 1000)
    if limit is not None:
        N = min(N, limit)

    written = 0
    for idx in range(N):
        ex = reader.read_example(idx)
        Xd, _ = disc.transform(ex["X"], header=ex["header"], end=ex.get("t", None))
        Xn = norm.transform(Xd)
        out_path = CACHE_DIR / f"{Path(ex['name']).stem}.npz"
        np.savez_compressed(out_path,
                            X=Xn.astype(np.float32),
                            y=np.int64(ex.get("y", -1)),
                            name=str(ex["name"]),
                            t=np.float32(ex.get("t", -1.0)))
        written += 1
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Materialize NPZs")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    ap.add_argument("--timestep", type=float, default=0.8)
    ap.add_argument("--append_masks", action="store_true")
    ap.add_argument("--imputation", type=str, default="previous",
                    choices=["previous", "zero", "normal_value", "next"])
    ap.add_argument("--start_time", type=str, default="zero", choices=["zero", "relative"])
    args = ap.parse_args()
    materialize_split(args.split,
                      timestep=args.timestep,
                      append_masks=args.append_masks,
                      imputation=args.imputation,
                      start_time=args.start_time)
