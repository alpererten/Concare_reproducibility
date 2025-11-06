# materialize_ram.py — saves per-scope outputs in separate subfolders
import os
import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from data_processing import Discretizer, Normalizer
from data_processing import InHospitalMortalityReader as IHMReader


def _cache_dir(scope: str) -> Path:
    """Return a distinct cache folder for each normalization scope."""
    d = Path(f"data/normalized_data_cache_{scope}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _stats_path(scope: str) -> Path:
    """Return path for saved normalization stats under that scope folder."""
    return _cache_dir(scope) / f"np_norm_stats_{scope}.npz"


def _accumulate_stats_over_reader(reader, disc, sum_x, sum_sq, N, F):
    """Compute running sums for mean/std across a reader."""
    for i in range(reader.get_number_of_examples()):
        ex = reader.read_example(i)
        Xd, _ = disc.transform(ex["X"], header=ex["header"], end=ex.get("t", None))
        Xd = Xd.astype(np.float32)
        if F is None:
            F = Xd.shape[1]
            sum_x = np.zeros(F, np.float64)
            sum_sq = np.zeros(F, np.float64)
        elif Xd.shape[1] != F:
            raise ValueError(f"Feature width changed: {Xd.shape[1]} vs {F}")
        sum_x += np.sum(Xd, axis=0, dtype=np.float64)
        sum_sq += np.sum(Xd.astype(np.float64) ** 2, axis=0, dtype=np.float64)
        N += Xd.shape[0]
    return sum_x, sum_sq, N, F


def _fit_normalizer(scope, disc, train_dir, train_list, val_dir, val_list, test_dir, test_list):
    """Fit means/stds for given scope and save them in its folder."""
    sum_x, sum_sq, N, F = None, None, 0, None
    train_reader = IHMReader(dataset_dir=train_dir, listfile=train_list)
    sum_x, sum_sq, N, F = _accumulate_stats_over_reader(train_reader, disc, sum_x, sum_sq, N, F)
    if scope == "all":
        for ddir, lfile in [(val_dir, val_list), (test_dir, test_list)]:
            r = IHMReader(dataset_dir=ddir, listfile=lfile)
            sum_x, sum_sq, N, F = _accumulate_stats_over_reader(r, disc, sum_x, sum_sq, N, F)

    means = (sum_x / max(N, 1)).astype(np.float32)
    var = (sum_sq - 2 * sum_x * means + N * (means.astype(np.float64) ** 2)) / max(N - 1, 1)
    stds = np.sqrt(np.maximum(var, 1e-14)).astype(np.float32)
    stds[stds < 1e-7] = 1e-7

    norm = Normalizer()
    norm._means = means
    norm._stds = stds

    np.savez_compressed(_stats_path(scope), means=means, stds=stds)
    return norm


def _load_or_fit_normalizer(norm_scope, timestep, imputation, store_masks, start_time,
                            train_dir, train_list, val_dir, val_list, test_dir, test_list):
    """Load or fit a normalizer in the chosen scope folder."""
    disc = Discretizer(
        timestep=timestep,
        store_masks=store_masks,
        impute_strategy=imputation,
        start_time=start_time,
        config_path=os.path.join("data", "discretizer_config.json"),
    )
    stats_file = _stats_path(norm_scope)
    if stats_file.exists():
        arr = np.load(stats_file, allow_pickle=True)
        means, stds = arr["means"].astype(np.float32), arr["stds"].astype(np.float32)
        norm = Normalizer()
        norm._means, norm._stds = means, stds
        return disc, norm
    norm = _fit_normalizer(norm_scope, disc,
                           train_dir, train_list, val_dir, val_list, test_dir, test_list)
    return disc, norm


def _write_split(reader, disc, norm, out_dir, split):
    """Write each split NPZ into its per-scope cache folder."""
    Xs, Ds, ys = [], [], []
    N = reader.get_number_of_examples()
    for i in range(N):
        ex = reader.read_example(i)
        Xd, _ = disc.transform(ex["X"], header=ex["header"], end=ex.get("t", None))
        Xn = norm.transform(Xd.astype(np.float32))
        Xs.append(Xn.astype(np.float32))
        ys.append(np.float32(ex["y"]))
        Ds.append(np.zeros((12,), np.float32))
    out_path = out_dir / f"{split}.npz"
    np.savez_compressed(out_path,
                        X=np.array(Xs, dtype=object),
                        D=np.array(Ds, np.float32),
                        y=np.array(ys, np.float32))
    print(f"[MATERIALIZE] scope={out_dir.name} wrote {out_path} ({N} stays)")


def materialize_split(split="train", timestep=0.8, append_masks=True,
                      imputation="previous", start_time="zero", norm_scope="train",
                      train_dir="data/train", train_list="data/train_listfile.csv",
                      val_dir="data/val", val_list="data/val_listfile.csv",
                      test_dir="data/test", test_list="data/test_listfile.csv"):
    """Main entry: writes split data under data/normalized_data_cache_{scope}/."""
    out_dir = _cache_dir(norm_scope)
    disc, norm = _load_or_fit_normalizer(norm_scope, timestep, imputation, append_masks, start_time,
                                         train_dir, train_list, val_dir, val_list, test_dir, test_list)
    reader = IHMReader(dataset_dir=f"data/{split}", listfile=f"data/{split}_listfile.csv")
    _write_split(reader, disc, norm, out_dir, split)


# CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser("Materialize NPZs")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    ap.add_argument("--timestep", type=float, default=0.8)
    ap.add_argument("--no_append_masks", action="store_true",
                    help="Disable appending mask columns (default: masks are appended)")
    ap.add_argument("--imputation", type=str, default="previous",
                    choices=["previous", "zero", "normal_value", "next"])
    ap.add_argument("--start_time", type=str, default="zero", choices=["zero", "relative"])
    ap.add_argument("--norm_scope", type=str, default="train", choices=["train", "all"],
                    help="Fit normalizer on 'train' (no leakage) or 'all' (train+val+test).")
    ap.add_argument("--train_dir", type=str, default="data/train")
    ap.add_argument("--train_listfile", type=str, default="data/train_listfile.csv")
    ap.add_argument("--val_dir", type=str, default="data/val")
    ap.add_argument("--val_listfile", type=str, default="data/val_listfile.csv")
    ap.add_argument("--test_dir", type=str, default="data/test")
    ap.add_argument("--test_listfile", type=str, default="data/test_listfile.csv")

    args = ap.parse_args()
    materialize_split(
        split=args.split,
        timestep=args.timestep,
        append_masks=not args.no_append_masks,
        imputation=args.imputation,
        start_time=args.start_time,
        norm_scope=args.norm_scope,
        train_dir=args.train_dir, train_list=args.train_listfile,
        val_dir=args.val_dir, val_list=args.val_listfile,
        test_dir=args.test_dir, test_list=args.test_listfile,
    )
