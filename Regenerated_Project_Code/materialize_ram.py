
import os
import argparse
import numpy as np
from pathlib import Path

# Authors' modules
# These must exist next to this file based on what you uploaded earlier
from authors_modules.preprocessing import Discretizer, Normalizer
from authors_modules.readers import InHospitalMortalityReader, DecompensationReader, LengthOfStayReader, PhenotypingReader, MultitaskReader

CACHE_DIR = Path("data/normalized_data_cache")
ARTIFACTS_DIR = Path("artifacts")
RESOURCES_DIR = Path(os.path.join(os.path.dirname(__file__), "resources"))

TASK_READERS = {
    "ihm": InHospitalMortalityReader,
    "decomp": DecompensationReader,
    "los": LengthOfStayReader,
    "pheno": PhenotypingReader,
    "multitask": MultitaskReader,
}

def _ensure_dirs():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

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

    _ensure_dirs()

    # Select reader by task
    task = task.lower()
    if task not in TASK_READERS:
        raise ValueError(f"Unknown task {task}. Expected one of {list(TASK_READERS.keys())}")
    ReaderCls = TASK_READERS[task]

    # Init reader
    reader = ReaderCls(dataset_dir=dataset_dir, listfile=listfile)

    # Init authors' discretizer and normalizer
    disc = Discretizer(
        timestep=timestep,
        store_masks=store_masks,
        impute_strategy=imputation,
        start_time=start_time,
        config_path=os.path.join("data", "discretizer_config.json"),
    )
    norm = Normalizer()
    norm.load_params(normalizer_state)

    N = reader.get_number_of_examples()
    if small_part:
        N = min(N, 1000)
    if limit is not None:
        N = min(N, limit)

    written = 0
    for idx in range(N):
        example = reader.read_example(idx)
        X_raw = example["X"]
        header = example["header"]
        t = example.get("t", None)
        y = example.get("y", None)
        name = example["name"]

        # For tasks without 'y' in the same shape, save what exists
        if y is None:
            # Attempt common label fields by task
            if task == "ihm":
                raise RuntimeError("IHM task must have 'y' in reader example")
            else:
                # For other tasks we can skip y or infer differently, but parity is only guaranteed for IHM now
                y = -1

        X_disc, _new_header = disc.transform(X_raw, header=header, end=t)
        X_norm = norm.transform(X_disc)

        out_path = CACHE_DIR / f"{Path(name).stem}.npz"
        np.savez_compressed(out_path, X=X_norm.astype(np.float32), y=np.int64(y), name=str(name), t=np.float32(t if t is not None else -1.0))
        written += 1

    return written


def main():
    parser = argparse.ArgumentParser("Materialize NPZs that are parity identical with authors' tensors")
    parser.add_argument("--dataset_dir", type=str, default="data/train")
    parser.add_argument("--listfile", type=str, default="data/train_listfile.csv")
    parser.add_argument("--task", type=str, default="ihm")
    parser.add_argument("--timestep", type=float, default=0.8)
    parser.add_argument("--imputation", type=str, default="previous", choices=["previous", "zero", "normal_value", "next"])
    parser.add_argument("--store_masks", action="store_true", default=True)
    parser.add_argument("--no_store_masks", dest="store_masks", action="store_false")
    parser.add_argument("--start_time", type=str, default="zero", choices=["zero", "relative"])
    parser.add_argument("--normalizer_state", type=str, default="data/ihm_normalizer")
    parser.add_argument("--small_part", action="store_true", default=False)
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    written = materialize_parity(
        dataset_dir=args.dataset_dir,
        listfile=args.listfile,
        task=args.task,
        timestep=args.timestep,
        imputation=args.imputation,
        store_masks=args.store_masks,
        start_time=args.start_time,
        normalizer_state=args.normalizer_state,
        small_part=args.small_part,
        limit=args.limit,
    )
    print(f"[PARITY] Wrote {written} NPZ files to {CACHE_DIR}")

if __name__ == "__main__":
    main()
