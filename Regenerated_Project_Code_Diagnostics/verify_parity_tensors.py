# verify_parity_tensors.py
import os
import argparse
import random
from pathlib import Path
import numpy as np

from authors_modules.readers import InHospitalMortalityReader
try:
    from authors_modules.preprocessing import Discretizer, Normalizer
except Exception:
    from authors_modules.preprocessing import Discretizer, Normalizer

def load_authors_tensor(reader, idx, discretizer, normalizer):
    ex = reader.read_example(idx)
    X_raw, header, t = ex["X"], ex["header"], ex["t"]
    X_disc, _ = discretizer.transform(X_raw, header=header, end=t)
    X_norm = normalizer.transform(X_disc)
    return X_norm, ex["name"]

def main():
    ap = argparse.ArgumentParser("Verify parity: authors' tensors vs parity NPZs")
    ap.add_argument("--dataset_dir", default="data/train")
    ap.add_argument("--listfile", default="data/train_listfile.csv")
    ap.add_argument("--cache_dir", default="data/normalized_data_cache")
    ap.add_argument("--normalizer_state", default="data/ihm_normalizer")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    reader = InHospitalMortalityReader(dataset_dir=args.dataset_dir, listfile=args.listfile)
    N = reader.get_number_of_examples()
    if N == 0:
        raise SystemExit("No examples found for given dataset_dir/listfile.")

    discretizer = Discretizer(
        timestep=0.8,
        store_masks=True,
        impute_strategy="previous",
        start_time="zero",
        config_path=os.path.join("data", "discretizer_config.json"),
    )
    normalizer = Normalizer()
    normalizer.load_params(args.normalizer_state)

    random.seed(args.seed)
    indices = random.sample(range(N), k=min(args.k, N))

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        raise SystemExit(f"Cache dir {cache_dir} does not exist. Run parity materializer first.")

    mismatches = []
    checked = 0

    for idx in indices:
        X_authors, name = load_authors_tensor(reader, idx, discretizer, normalizer)
        stem = Path(name).stem
        parity_npz = cache_dir / f"{stem}.npz"
        if not parity_npz.exists():
            raise SystemExit(f"Expected parity NPZ not found: {parity_npz}. Materialize this split first.")

        with np.load(parity_npz, allow_pickle=True) as z:
            X_mine = z["X"]

        try:
            np.testing.assert_allclose(
                X_authors.astype(np.float32),
                X_mine.astype(np.float32),
                rtol=0, atol=1e-6
            )
        except AssertionError as e:
            mismatches.append((stem, str(e)))
        checked += 1

    if len(mismatches) == 0:
        print(f"[OK] All {checked} stays matched element-wise at atol=1e-6.")
    else:
        print(f"[FAIL] {len(mismatches)} / {checked} mismatches.")
        for stem, err in mismatches[:10]:
            print(f" - {stem}: {err.splitlines()[0]}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
