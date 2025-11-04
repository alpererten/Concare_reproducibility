# compare_one_npz.py
import argparse, numpy as np, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    args = ap.parse_args()

    z = np.load(os.path.join(args.cache_dir, "train.npz"), allow_pickle=True)
    Xs = z["X"]
    ys = z["y"]
    Ds = z["D"] if "D" in z.files else None

    # concatenate numerically
    X = np.concatenate([x.astype(np.float32) for x in Xs], axis=0)
    n_rows, n_feat = X.shape

    # detect binary looking columns by sampling
    sample = X[:min(20000, n_rows)]
    def is_binary(col):
        u = np.unique(col)
        return np.all(np.isin(u, [0.0, 1.0]))

    bin_mask = np.array([is_binary(sample[:, j]) for j in range(n_feat)], dtype=bool)
    cont_mask = ~bin_mask
    Xc = X[:, cont_mask] if np.any(cont_mask) else np.empty((n_rows, 0), dtype=np.float32)

    if Xc.shape[1] > 0:
        m = Xc.mean(axis=0)
        s = Xc.std(axis=0, ddof=1)
        mean_abs_avg = float(np.mean(np.abs(m)))
        std_avg = float(np.mean(s))
        frac_mean_small = float(np.mean(np.abs(m) < 0.05))
        frac_std_oneish = float(np.mean((s > 0.9) & (s < 1.1)))
    else:
        mean_abs_avg = std_avg = frac_mean_small = frac_std_oneish = float("nan")

    print(f"rows x feat: {n_rows} x {n_feat}")
    print(f"binary columns: {int(bin_mask.sum())}")
    print(f"continuous columns: {int(cont_mask.sum())}")
    print(f"continuous mean abs avg: {mean_abs_avg}")
    print(f"continuous std avg: {std_avg}")
    print(f"fraction with |mean|<0.05: {frac_mean_small}")
    print(f"fraction with 0.9<std<1.1: {frac_std_oneish}")
    if Ds is None:
        print("demographics: missing")
    else:
        print(f"demographics: present, shape={Ds.shape}, nonzero_frac={float(np.mean(np.abs(Ds) > 0)):.3f}")

if __name__ == "__main__":
    main()
