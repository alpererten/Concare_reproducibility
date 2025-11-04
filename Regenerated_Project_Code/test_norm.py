import numpy as np

z = np.load("data/normalized_data_cache/train.npz", allow_pickle=True)
Xs = z["X"]  # list of [T_i, F]

X = np.concatenate([x.astype(np.float32) for x in Xs], axis=0)


# identify likely binary columns to exclude from the check
col_is_binary = []
for j in range(X.shape[1]):
    vals = X[:10000, j]  # sample
    uniq = np.unique(vals)
    col_is_binary.append(np.all(np.isin(uniq, [0.0, 1.0])))

mask_cont = ~np.array(col_is_binary)
Xc = X[:, mask_cont]

m = Xc.mean(axis=0)
s = Xc.std(axis=0, ddof=1)

print("continuous cols:", Xc.shape[1])
print("mean abs avg:", float(np.mean(np.abs(m))))
print("std avg:", float(np.mean(s)))
print("fraction with |mean|<0.05:", float(np.mean(np.abs(m) < 0.05)))
print("fraction with 0.9< std <1.1:", float(np.mean((s > 0.9) & (s < 1.1))))
