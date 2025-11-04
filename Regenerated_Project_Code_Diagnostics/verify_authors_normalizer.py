import os, pickle, numpy as np
from authors_modules.preprocessing import Discretizer as ADisc
from authors_modules.readers import InHospitalMortalityReader as IHMReader

def compute_train_stats():
    disc = ADisc(timestep=0.8, store_masks=True, impute_strategy="previous",
                 start_time="zero", config_path=os.path.join("data","discretizer_config.json"))
    rdr = IHMReader(dataset_dir="data/train", listfile="data/train_listfile.csv")
    sum_x = None; sum_sq = None; N = 0; F = None
    for i in range(rdr.get_number_of_examples()):
        ex = rdr.read_example(i)
        Xd, _ = disc.transform(ex["X"], header=ex["header"], end=ex.get("t", None))
        Xd = Xd.astype(np.float32)
        if F is None:
            F = Xd.shape[1]
            sum_x = np.zeros(F, np.float64); sum_sq = np.zeros(F, np.float64)
        sum_x += Xd.sum(axis=0, dtype=np.float64)
        sum_sq += (Xd.astype(np.float64)**2).sum(axis=0, dtype=np.float64)
        N += Xd.shape[0]
    means = (sum_x / max(N,1)).astype(np.float32)
    var = (sum_sq - 2*sum_x*means + N*(means.astype(np.float64)**2)) / max(N-1,1)
    stds = np.sqrt(np.maximum(var, 1e-14)).astype(np.float32)
    stds[stds < 1e-7] = 1e-7
    return means, stds

with open("data/ihm_normalizer","rb") as f:
    d = pickle.load(f, encoding="latin1")
m_ref, s_ref = d["means"].astype(np.float32), d["stds"].astype(np.float32)
m_tr, s_tr = compute_train_stats()

def report(name, a, b):
    diff = a - b
    print(name,
          "L2:", float(np.linalg.norm(diff)),
          "max_abs:", float(np.max(np.abs(diff))),
          "mean_abs:", float(np.mean(np.abs(diff))))

report("means   diff", m_ref, m_tr)
report("stds    diff", s_ref, s_tr)
