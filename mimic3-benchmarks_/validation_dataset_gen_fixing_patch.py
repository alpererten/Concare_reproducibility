import os, shutil, pandas as pd

base = "data/in-hospital-mortality"
df = pd.read_csv(f"{base}/val_listfile.csv")

src_dir = f"{base}/train"   # the files are physically in train/
dst_dir = f"{base}/val"
os.makedirs(dst_dir, exist_ok=True)

for fname in df["stay"]:
    src = os.path.join(src_dir, fname)
    dst = os.path.join(dst_dir, fname)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy(src, dst)

print("Copied", len(df), "validation episode files.")


