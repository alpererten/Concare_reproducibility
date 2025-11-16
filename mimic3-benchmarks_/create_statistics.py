import os, json, pandas as pd, glob

train_path = "data/root/train"
files = glob.glob(os.path.join(train_path, "*.csv"))

stats = {}
for f in files:
    df = pd.read_csv(f)
    for col in df.columns:
        if col == "Hours":  # skip time column
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        if col not in stats:
            stats[col] = []
        stats[col].append(series)

final = {}
for col, vals in stats.items():
    merged = pd.concat(vals)
    final[col] = {
        "mean": float(merged.mean()),
        "std": float(merged.std()),
        "min": float(merged.min()),
        "max": float(merged.max()),
        "count": int(merged.count())
    }

with open("data/statistics.json", "w") as f:
    json.dump(final, f, indent=2)

print("statistics.json created at data/")

