import pandas as pd, os

base = "data/in-hospital-mortality"
for split in ["train", "val", "test"]:
    df = pd.read_csv(f"{base}/{split}_listfile.csv")
    print(f"\n=== {split.upper()} ===")
    print(f"Samples: {len(df):,}")
    print(f"Positives: {df['y_true'].sum()} ({100*df['y_true'].mean():.2f}%)")
    # Check that all listed files actually exist
    missing = [f for f in df['stay'] if not os.path.exists(f"{base}/{split}/{f}")]
    print(f"Missing episode files: {len(missing)}")


