# build_demographics.py — place & run inside .../Concare_reproducibility/data
import os, re, csv, pandas as pd

print("=== ConCare Demographic Builder (listfiles, no inputs) ===")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_CSV = os.path.join(BASE_DIR, "demographics_per_subject.csv")
OUT_DIR = os.path.join(BASE_DIR, "demographic")
LISTFILES = ["train_listfile.csv", "val_listfile.csv", "test_listfile.csv"]

HEADER_BASE = ["Icustay", "Gender", "Ethnicity", "Age", "Height", "Weight"]
DIAG_FIXED = [f"Diagnosis {i:03d}" for i in range(128)]
FILENAME_RX = re.compile(r"(?P<sid>\d+)_episode(?P<ep>\d+)_timeseries\.csv$", re.I)

def to_empty(x):
    if x is None: return ""
    if isinstance(x, str):
        s = x.strip()
        return "" if s.lower() in {"nan","none","null"} else s
    try:
        return "" if pd.isna(x) else x
    except Exception:
        return ""

def safe_int(x):
    x = to_empty(x)
    if x == "": return ""
    try: return int(float(x))
    except Exception: return ""

def load_pairs_from_listfile(path):
    pairs = []
    if not os.path.isfile(path):
        return pairs
    df = pd.read_csv(path)
    col = "stay" if "stay" in df.columns else df.columns[0]
    for name in df[col].astype(str).values:
        m = FILENAME_RX.search(os.path.basename(name))
        if m:
            pairs.append((m.group("sid"), m.group("ep")))
    return pairs

def main():
    if not os.path.isfile(DEMO_CSV):
        print("❌ demographics_per_subject.csv not found next to listfiles")
        return
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load demographics per subject id
    demo = pd.read_csv(DEMO_CSV)
    cols = {c.lower(): c for c in demo.columns}
    req = ["subject_id","gender_code","ethnicity_code","age","height_cm","weight_kg"]
    for r in req:
        if r not in cols:
            raise ValueError(f"Missing column '{r}' in demographics_per_subject.csv")
    sid_col = cols["subject_id"]; g_col = cols["gender_code"]; e_col = cols["ethnicity_code"]
    a_col = cols["age"]; h_col = cols["height_cm"]; w_col = cols["weight_kg"]
    demo[sid_col] = demo[sid_col].astype(str).str.replace(r"\.0$","",regex=True)
    demo_lu = {str(r[sid_col]): r for _, r in demo.iterrows()}

    # Collect unique (subject, episode) from listfiles
    pairs = []
    for lf in LISTFILES:
        lf_path = os.path.join(BASE_DIR, lf)
        got = load_pairs_from_listfile(lf_path)
        print(f"Found {len(got)} episodes listed in {lf}")
        pairs.extend(got)
    pairs = sorted(set(pairs), key=lambda x: (int(x[0]), int(x[1])))
    print(f"Total unique (subject, episode): {len(pairs)}")

    header = HEADER_BASE + DIAG_FIXED
    written = 0
    for sid, ep in pairs:
        row = demo_lu.get(str(sid))
        g = safe_int(row[g_col]) if row is not None else ""
        e = safe_int(row[e_col]) if row is not None else ""
        a = to_empty(row[a_col]) if row is not None else ""
        h = to_empty(row[h_col]) if row is not None else ""
        w = to_empty(row[w_col]) if row is not None else ""

        out_path = os.path.join(OUT_DIR, f"{sid}_episode{ep}.csv")
        with open(out_path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(header)
            wr.writerow([f"{sid}_{ep}", g, e, a, h, w] + [0]*128)
        written += 1

    print(f"✅ Wrote {written} demographic files to {OUT_DIR}")

if __name__ == "__main__":
    main()
