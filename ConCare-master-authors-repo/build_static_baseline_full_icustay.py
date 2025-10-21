#!/usr/bin/env python3
# build_static_baseline_full_icustay.py
# Run from your main ConCare project folder (the one containing data/ and data_input/).
#
# Uses:
#   data/train_listfile.csv, data/val_listfile.csv, data/test_listfile.csv
#   data/demographics_per_subject.csv
#   One template demographic CSV (e.g., 99517_episode1.csv) in this same main folder
#   data_input/PATIENTS.csv, data_input/ADMISSIONS.csv, data_input/DIAGNOSES_ICD.csv, data_input/ICUSTAYS.csv
#
# Writes ConCare-style demographics with ICU ID, Length of Stay, mortality label,
# and per-admission diagnosis flags into data/demographic/.

import os, re, csv, pandas as pd

PRINT = lambda *a, **k: print(">>>", *a, **k)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data", "demographic")
DATA_INPUT = os.path.join(BASE_DIR, "data_input")
DATA_DIR = os.path.join(BASE_DIR, "data")

DEMO_CSV = os.path.join(DATA_DIR, "demographics_per_subject.csv")
PATIENTS = os.path.join(DATA_INPUT, "PATIENTS.csv")
ADMISSIONS = os.path.join(DATA_INPUT, "ADMISSIONS.csv")
DIAG_ICD = os.path.join(DATA_INPUT, "DIAGNOSES_ICD.csv")
ICUSTAYS = os.path.join(DATA_INPUT, "ICUSTAYS.csv")

LISTFILES = [
    os.path.join(DATA_DIR, "train_listfile.csv"),
    os.path.join(DATA_DIR, "val_listfile.csv"),
    os.path.join(DATA_DIR, "test_listfile.csv")
]
FILENAME_RX = re.compile(r"(?P<sid>\d+)_episode(?P<ep>\d+)_timeseries\.csv$", re.I)

def die(msg):
    PRINT(msg)
    raise SystemExit(1)

def find_template():
    for name in os.listdir(BASE_DIR):
        if "_episode" in name and name.endswith(".csv") and "_timeseries" not in name:
            return os.path.join(BASE_DIR, name)
    return None

def load_template_header(path):
    df = pd.read_csv(path, nrows=1)
    cols = list(df.columns)
    if len(cols) < 8:
        die("Template CSV has too few columns.")
    return cols[:8], cols[8:]

def to_empty(x):
    if x is None: return ""
    if isinstance(x, str):
        s = x.strip()
        return "" if s.lower() in {"nan", "none", "null"} else s
    try:
        return "" if pd.isna(x) else x
    except Exception:
        return ""

def safe_int(x):
    x = to_empty(x)
    if x == "": return ""
    try: return int(float(x))
    except Exception: return ""

def load_pairs(listfile):
    if not os.path.isfile(listfile): return []
    df = pd.read_csv(listfile)
    col_fn = "stay" if "stay" in df.columns else df.columns[0]
    col_y = "y_true" if "y_true" in df.columns else None
    out = []
    for _, r in df.iterrows():
        m = FILENAME_RX.search(os.path.basename(str(r[col_fn])))
        if not m: continue
        sid, ep = m.group("sid"), m.group("ep")
        y = int(r[col_y]) if col_y in df.columns else ""
        out.append((sid, ep, y))
    return out

def main():
    PRINT("Building ConCare paper-style demographics with ICUSTAY mapping")
    for p in [DEMO_CSV, PATIENTS, ADMISSIONS, DIAG_ICD, ICUSTAYS]:
        if not os.path.isfile(p):
            die(f"Missing file: {p}")
    tmpl = find_template()
    if tmpl is None:
        die("Place one template demographics episode CSV (e.g., 99517_episode1.csv) in this folder.")
    PRINT("Using template:", os.path.basename(tmpl))
    base8, diag_cols = load_template_header(tmpl)
    header = base8 + diag_cols

    demo = pd.read_csv(DEMO_CSV)
    cols = {c.lower(): c for c in demo.columns}
    for need in ["subject_id","gender_code","ethnicity_code","age","height_cm","weight_kg"]:
        if need not in cols:
            die(f"Missing column '{need}' in demographics_per_subject.csv")
    sid_col, g_col, e_col, a_col, h_col, w_col = (
        cols["subject_id"], cols["gender_code"], cols["ethnicity_code"],
        cols["age"], cols["height_cm"], cols["weight_kg"]
    )
    demo[sid_col] = demo[sid_col].astype(str).str.replace(r"\.0$","",regex=True)
    demo_lu = {str(r[sid_col]): r for _, r in demo.iterrows()}

    icu = pd.read_csv(ICUSTAYS, usecols=["SUBJECT_ID","HADM_ID","ICUSTAY_ID","INTIME","OUTTIME","LOS"])
    icu["SUBJECT_ID"] = icu["SUBJECT_ID"].astype(int).astype(str)
    icu = icu.sort_values(["SUBJECT_ID","INTIME"])
    icu["episode"] = icu.groupby("SUBJECT_ID").cumcount() + 1
    icu_key = {(str(r.SUBJECT_ID), str(int(r.episode))): 
               (int(r.ICUSTAY_ID), str(int(r.HADM_ID)) if pd.notna(r.HADM_ID) else "", 
                float(r.LOS) if pd.notna(r.LOS) else "")
               for _, r in icu.iterrows()}

    diag = pd.read_csv(DIAG_ICD, usecols=["HADM_ID","ICD9_CODE"])
    diag["HADM_ID"] = diag["HADM_ID"].astype(int).astype(str)
    diag["ICD9_CODE"] = diag["ICD9_CODE"].astype(str).str.replace(r"\.0$","",regex=True)
    icd_by_hadm = diag.groupby("HADM_ID")["ICD9_CODE"].apply(lambda s: set(s.dropna().astype(str))).to_dict()

    wanted_codes = [re.match(r"Diagnosis\s+(\S+)", str(c)).group(1) if re.match(r"Diagnosis\s+(\S+)", str(c)) else None for c in diag_cols]

    triples = []
    for lf in LISTFILES:
        triples.extend(load_pairs(lf))
    seen = {}
    for sid, ep, y in triples:
        seen.setdefault((sid,ep), y)
    pairs = [(sid,ep,seen[(sid,ep)]) for sid,ep in sorted(seen.keys(), key=lambda x:(int(x[0]), int(x[1])))]

    os.makedirs(OUT_DIR, exist_ok=True)

    written, missing_map = 0, 0
    for sid, ep, y in pairs:
        key = (str(sid), str(ep))
        icustay_id, hadm_id, los = icu_key.get(key, ("","",""))
        if icustay_id == "": missing_map += 1
        drow = demo_lu.get(str(sid))
        vals = {
            base8[0]: icustay_id if icustay_id != "" else f"{sid}_{ep}",
            base8[1]: safe_int(drow[e_col]) if drow is not None else "",
            base8[2]: safe_int(drow[g_col]) if drow is not None else "",
            base8[3]: to_empty(drow[a_col]) if drow is not None else "",
            base8[4]: to_empty(drow[h_col]) if drow is not None else "",
            base8[5]: to_empty(drow[w_col]) if drow is not None else "",
            base8[6]: los,
            base8[7]: int(y) if y != "" else ""
        }
        hadm_codes = icd_by_hadm.get(str(hadm_id), set()) if hadm_id != "" else set()
        diag_vector = [(1 if (code in hadm_codes) else 0) if code else 0 for code in wanted_codes]

        out_row = [vals.get(col, diag_vector[diag_cols.index(col)] if col in diag_cols else "") for col in header]
        out_path = os.path.join(OUT_DIR, f"{sid}_episode{ep}.csv")
        with open(out_path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(header)
            wr.writerow(out_row)
        written += 1

    PRINT(f"Wrote {written} files to {OUT_DIR}. Episodes missing ICU mapping: {missing_map}")
    if missing_map:
        PRINT("Note: some episodes could not be mapped to ICUSTAY_ID by nth stay order.")

if __name__ == "__main__":
    main()
