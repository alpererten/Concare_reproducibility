"""
Extract ConCare-style demographics per SUBJECT_ID from MIMIC-III CSVs
stored in data_input/.

Reads:
  data_input/PATIENTS.csv
  data_input/ADMISSIONS.csv
  data_input/CHARTEVENTS.csv

Writes:
  data_input/demographics_per_subject.csv

Columns:
  SUBJECT_ID, AGE, GENDER_CODE, ETHNICITY_CODE, HEIGHT_CM, WEIGHT_KG
"""

import os
import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = os.path.join(os.path.dirname(__file__), "data_input")
PATH_PATIENTS = os.path.join(BASE_DIR, "PATIENTS.csv")
PATH_ADMISSIONS = os.path.join(BASE_DIR, "ADMISSIONS.csv")
PATH_CHARTEVENTS = os.path.join(BASE_DIR, "CHARTEVENTS.csv")
OUT_PATH = os.path.join("data", "demographics_per_subject.csv")

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
HEIGHT_ITEMIDS = {763, 226512}                 # Height
WEIGHT_ITEMIDS = {762, 5802, 226531, 224639}   # Weight

# ------------------------------------------------------------------
# Mappers and converters
# ------------------------------------------------------------------
def map_gender_code(g):
    s = str(g).strip().upper()
    if s.startswith("M"):
        return 1
    if s.startswith("F"):
        return 2
    return 0  # unknown or other

def map_ethnicity_code(e):
    s = str(e).upper()
    if "ASIAN" in s:
        return 0
    if "BLACK" in s or "AFRICAN" in s:
        return 1
    if "HISP" in s or "LATINO" in s:
        return 2
    if "WHITE" in s or "CAUCASIAN" in s:
        return 3
    return 4

def normalize_height(value, uom):
    if pd.isna(value):
        return np.nan
    u = str(uom).lower()
    if "in" in u:
        return float(value) * 2.54
    return float(value)  # assume cm

def normalize_weight(value, uom):
    if pd.isna(value):
        return np.nan
    u = str(uom).lower()
    if "lb" in u:
        return float(value) * 0.453592
    return float(value)  # assume kg

# ------------------------------------------------------------------
# Processing functions
# ------------------------------------------------------------------
def compute_age_at_first_admit(patients, admissions):
    admissions = admissions.copy()
    admissions["ADMITTIME"] = pd.to_datetime(admissions["ADMITTIME"], errors="coerce", utc=True).dt.tz_localize(None)
    first_admit = (
        admissions.sort_values(["SUBJECT_ID", "ADMITTIME"])
        .groupby("SUBJECT_ID", as_index=False)
        .first()
    )

    pts = patients[["SUBJECT_ID", "GENDER", "DOB"]].copy()
    pts["DOB"] = pd.to_datetime(pts["DOB"], errors="coerce", utc=True).dt.tz_localize(None)
    base = first_admit.merge(pts, on="SUBJECT_ID", how="left")

    def compute_age_row(row):
        admit = row["ADMITTIME"]
        dob = row["DOB"]
        if pd.isna(admit) or pd.isna(dob):
            return np.nan
        age_years = admit.year - dob.year - ((admit.month, admit.day) < (dob.month, dob.day))
        return float(age_years)

    base["AGE"] = base.apply(compute_age_row, axis=1).astype(float)
    base["AGE"] = base["AGE"].clip(lower=0, upper=90)

    base["GENDER_CODE"] = base["GENDER"].map(map_gender_code).astype("Int64")
    base["ETHNICITY_CODE"] = base["ETHNICITY"].map(map_ethnicity_code).astype("Int64")
    return base[["SUBJECT_ID", "AGE", "GENDER_CODE", "ETHNICITY_CODE"]]


def compute_height_weight(chartevents_path=PATH_CHARTEVENTS, chunksize=500_000):
    """
    Stream CHARTEVENTS, parse numeric from VALUENUM or VALUE,
    convert units, filter plausible ranges, take median per SUBJECT_ID.
    """
    usecols = ["SUBJECT_ID", "ITEMID", "CHARTTIME", "VALUENUM", "VALUE", "VALUEUOM"]
    dtype = {
        "SUBJECT_ID": "int32",
        "ITEMID": "int32",
        # IMPORTANT: read VALUENUM as float64 to avoid casting errors when merging with fallback
        "VALUENUM": "float64",
        "VALUE": "string",
        "VALUEUOM": "category",
    }

    target_itemids = HEIGHT_ITEMIDS.union(WEIGHT_ITEMIDS)

    height_rows = []   # frames to concat at the end
    weight_rows = []

    total_rows = 0
    kept_height = 0
    kept_weight = 0

    for chunk in pd.read_csv(
        chartevents_path,
        usecols=usecols,
        dtype=dtype,
        parse_dates=False,   # CHARTTIME not needed for median
        chunksize=chunksize,
        low_memory=True
    ):
        total_rows += len(chunk)
        print(f"    -> Scanned {total_rows:,} rows", flush=True)

        # keep only target items
        chunk = chunk[chunk["ITEMID"].isin(target_itemids)]
        if chunk.empty:
            continue

        # Build a numeric series: prefer VALUENUM; fallback to number extracted from VALUE (e.g., "170 cm")
        # Make both float64 to avoid dtype conflicts.
        num_val = chunk["VALUENUM"].astype("float64")
        extracted = chunk["VALUE"].str.extract(r"([-+]?\d*\.?\d+)", expand=False)  # first number in text
        fallback = pd.to_numeric(extracted, errors="coerce").astype("float64")
        num = num_val.where(~num_val.isna(), fallback)

        is_height = chunk["ITEMID"].isin(HEIGHT_ITEMIDS)
        is_weight = ~is_height

        # ----- Heights to cm -----
        if is_height.any():
            h = pd.DataFrame({
                "SUBJECT_ID": chunk.loc[is_height, "SUBJECT_ID"].values,
                "VALUEUOM":   chunk.loc[is_height, "VALUEUOM"].astype("string").str.lower().values,
                "NUM":        num.loc[is_height].values,  # already float64
            })
            # unit conversion (in -> cm)
            mask_in = h["VALUEUOM"].str.contains("in", na=False)
            h.loc[mask_in, "NUM"] = h.loc[mask_in, "NUM"] * 2.54
            # plausible range
            h = h[(h["NUM"] >= 100.0) & (h["NUM"] <= 230.0)]
            kept_height += len(h)
            if not h.empty:
                height_rows.append(h[["SUBJECT_ID", "NUM"]].rename(columns={"NUM": "HEIGHT_CM"}))

        # ----- Weights to kg -----
        if is_weight.any():
            w = pd.DataFrame({
                "SUBJECT_ID": chunk.loc[is_weight, "SUBJECT_ID"].values,
                "VALUEUOM":   chunk.loc[is_weight, "VALUEUOM"].astype("string").str.lower().values,
                "NUM":        num.loc[is_weight].values,  # already float64
            })
            # unit conversion (lb -> kg)
            mask_lb = w["VALUEUOM"].str.contains("lb", na=False)
            w.loc[mask_lb, "NUM"] = w.loc[mask_lb, "NUM"] * 0.453592
            # plausible range
            w = w[(w["NUM"] >= 30.0) & (w["NUM"] <= 250.0)]
            kept_weight += len(w)
            if not w.empty:
                weight_rows.append(w[["SUBJECT_ID", "NUM"]].rename(columns={"NUM": "WEIGHT_KG"}))

    # Combine and aggregate medians per subject
    if height_rows:
        h_all = pd.concat(height_rows, ignore_index=True)
        h_med = h_all.groupby("SUBJECT_ID", as_index=False)["HEIGHT_CM"].median()
    else:
        h_med = pd.DataFrame(columns=["SUBJECT_ID", "HEIGHT_CM"])

    if weight_rows:
        w_all = pd.concat(weight_rows, ignore_index=True)
        w_med = w_all.groupby("SUBJECT_ID", as_index=False)["WEIGHT_KG"].median()
    else:
        w_med = pd.DataFrame(columns=["SUBJECT_ID", "WEIGHT_KG"])

    agg = pd.merge(h_med, w_med, on="SUBJECT_ID", how="outer")
    agg["HEIGHT_CM"] = agg["HEIGHT_CM"].astype("float64")
    agg["WEIGHT_KG"] = agg["WEIGHT_KG"].astype("float64")

    # Coverage summary
    print(f"    -> Height rows kept: {kept_height:,}", flush=True)
    print(f"    -> Weight rows kept: {kept_weight:,}", flush=True)
    print(f"    -> Subjects with height: {h_med['SUBJECT_ID'].nunique():,}", flush=True)
    print(f"    -> Subjects with weight: {w_med['SUBJECT_ID'].nunique():,}", flush=True)
    print(f"    -> Subjects with at least one of height/weight: {agg['SUBJECT_ID'].nunique():,}", flush=True)

    return agg

# ------------------------------------------------------------------
# Main execution
# ------------------------------------------------------------------
def main():
    print("=== ConCare Demographics Extraction ===")
    print("Reading input CSVs from:", BASE_DIR)

    # Step 1 – Load PATIENTS
    print("[1/6] Loading PATIENTS.csv ...", flush=True)
    patients = pd.read_csv(PATH_PATIENTS, low_memory=False)
    print(f"    -> Loaded {len(patients):,} patient rows", flush=True)

    # Step 2 – Load ADMISSIONS
    print("[2/6] Loading ADMISSIONS.csv ...", flush=True)
    admissions = pd.read_csv(PATH_ADMISSIONS, low_memory=False)
    print(f"    -> Loaded {len(admissions):,} admission rows", flush=True)

    # Step 3 – Stream CHARTEVENTS
    print("[3/6] Loading CHARTEVENTS.csv in streaming mode ...", flush=True)
    print("    -> Extracting height and weight per subject while scanning", flush=True)
    hw = compute_height_weight(PATH_CHARTEVENTS)
    print(f"    -> Extracted height/weight for {len(hw):,} subjects", flush=True)

    # Step 4 – Compute base demographics
    print("[4/6] Computing age, gender, ethnicity per subject ...", flush=True)
    demo_base = compute_age_at_first_admit(patients, admissions)
    print(f"    -> Computed base demographics for {len(demo_base):,} subjects", flush=True)

    # Step 5 – Merge and finalize
    print("[5/6] Merging results and imputing missing values ...", flush=True)
    demo = demo_base.merge(hw, on="SUBJECT_ID", how="left")
    #demo["HEIGHT_CM"] = demo["HEIGHT_CM"].fillna(160.0)
    #demo["WEIGHT_KG"] = demo["WEIGHT_KG"].fillna(60.0)
    demo = demo.sort_values("SUBJECT_ID").reset_index(drop=True)

    # Step 6 – Save
    print("[6/6] Writing output ...", flush=True)
    os.makedirs(BASE_DIR, exist_ok=True)
    demo.to_csv(OUT_PATH, index=False)
    print(f"=== DONE: wrote {len(demo):,} rows to {OUT_PATH} ===", flush=True)

if __name__ == "__main__":
    main()
