
import gzip, shutil, glob, os

src = r"data_input"
gz_files = sorted(glob.glob(os.path.join(src, "*.csv.gz")))
if not gz_files:
    raise SystemExit("No .csv.gz files found in data_input/. Point me to the right folder.")

for gz in gz_files:
    out = gz[:-3]  # strip .gz -> .csv
    if os.path.exists(out):
        print(f"skip (exists): {os.path.basename(out)}")
        continue
    print(f"decompressing: {os.path.basename(gz)} -> {os.path.basename(out)}")
    with gzip.open(gz, "rb") as f_in, open(out, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

# quick sanity check for a few critical tables
must_have = ["PATIENTS.csv","ADMISSIONS.csv","ICUSTAYS.csv","DIAGNOSES_ICD.csv",
             "D_ICD_DIAGNOSES.csv","PROCEDURES_ICD.csv","D_ICD_PROCEDURES.csv",
             "CHARTEVENTS.csv","LABEVENTS.csv"]
missing = [f for f in must_have if not os.path.exists(os.path.join(src,f))]
if missing:
    print("\nMissing after decompression:", ", ".join(missing))
else:
    print("\nAll core CSVs present.")
print("done.")

