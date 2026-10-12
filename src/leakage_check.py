"""Step 0b: check that nothing in the data lets a model "cheat".

Run AFTER preprocessing: python src/leakage_check.py
Input : data/processed_news.csv
Output: printed report + results/leakage_report.csv
"""
import pandas as pd

from settings import PROCESSED_CSV, RESULTS_DIR

data = pd.read_csv(PROCESSED_CSV)
print("Rows:", len(data))

# 1. Duplicates / conflicting labels -----------------------------------------
print("\n1. Duplicate cleaned texts      :", data.duplicated(subset=["cleaned_text"]).sum())
print("   Duplicate raw texts          :", data.duplicated(subset=["text"]).sum())
conflicts = (data.groupby("cleaned_text")["label"].nunique() > 1).sum()
print("   Same text, different labels  :", conflicts)

# 2. Subject column (NOT used for the model, but it separates the classes) ---
print("\n2. Subject distribution by label (do not use 'subject' as a model input):")
print(pd.crosstab(data["subject"], data["label"]))

# 3. Publisher fingerprints: raw text vs cleaned text ------------------------
MARKERS = {
    "reuters": r"reuters",
    "washington": r"washington",
    "said on <weekday>": r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    "featured image": r"featured image",
    "image via": r"image via",
    "read more": r"read more",
    "21st century wire": r"21st century wire",
    "getty": r"getty",
    "pic twitter": r"pic\.twitter",
    "video": r"\bvideo\b",
}
rows = []
for name, pattern in MARKERS.items():
    row = {"marker": name}
    for col, tag in (("text", "raw"), ("cleaned_text", "clean")):
        has = data[col].fillna("").str.contains(pattern, case=False, regex=True)
        for label in ("FAKE", "REAL"):
            row[f"{tag}_{label}_%"] = round(100 * has[data["label"] == label].mean(), 2)
    rows.append(row)

report = pd.DataFrame(rows)
report.to_csv(RESULTS_DIR / "leakage_report.csv", index=False)
print("\n3. % of articles containing each marker (raw text vs after cleaning):")
print(report.to_string(index=False))

# 4. Verdict ------------------------------------------------------------------
flag = report[
    ((report["clean_FAKE_%"] - report["clean_REAL_%"]).abs() > 5)
]
print("\n4. Verdict")
if flag.empty:
    print("   OK - no marker differs by more than 5 percentage points after cleaning.")
else:
    print("   WARNING - these markers still separate FAKE from REAL after cleaning:")
    print("  ", ", ".join(flag["marker"]))
    print("   Add patterns to text_utils.py (or accept them knowingly and document it).")
