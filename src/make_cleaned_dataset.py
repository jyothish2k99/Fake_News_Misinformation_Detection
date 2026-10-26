"""Build data/processed/cleaned_dataset.csv (row_id, label, cleaned_text) from AI/ML-01's file.

AI/ML-01's processed_news.csv has no row_id and uses REAL/FAKE strings. This script
attaches the row_id (and integer label) from master.csv by matching title + text exactly,
and refuses to continue if anything looks wrong.

Run from the project root:
    python -m src.make_cleaned_dataset
"""
import pandas as pd

from src.config import INTERIM_DIR, PROCESSED_DIR

SRC_FILE = PROCESSED_DIR / "processed_news.csv"
OUT_FILE = PROCESSED_DIR / "cleaned_dataset.csv"


def main():
    master = pd.read_csv(INTERIM_DIR / "master.csv", usecols=["row_id", "title", "text", "label"])
    proc = pd.read_csv(SRC_FILE, usecols=["title", "text", "label", "cleaned_text"])
    for d in (master, proc):
        d[["title", "text"]] = d[["title", "text"]].fillna("")
    proc["cleaned_text"] = proc["cleaned_text"].fillna("")

    merged = proc.merge(master, on=["title", "text"], how="left", suffixes=("_proc", ""))
    assert merged["row_id"].notna().all(), "some rows of processed_news do not exist in master.csv"
    assert merged["row_id"].is_unique, "a row_id matched more than once"
    label_proc = merged["label_proc"].map({"FAKE": 1, "REAL": 0})
    assert (label_proc == merged["label"]).all(), "labels disagree between the two files"

    out = merged[["row_id", "label", "cleaned_text"]].astype({"row_id": int, "label": int})
    out = out.sort_values("row_id").reset_index(drop=True)
    out.to_csv(OUT_FILE, index=False)

    print(f"Saved {len(out)} rows to {OUT_FILE}")
    print(f"master.csv has {len(master)} rows, so {len(master) - len(out)} articles are missing")
    gone = master[~master["row_id"].isin(out["row_id"])]
    print("Missing by label (0=Real, 1=Fake):", gone["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()