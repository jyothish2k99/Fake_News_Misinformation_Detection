"""Step 1: load raw CSVs -> label -> remove duplicates -> clean text -> save.

Run from the repo root:
    python src/preprocessing.py
Input : data/True.csv, data/Fake.csv
Output: data/processed_news.csv
"""
import pandas as pd

from settings import PROCESSED_CSV, RAW_FAKE, RAW_TRUE
from text_utils import preprocess_text

MIN_CLEAN_WORDS = 5   # drop articles that are (almost) empty after cleaning

# ------------------------------------------------------------------ load
true_data = pd.read_csv(RAW_TRUE)
fake_data = pd.read_csv(RAW_FAKE)
true_data["label"] = "REAL"
fake_data["label"] = "FAKE"

data = pd.concat([true_data, fake_data], ignore_index=True)
print("Loaded rows:", len(data))

# ------------------------------------------------ duplicates (raw text)
data = data.drop_duplicates()
data = data.drop_duplicates(subset=["text"])
print("After removing duplicate rows / raw texts:", len(data))

# ------------------------------------------------------------- cleaning
data["cleaned_text"] = data["text"].apply(preprocess_text)

# Some articles are empty (title only) or become empty after cleaning.
data["clean_word_count"] = data["cleaned_text"].str.split().str.len()
before = len(data)
data = data[data["clean_word_count"] >= MIN_CLEAN_WORDS]
print(f"Dropped {before - len(data)} articles with < {MIN_CLEAN_WORDS} words after cleaning")

# ------------------------- duplicates AFTER cleaning (prevents train/test leak)
before = len(data)
data = data.drop_duplicates(subset=["cleaned_text", "label"])
print("Removed near-duplicates (same cleaned text, same label):", before - len(data))

# Same cleaned text with BOTH labels is ambiguous -> remove all copies.
label_per_text = data.groupby("cleaned_text")["label"].transform("nunique")
conflicts = int((label_per_text > 1).sum())
data = data[label_per_text == 1]
print("Removed rows whose text appears with conflicting labels:", conflicts)

data = data.drop(columns=["clean_word_count"]).reset_index(drop=True)
data.insert(0, "article_id", data.index)   # stable id, reused for train/test splits

# ---------------------------------------------------------------- report
print("\nFinal dataset shape:", data.shape)
print("\nLabel distribution:")
print(data["label"].value_counts())
print("\nOriginal text:\n", data["text"].iloc[0][:300])
print("\nCleaned text:\n", data["cleaned_text"].iloc[0][:300])

data.to_csv(PROCESSED_CSV, index=False)
print(f"\nProcessed dataset saved to {PROCESSED_CSV}")
