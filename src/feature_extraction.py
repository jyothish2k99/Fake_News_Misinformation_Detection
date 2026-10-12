"""Step 2: linguistic features for the analytics team.

Features are computed on the RAW article text (publisher markers removed), NOT on
the stopword-free cleaned text, so counts such as words, capitals and "!" mean
what they say.

Run: python src/feature_extraction.py
Input : data/processed_news.csv
Output: data/features_news.csv
"""
import re

import pandas as pd
import textstat

from settings import FEATURES_CSV, PROCESSED_CSV
from text_utils import remove_source_markers

data = pd.read_csv(PROCESSED_CSV)
print("Loaded:", data.shape)

# Raw text with publisher fingerprints removed (keeps case + punctuation).
raw = data["text"].fillna("").astype(str).apply(remove_source_markers)

WORD = re.compile(r"[A-Za-z0-9']+")


def words_of(text):
    return WORD.findall(text)


def safe_div(a, b):
    return a / b if b else 0.0


rows = []
for text in raw:
    words = words_of(text)
    n_words = len(words)
    n_letters = sum(c.isalpha() for c in text)
    n_upper = sum(c.isupper() for c in text)
    caps_words = [w for w in words if len(w) >= 2 and w.isalpha() and w.isupper()]
    unique = len({w.lower() for w in words})
    rows.append({
        # --- size / structure
        "word_count": n_words,
        "sentence_count": len(re.findall(r"[.!?]+", text)),
        "article_length": len(text),
        "average_word_length": safe_div(sum(len(w) for w in words), n_words),
        "unique_word_count": unique,
        "lexical_diversity": safe_div(unique, n_words),
        # --- sensational style
        "punctuation_count": len(re.findall(r"[^\w\s]", text)),
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "capital_letter_count": n_upper,
        "capital_letter_ratio": safe_div(n_upper, n_letters),
        "all_caps_word_count": len(caps_words),
        "all_caps_word_ratio": safe_div(len(caps_words), n_words),
        # --- readability
        "flesch_reading_ease": textstat.flesch_reading_ease(text) if n_words else 0.0,
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text) if n_words else 0.0,
    })

features = pd.DataFrame(rows, index=data.index)
features["exclamations_per_100_words"] = (
    100 * features["exclamation_count"] / features["word_count"].replace(0, pd.NA)
).fillna(0)

data = pd.concat([data, features], axis=1)

# ---------------------------------------------------------------- dates
# The date column has mixed formats and some junk values (e.g. URLs).
date_parsed = pd.to_datetime(
    data["date"].astype(str).str.strip(), errors="coerce", format="mixed"
)
bad_dates = int(date_parsed.isna().sum())
print(f"Unparseable dates: {bad_dates} ({bad_dates / len(data):.2%})")
data["date_parsed"] = date_parsed
data["year"] = date_parsed.dt.year
data["month"] = date_parsed.dt.month
data["year_month"] = date_parsed.dt.to_period("M").astype(str).replace("NaT", pd.NA)

# --------------------------------------------------------------- summary
FEATURE_COLS = list(features.columns)
print("\nAverage features by label:")
print(data.groupby("label")[FEATURE_COLS].mean().round(3).T)

data.to_csv(FEATURES_CSV, index=False)
print(f"\nFeature dataset saved to {FEATURES_CSV}")
