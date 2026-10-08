import pandas as pd

# Load processed dataset
data = pd.read_csv("data/processed_news.csv")


# ==========================================
# 1. CHECK DUPLICATE CLEANED TEXT
# ==========================================

duplicate_count = data.duplicated(
    subset=["cleaned_text"]
).sum()

print("Duplicate cleaned text rows:", duplicate_count)


# ==========================================
# 2. CHECK SAME TEXT WITH DIFFERENT LABELS
# ==========================================

label_counts = data.groupby(
    "cleaned_text"
)["label"].nunique()

conflicting_texts = label_counts[
    label_counts > 1
]

print(
    "Same text with different labels:",
    len(conflicting_texts)
)


# ==========================================
# 3. CHECK SUBJECT DISTRIBUTION
# ==========================================

print("\nSubject distribution by label:")

print(
    pd.crosstab(
        data["subject"],
        data["label"]
    )
)


# ==========================================
# 4. CHECK REUTERS MARKER
# ==========================================

data["has_reuters"] = data["text"].str.contains(
    r"\(Reuters\)",
    case=False,
    na=False
)

print("\nReuters marker by label:")

print(
    pd.crosstab(
        data["has_reuters"],
        data["label"]
    )
)
# Check exact duplicate article texts
text_duplicate_count = data.duplicated(
    subset=["text"]
).sum()

print("\nExact duplicate article texts:", text_duplicate_count)