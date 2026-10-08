import pandas as pd
import re


# Load the processed dataset
data = pd.read_csv("data/processed_news.csv")


# 1. Word count
data["word_count"] = data["cleaned_text"].apply(
    lambda x: len(str(x).split())
)


# 2. Sentence count
data["sentence_count"] = data["text"].apply(
    lambda x: len(re.findall(r"[.!?]+", str(x)))
)


# 3. Article length
data["article_length"] = data["text"].apply(
    lambda x: len(str(x))
)


# 4. Average word length
data["average_word_length"] = data["cleaned_text"].apply(
    lambda x: sum(len(word) for word in str(x).split())
    / len(str(x).split())
    if len(str(x).split()) > 0 else 0
)


# 5. Punctuation count
data["punctuation_count"] = data["text"].apply(
    lambda x: len(re.findall(r"[^\w\s]", str(x)))
)


# 6. Capital letter count
data["capital_letter_count"] = data["text"].apply(
    lambda x: sum(1 for char in str(x) if char.isupper())
)


# 7. Unique word count
data["unique_word_count"] = data["cleaned_text"].apply(
    lambda x: len(set(str(x).split()))
)


# Display the new features
print("Dataset shape:", data.shape)

print("\nLinguistic features:")
print(
    data[
        [
            "word_count",
            "sentence_count",
            "article_length",
            "average_word_length",
            "punctuation_count",
            "capital_letter_count",
            "unique_word_count"
        ]
    ].head()
)


# Compare features between FAKE and REAL news
print("\nAverage features by label:")
print(
    data.groupby("label")[
        [
            "word_count",
            "sentence_count",
            "article_length",
            "average_word_length",
            "punctuation_count",
            "capital_letter_count",
            "unique_word_count"
        ]
    ].mean()
)


# Save the feature dataset
data.to_csv(
    "data/features_news.csv",
    index=False
)

print("\nFeature dataset saved successfully!")