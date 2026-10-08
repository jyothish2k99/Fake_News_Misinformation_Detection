import pandas as pd
import matplotlib.pyplot as plt


# Load feature dataset
data = pd.read_csv("data/features_news.csv")


# ==============================
# 1. FAKE vs REAL DISTRIBUTION
# ==============================

label_counts = data["label"].value_counts()

plt.figure(figsize=(6, 4))
label_counts.plot(kind="bar")

plt.title("Fake vs Real News Distribution")
plt.xlabel("News Label")
plt.ylabel("Number of Articles")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("data/fake_vs_real_distribution.png")
plt.show()


# ==============================
# 2. WORD COUNT DISTRIBUTION
# ==============================

plt.figure(figsize=(8, 5))

data[data["label"] == "FAKE"]["word_count"].plot(
    kind="hist",
    bins=30,
    alpha=0.6,
    label="FAKE"
)

data[data["label"] == "REAL"]["word_count"].plot(
    kind="hist",
    bins=30,
    alpha=0.6,
    label="REAL"
)

plt.title("Word Count Distribution")
plt.xlabel("Word Count")
plt.ylabel("Number of Articles")
plt.legend()

plt.tight_layout()
plt.savefig("data/word_count_distribution.png")
plt.show()


# ==============================
# 3. ARTICLE LENGTH COMPARISON
# ==============================

average_length = data.groupby("label")["article_length"].mean()

plt.figure(figsize=(6, 4))
average_length.plot(kind="bar")

plt.title("Average Article Length: Fake vs Real")
plt.xlabel("News Label")
plt.ylabel("Average Article Length")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("data/article_length_comparison.png")
plt.show()


# ==============================
# 4. UNIQUE WORD COUNT
# ==============================

average_unique_words = data.groupby("label")["unique_word_count"].mean()

plt.figure(figsize=(6, 4))
average_unique_words.plot(kind="bar")

plt.title("Average Unique Word Count: Fake vs Real")
plt.xlabel("News Label")
plt.ylabel("Average Unique Words")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("data/unique_word_comparison.png")
plt.show()


# ==============================
# 5. PRINT EDA SUMMARY
# ==============================

print("\n========== EDA SUMMARY ==========")

print("\nNews distribution:")
print(label_counts)

print("\nAverage linguistic features:")
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

print("\nEDA completed successfully!")
