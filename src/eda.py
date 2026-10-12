"""Step 3: EDA charts + summary table (Fake vs Real).

Run: python src/eda.py
Input : data/features_news.csv
Output: results/figures/*.png, results/eda_summary.csv
"""
import matplotlib
matplotlib.use("Agg")                     # save charts without opening windows
import matplotlib.pyplot as plt
import pandas as pd

from settings import FEATURES_CSV, FIGURES_DIR, RESULTS_DIR

data = pd.read_csv(FEATURES_CSV)
COLORS = {"FAKE": "#d62728", "REAL": "#1f77b4"}


def save(name):
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / name, dpi=150)
    plt.close()


# 1. Class balance --------------------------------------------------------
counts = data["label"].value_counts()
plt.figure(figsize=(5, 4))
plt.bar(counts.index, counts.values, color=[COLORS[l] for l in counts.index])
plt.title("Fake vs Real News Distribution")
plt.ylabel("Number of articles")
save("fake_vs_real_distribution.png")

# 2. Histogram of a feature for both classes ------------------------------
def hist(col, title, xlabel, clip_q=0.99, bins=40):
    plt.figure(figsize=(7, 4.5))
    upper = data[col].quantile(clip_q)            # clip extreme outliers
    for label in ("FAKE", "REAL"):
        vals = data.loc[data["label"] == label, col].clip(upper=upper)
        plt.hist(vals, bins=bins, alpha=0.55, label=label, color=COLORS[label])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of articles")
    plt.legend()
    save(f"hist_{col}.png")


hist("word_count", "Word Count Distribution", "Words")
hist("flesch_reading_ease", "Reading Ease (higher = easier)", "Flesch reading ease")
hist("capital_letter_ratio", "Capital Letter Ratio", "Capitals / letters")

# 3. Boxplots of the sensational-style features ---------------------------
BOX = {
    "word_count": "Words per article",
    "exclamation_count": "Exclamation marks per article",
    "capital_letter_ratio": "Capital letter ratio",
    "all_caps_word_ratio": "ALL-CAPS word ratio",
    "flesch_kincaid_grade": "Flesch-Kincaid grade level",
    "lexical_diversity": "Lexical diversity (unique/total words)",
}
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, (col, title) in zip(axes.ravel(), BOX.items()):
    groups = [data.loc[data["label"] == l, col].dropna() for l in ("FAKE", "REAL")]
    bp = ax.boxplot(groups, showfliers=False, patch_artist=True)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["FAKE", "REAL"])
    for patch, l in zip(bp["boxes"], ("FAKE", "REAL")):
        patch.set_facecolor(COLORS[l])
        patch.set_alpha(0.6)
    ax.set_title(title, fontsize=10)
save("boxplots_linguistic_features.png")

# 4. Summary table ---------------------------------------------------------
FEATURE_COLS = [
    "word_count", "sentence_count", "article_length", "average_word_length",
    "unique_word_count", "lexical_diversity", "punctuation_count",
    "exclamation_count", "exclamations_per_100_words", "question_count",
    "capital_letter_count", "capital_letter_ratio", "all_caps_word_ratio",
    "flesch_reading_ease", "flesch_kincaid_grade",
]
summary = data.groupby("label")[FEATURE_COLS].agg(["mean", "median"]).round(3)
summary.to_csv(RESULTS_DIR / "eda_summary.csv")

print("========== EDA SUMMARY ==========")
print("\nNews distribution:\n", counts)
print("\nMean of each feature by label:")
print(data.groupby("label")[FEATURE_COLS].mean().round(3).T)
print(f"\nCharts saved in {FIGURES_DIR}")
print("EDA completed successfully!")
