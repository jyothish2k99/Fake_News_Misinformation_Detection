"""Step 7: which words push the baseline toward FAKE / REAL?

Run: python src/feature_importance.py
Input : models/model.pkl, models/tfidf_vectorizer.pkl
Output: results/top_fake_terms.csv, results/top_real_terms.csv,
        results/figures/feature_importance.png
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from settings import FIGURES_DIR, MODEL_PATH, RESULTS_DIR, TFIDF_PATH

TOP_N = 30

model = joblib.load(MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)
terms = np.array(tfidf.get_feature_names_out())

# coef_ > 0 pushes toward classes_[1]; flip sign so that positive = FAKE.
fake_is_positive = list(model.classes_)[1] == "FAKE"
fake_weight = model.coef_[0] if fake_is_positive else -model.coef_[0]

order = np.argsort(fake_weight)
top_fake = pd.DataFrame({"term": terms[order[::-1][:TOP_N]],
                         "weight": fake_weight[order[::-1][:TOP_N]]})
top_real = pd.DataFrame({"term": terms[order[:TOP_N]],
                         "weight": fake_weight[order[:TOP_N]]})
top_fake.to_csv(RESULTS_DIR / "top_fake_terms.csv", index=False)
top_real.to_csv(RESULTS_DIR / "top_real_terms.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 7))
for ax, df, title, color in (
    (axes[0], top_fake.head(20)[::-1], "Words pushing toward FAKE", "#d62728"),
    (axes[1], top_real.head(20)[::-1], "Words pushing toward REAL", "#1f77b4"),
):
    ax.barh(df["term"], df["weight"].abs(), color=color)
    ax.set_title(title)
    ax.set_xlabel("|coefficient|")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
plt.close()

print("Top FAKE terms:", ", ".join(top_fake["term"].head(15)))
print("Top REAL terms:", ", ".join(top_real["term"].head(15)))
print("\nCHECK THIS LIST: if you still see publisher names, datelines or site "
      "footers (e.g. 'reuters', 'featured image'), add them to text_utils.py.")
