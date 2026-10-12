"""Step 5: baseline model = TF-IDF + Logistic Regression.

Run: python src/model_training.py
Input : data/processed_news.csv
Output: models/model.pkl, models/tfidf_vectorizer.pkl,
        results/split_ids.csv  (reuse the SAME split for DistilBERT in week 2),
        results/evaluation_results.csv
"""
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from settings import (MODEL_PATH, PROCESSED_CSV, RANDOM_STATE, RESULTS_DIR,
                      TFIDF_PATH)

data = pd.read_csv(PROCESSED_CSV, usecols=["article_id", "cleaned_text", "label"])
data = data.dropna(subset=["cleaned_text"]).reset_index(drop=True)

# ---------------------------------------------------------- 80 / 10 / 10 split
train_df, temp_df = train_test_split(
    data, test_size=0.20, random_state=RANDOM_STATE, stratify=data["label"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=RANDOM_STATE, stratify=temp_df["label"]
)

split_ids = pd.concat([
    train_df[["article_id"]].assign(split="train"),
    val_df[["article_id"]].assign(split="val"),
    test_df[["article_id"]].assign(split="test"),
])
split_ids.to_csv(RESULTS_DIR / "split_ids.csv", index=False)

# --------------------------------------------------------------------- TF-IDF
# Fit ONLY on training data (fitting on everything would leak test vocabulary).
tfidf = TfidfVectorizer(
    ngram_range=(1, 2), max_features=50000, min_df=5, sublinear_tf=True
)
X_train = tfidf.fit_transform(train_df["cleaned_text"])
X_val = tfidf.transform(val_df["cleaned_text"])
X_test = tfidf.transform(test_df["cleaned_text"])

# ----------------------------------------------------------------- model
model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
model.fit(X_train, train_df["label"])

fake_idx = list(model.classes_).index("FAKE")


def predictions(df, X, name):
    out = pd.DataFrame({
        "article_id": df["article_id"].values,
        "split": name,
        "actual": df["label"].values,
        "predicted": model.predict(X),
        "prob_fake": model.predict_proba(X)[:, fake_idx],
    })
    return out


results = pd.concat([predictions(val_df, X_val, "val"),
                     predictions(test_df, X_test, "test")])
results.to_csv(RESULTS_DIR / "evaluation_results.csv", index=False)

joblib.dump(model, MODEL_PATH)
joblib.dump(tfidf, TFIDF_PATH)

print("Training samples  :", X_train.shape[0])
print("Validation samples:", X_val.shape[0])
print("Testing samples   :", X_test.shape[0])
print("TF-IDF shape      :", X_train.shape)
print("\nModel, vectorizer, split ids and predictions saved.")
print("Now run: python src/evaluation.py")
