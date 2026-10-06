import argparse
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.config import (INTERIM_DIR, PROCESSED_DIR, SPLIT_DIR,
                        MODEL_DIR, RESULTS_DIR, RANDOM_STATE)


def load_data(version):
    splits = pd.read_csv(SPLIT_DIR / "split.csv")
    if version == "v0_raw":
        df = pd.read_csv(INTERIM_DIR / "master.csv", usecols=["row_id", "label", "raw_text"])
        text_col = "raw_text"
    else:  # v1_cleaned
        df = pd.read_csv(PROCESSED_DIR / "cleaned_dataset.csv")
        text_col = "cleaned_text"
        df[text_col] = df[text_col].fillna("")
    df = df.merge(splits, on="row_id", how="inner")
    assert len(df) == len(splits), "row_id mismatch with split.csv!"
    return df, text_col


def main(version):
    df, text_col = load_data(version)
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]
    print(f"Version: {version} | train {len(train)}, val {len(val)}, test {len(test)}")

    # 1) TF-IDF: fit ONLY on training data
    vectorizer = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),   # single words and word pairs
        min_df=5,             # ignore very rare terms
        max_df=0.9,           # ignore terms in more than 90% of articles
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train[text_col])
    X_val = vectorizer.transform(val[text_col])
    X_test = vectorizer.transform(test[text_col])

    # 2) Logistic Regression
    model = LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
    model.fit(X_train, train["label"])

    # 3) Quick sanity metrics (official evaluation is AI/ML-03's job)
    metrics = {}
    for name, X, part in [("val", X_val, val), ("test", X_test, test)]:
        pred = model.predict(X)
        metrics[name] = {"accuracy": round(accuracy_score(part["label"], pred), 4),
                         "f1_fake": round(f1_score(part["label"], pred), 4)}
    print(json.dumps(metrics, indent=2))

    # 4) Save model files and metrics
    out_model = MODEL_DIR / version
    out_model.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, out_model / "tfidf_vectorizer.joblib")
    joblib.dump(model, out_model / "logreg_model.joblib")
    (RESULTS_DIR / f"metrics_{version}.json").write_text(json.dumps(metrics, indent=2))

    # 5) Predictions for AI/ML-03 (val + test)
    rows = []
    for part, X in [(val, X_val), (test, X_test)]:
        rows.append(pd.DataFrame({
            "row_id": part["row_id"].values,
            "split": part["split"].values,
            "y_true": part["label"].values,
            "y_pred": model.predict(X),
            "prob_fake": model.predict_proba(X)[:, 1],
        }))
    pd.concat(rows).to_csv(RESULTS_DIR / f"predictions_{version}.csv", index=False)

    # 6) Most influential words
    names = vectorizer.get_feature_names_out()
    top = pd.DataFrame({"term": names, "weight": model.coef_[0]}).sort_values("weight")
    top.head(25).assign(pushes_toward="REAL").to_csv(
        RESULTS_DIR / f"top_real_terms_{version}.csv", index=False)
    top.tail(25).iloc[::-1].assign(pushes_toward="FAKE").to_csv(
        RESULTS_DIR / f"top_fake_terms_{version}.csv", index=False)
    print("Saved model, predictions, metrics and top terms.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--version", choices=["v0_raw", "v1_cleaned"], required=True)
    main(p.parse_args().version)