"""Side-by-side comparison of the baseline, DistilBERT and a one-word rule on the TEST split.

Run from the project root with the venv active:
    python -m src.comparison.compare_models

Reads:  results/baseline/predictions_v0_raw.csv
        results/transformer/predictions_distilbert.csv
        data/interim/master.csv, data/splits/split.csv
Writes: results/comparison/comparison_table.csv
"""
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from src.config import INTERIM_DIR, ROOT, SPLIT_DIR

OUT_DIR = ROOT / "results" / "comparison"


def score(name, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_fake": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall_fake": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_fake": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp, "errors": int(fp + fn),
    }


def load_test(path):
    p = pd.read_csv(path)
    return p[p["split"] == "test"].set_index("row_id").sort_index()


def main():
    base = load_test(ROOT / "results" / "baseline" / "predictions_v0_raw.csv")
    dist = load_test(ROOT / "results" / "transformer" / "predictions_distilbert.csv")
    assert base.index.equals(dist.index), "test row_ids differ between the two files!"
    assert (base["y_true"] == dist["y_true"]).all(), "labels differ between the two files!"

    master = pd.read_csv(INTERIM_DIR / "master.csv", usecols=["row_id", "raw_text"])
    text = master.set_index("row_id").loc[base.index, "raw_text"]
    rule_pred = (~text.str.contains("(Reuters)", regex=False)).astype(int)  # Reuters -> Real(0)

    y = base["y_true"]
    table = pd.DataFrame([
        score("Rule: '(Reuters)' = Real, else Fake", y, rule_pred),
        score("TF-IDF + Logistic Regression", y, base["y_pred"]),
        score("DistilBERT", y, dist["y_pred"]),
    ])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "comparison_table.csv", index=False)
    print(f"Test set: {len(y)} articles\n")
    print(table.to_string(index=False))

    b_err = base["y_pred"] != y
    d_err = dist["y_pred"] != y
    r_err = rule_pred != y
    print(f"\nBaseline errors: {int(b_err.sum())} | DistilBERT errors: {int(d_err.sum())}"
          f" | Rule errors: {int(r_err.sum())}")
    print(f"Baseline errors that DistilBERT gets right: {int((b_err & ~d_err).sum())}")
    print(f"Baseline errors that the one-word rule gets right: {int((b_err & ~r_err).sum())}")
    print(f"Errors made by DistilBERT but not by the baseline: {int((d_err & ~b_err).sum())}")


if __name__ == "__main__":
    main()