"""Step 6: metrics for the baseline (FAKE is the positive class).

Run: python src/evaluation.py
Input : results/evaluation_results.csv
Output: results/metrics.json, results/classification_report.txt,
        results/figures/confusion_matrix.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

from settings import FIGURES_DIR, RESULTS_DIR

data = pd.read_csv(RESULTS_DIR / "evaluation_results.csv")
metrics, report_lines = {}, []

for split in ("val", "test"):
    part = data[data["split"] == split]
    y_true, y_pred = part["actual"], part["predicted"]
    metrics[split] = {
        "n": int(len(part)),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_fake": round(precision_score(y_true, y_pred, pos_label="FAKE"), 4),
        "recall_fake": round(recall_score(y_true, y_pred, pos_label="FAKE"), 4),
        "f1_fake": round(f1_score(y_true, y_pred, pos_label="FAKE"), 4),
    }
    cm = confusion_matrix(y_true, y_pred, labels=["FAKE", "REAL"])
    report_lines += [
        f"===== {split.upper()} SET =====",
        json.dumps(metrics[split], indent=2),
        "Confusion matrix (rows = actual, cols = predicted; order FAKE, REAL):",
        str(cm),
        classification_report(y_true, y_pred),
        "",
    ]
    if split == "test":
        ConfusionMatrixDisplay(cm, display_labels=["FAKE", "REAL"]).plot(cmap="Blues")
        plt.title("Baseline (TF-IDF + LR) - Test set")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
        plt.close()

(RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
(RESULTS_DIR / "classification_report.txt").write_text("\n".join(report_lines))

print("========== MODEL EVALUATION ==========")
print("\n".join(report_lines))
print("Saved: results/metrics.json, classification_report.txt, figures/confusion_matrix.png")
