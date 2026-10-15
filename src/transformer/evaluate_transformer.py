"""Score the saved DistilBERT model on the val/test splits (official evaluation).

Run from the project root with the venv active:
    python -m src.transformer.evaluate_transformer
    python -m src.transformer.evaluate_transformer --splits test      # test only, faster

Writes:
    results/transformer/predictions_distilbert.csv   (row_id, split, y_true, y_pred, prob_fake)
    results/transformer/metrics_distilbert.json
"""
import argparse
import json
import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # avoids a Windows OpenMP crash

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import INTERIM_DIR, ROOT, SPLIT_DIR

MODEL_PATH = ROOT / "models" / "transformer" / "distilbert"
OUT_DIR = ROOT / "results" / "transformer"
MAX_LEN = 256


def main(splits, batch_size):
    df = pd.read_csv(INTERIM_DIR / "master.csv", usecols=["row_id", "label", "raw_text"])
    df = df.merge(pd.read_csv(SPLIT_DIR / "split.csv"), on="row_id")
    df = df[df["split"].isin(splits)].reset_index(drop=True)
    print(f"Scoring {len(df)} articles from splits: {splits}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device).eval()

    texts = df["raw_text"].tolist()
    order = np.argsort([len(t) for t in texts])      # similar lengths -> less padding -> faster
    probs = np.zeros(len(texts))

    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            enc = tok([texts[i] for i in idx], truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt").to(device)
            p = torch.softmax(model(**enc).logits.float(), dim=1)[:, 1]
            probs[idx] = p.cpu().numpy()
            done = start + len(idx)
            if (start // batch_size) % 20 == 0:
                print(f"  {done}/{len(order)} articles | {time.time() - t0:.0f}s", flush=True)
    secs = time.time() - t0

    out = pd.DataFrame({
        "row_id": df["row_id"], "split": df["split"], "y_true": df["label"],
        "y_pred": (probs >= 0.5).astype(int), "prob_fake": probs,
    })
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "predictions_distilbert.csv", index=False)

    metrics = {}
    for sp in splits:
        d = out[out["split"] == sp]
        metrics[sp] = {
            "accuracy": round(accuracy_score(d["y_true"], d["y_pred"]), 4),
            "precision_fake": round(precision_score(d["y_true"], d["y_pred"]), 4),
            "recall_fake": round(recall_score(d["y_true"], d["y_pred"]), 4),
            "f1_fake": round(f1_score(d["y_true"], d["y_pred"]), 4),
            "confusion_matrix": confusion_matrix(d["y_true"], d["y_pred"]).tolist(),
        }
    metrics["device"] = device
    metrics["seconds_total"] = round(secs, 1)
    metrics["seconds_per_article"] = round(secs / len(out), 4)

    (OUT_DIR / "metrics_distilbert.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()
    main(a.splits, a.batch_size)