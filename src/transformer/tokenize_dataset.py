"""Tokenize the agreed base dataset for DistilBERT, in the agreed format.

Reads data/interim/master.csv (all 38,656 articles) and tokenizes `raw_text` (natural text,
no stopword removal or lemmatization). Labels keep the team convention: Fake = 1, Real = 0.
Every array is aligned with row_id, so the file can be joined to data/splits/split.csv.

Run from the project root:
    python -m src.transformer.tokenize_dataset

Output (gitignored): data/processed/tokenized_raw_text.npz with arrays
    row_id, split, input_ids, attention_mask, labels
Note: train_transformer.py tokenizes by itself and does not need this file.
"""
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from src.config import INTERIM_DIR, PROCESSED_DIR, SPLIT_DIR

MODEL = "distilbert-base-uncased"
MAX_LEN = 256
BATCH = 512


def main():
    df = pd.read_csv(INTERIM_DIR / "master.csv", usecols=["row_id", "label", "raw_text"])
    df = df.merge(pd.read_csv(SPLIT_DIR / "split.csv"), on="row_id", how="left")
    assert df["split"].notna().all(), "some row_ids are missing from split.csv"
    df["raw_text"] = df["raw_text"].fillna("").astype(str)   # no rows are dropped

    tok = AutoTokenizer.from_pretrained(MODEL)
    n = len(df)
    input_ids = np.zeros((n, MAX_LEN), dtype=np.int32)
    attention = np.zeros((n, MAX_LEN), dtype=np.int32)
    texts = df["raw_text"].tolist()

    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        enc = tok(texts[start:end], truncation=True, padding="max_length", max_length=MAX_LEN)
        input_ids[start:end] = np.asarray(enc["input_ids"], dtype=np.int32)
        attention[start:end] = np.asarray(enc["attention_mask"], dtype=np.int32)
        print(f"Tokenized {end}/{n}", flush=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "tokenized_raw_text.npz"
    np.savez_compressed(
        out,
        row_id=df["row_id"].to_numpy(dtype=np.int32),
        split=df["split"].to_numpy(dtype=str),
        input_ids=input_ids,
        attention_mask=attention,
        labels=df["label"].to_numpy(dtype=np.int32),
    )
    print(f"\nSaved {out}")
    print("articles:", n, "| input_ids:", input_ids.shape, "| label mapping: Fake = 1, Real = 0")
    print(df["split"].value_counts().to_dict())


if __name__ == "__main__":
    main()