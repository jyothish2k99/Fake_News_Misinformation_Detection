"""Fine-tune DistilBERT for Fake (1) / Real (0) classification.

Reproduces the Kaggle run (lr 2e-5, 256 tokens, 2 epochs, batch 16 per GPU, seed 42).
Only the train and validation splits are used. The test split is never loaded.

On the project (laptop):
    python -m src.transformer.train_transformer --max-rows 200 --epochs 1     # quick smoke run
On Kaggle (master.csv and split.csv in one folder):
    python train_transformer.py --data-dir /kaggle/input/datasets/<user>/<dataset> \
        --out-dir /kaggle/working/distilbert_v1

Other settings can be changed with --lr, --max-len, --epochs, --batch-size, --seed
(the tuning grid used --lr 3e-5 / 5e-5 and --max-len 128).
"""
import argparse
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # avoids a Windows OpenMP crash

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

try:  # inside the project
    from src.config import INTERIM_DIR, ROOT, SPLIT_DIR
except ImportError:  # standalone, e.g. on Kaggle
    ROOT = INTERIM_DIR = SPLIT_DIR = None

MODEL = "distilbert-base-uncased"


def metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "precision": precision_score(p.label_ids, preds),
        "recall": recall_score(p.label_ids, preds),
        "f1": f1_score(p.label_ids, preds),
    }


def main(a):
    set_seed(a.seed)

    if a.data_dir:
        master_path, split_path = Path(a.data_dir) / "master.csv", Path(a.data_dir) / "split.csv"
    else:
        master_path, split_path = INTERIM_DIR / "master.csv", SPLIT_DIR / "split.csv"
    out_dir = Path(a.out_dir) if a.out_dir else ROOT / "models" / "transformer" / "distilbert_rerun"

    df = pd.read_csv(master_path, usecols=["row_id", "label", "raw_text"])
    df = df.merge(pd.read_csv(split_path), on="row_id")
    train, val = df[df["split"] == "train"], df[df["split"] == "val"]
    if a.max_rows:                                     # quick smoke run
        train = train.sample(min(a.max_rows, len(train)), random_state=a.seed)
        val = val.sample(min(max(a.max_rows // 2, 1), len(val)), random_state=a.seed)
    print(f"train: {len(train)} | val: {len(val)} | device: "
          f"{'cuda x' + str(torch.cuda.device_count()) if torch.cuda.is_available() else 'cpu'}")

    tok = AutoTokenizer.from_pretrained(MODEL)

    def to_ds(d):
        ds = Dataset.from_dict({"text": d["raw_text"].tolist(), "label": d["label"].tolist()})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=a.max_len),
                      batched=True, remove_columns=["text"])

    train_ds, val_ds = to_ds(train), to_ds(val)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=64,
        learning_rate=a.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        seed=a.seed,
        report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
    trainer.train()
    print(trainer.evaluate())

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trainer.state.log_history).to_csv(out_dir / "training_log.csv", index=False)
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    print("Saved model, tokenizer and training_log.csv to", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", help="folder holding master.csv and split.csv (default: project paths)")
    ap.add_argument("--out-dir", help="where to save the model (default: models/transformer/distilbert_rerun)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16, help="per device")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-rows", type=int, default=0, help="use only N training rows (smoke test)")
    main(ap.parse_args())