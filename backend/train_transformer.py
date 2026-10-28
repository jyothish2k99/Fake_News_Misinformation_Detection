"""Optional offline Hugging Face classifier training; baseline artifacts are untouched."""

from __future__ import annotations

import argparse
import inspect
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from app.services.analytics_service import load_labeled_data
from train_model import near_duplicate_groups, stratified_group_split


BACKEND_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BACKEND_DIR / "artifacts"
DEFAULT_OUTPUT_DIR = ARTIFACT_DIR / "transformer"
SEED = 42


def prepare_records(frame: pd.DataFrame, minimum_chars: int = 30) -> tuple[pd.DataFrame, dict[str, int]]:
    required = {"title", "text", "label"}
    if not required.issubset(frame.columns):
        raise ValueError("Training data must include title, text, and numeric label columns.")

    records = frame.copy()
    for column in ("title", "text"):
        records[column] = records[column].fillna("").astype(str).str.strip()
    records["label"] = pd.to_numeric(records["label"], errors="coerce")
    records = records[records["label"].isin([0, 1])].copy()
    records["model_text"] = (
        records["title"].str.cat(records["text"], sep=" ")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.slice(0, 1800)
    )
    records = records[records["model_text"].str.len() >= minimum_chars].copy()
    records["text_key"] = records["model_text"].str.lower().str.replace(r"\s+", " ", regex=True)

    conflicting_keys = set(
        records.groupby("text_key")["label"].nunique().loc[lambda counts: counts > 1].index
    )
    conflicting_rows = int(records["text_key"].isin(conflicting_keys).sum())
    records = records[~records["text_key"].isin(conflicting_keys)].copy()
    rows_before_deduplication = len(records)
    records = records.drop_duplicates("text_key").copy()
    duplicate_rows = rows_before_deduplication - len(records)
    records["label"] = records["label"].astype(int)
    records = records.drop(columns="text_key").reset_index(drop=True)
    groups, near_duplicate_audit = near_duplicate_groups(records["model_text"].tolist())
    records["near_duplicate_group"] = groups

    counts = records["label"].value_counts()
    if len(counts) != 2 or counts.min() < 2:
        raise ValueError("At least two unique articles from each REAL/FAKE class are required.")
    return records, {
        "invalid_or_short_rows_removed": int(len(frame) - len(records) - duplicate_rows - conflicting_rows),
        "exact_duplicate_rows_removed": duplicate_rows,
        "conflicting_duplicate_rows_excluded": conflicting_rows,
        **near_duplicate_audit,
    }


def split_records(
    records: pd.DataFrame, test_size: float = 0.2, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    groups = records["near_duplicate_group"].to_numpy() if "near_duplicate_group" in records else near_duplicate_groups(records["model_text"].tolist())[0]
    train_indices, test_indices = stratified_group_split(
        records, groups, seed=seed, n_splits=round(1 / test_size)
    )
    return records.iloc[train_indices].reset_index(drop=True), records.iloc[test_indices].reset_index(drop=True)


def validate_local_model_path(model_path: str | Path) -> Path:
    path = Path(model_path).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(
            "Pass a local directory containing a Hugging Face model; remote model downloads are disabled."
        )
    return path.resolve()


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def train(
    model_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    epochs: float = 1.0,
    batch_size: int = 8,
    max_length: int = 256,
    seed: int = SEED,
) -> dict[str, object]:
    local_model_path = validate_local_model_path(model_path)
    records, data_audit = prepare_records(load_labeled_data())
    train_records, test_records = split_records(records, seed=seed)

    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Optional transformer packages are missing. Install backend/requirements-transformer.txt first."
        ) from exc

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tokenizer = AutoTokenizer.from_pretrained(str(local_model_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(local_model_path),
        local_files_only=True,
        num_labels=2,
        id2label={0: "REAL", 1: "FAKE"},
        label2id={"REAL": 0, "FAKE": 1},
        ignore_mismatched_sizes=True,
    )

    class ArticleDataset(torch.utils.data.Dataset):
        def __init__(self, split: pd.DataFrame):
            self.encodings = tokenizer(
                split["model_text"].tolist(),
                truncation=True,
                max_length=max_length,
            )
            self.labels = split["label"].tolist()

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
            return item

    destination = Path(output_dir).resolve()
    trainer_output = destination / "trainer"
    training_options = {
        "output_dir": str(trainer_output),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "logging_strategy": "epoch",
        "save_strategy": "no",
        "report_to": "none",
        "seed": seed,
        "data_seed": seed,
    }
    argument_parameters = inspect.signature(TrainingArguments).parameters
    if "eval_strategy" in argument_parameters:
        training_options["eval_strategy"] = "epoch"
    else:
        training_options["evaluation_strategy"] = "epoch"
    if "use_cpu" in argument_parameters:
        training_options["use_cpu"] = not torch.cuda.is_available()
    else:
        training_options["no_cuda"] = not torch.cuda.is_available()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_options),
        train_dataset=ArticleDataset(train_records),
        eval_dataset=ArticleDataset(test_records),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        tokenizer=tokenizer,
    )
    trainer.train()
    prediction = trainer.predict(ArticleDataset(test_records))
    logits = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
    probabilities = _softmax(np.asarray(logits))[:, 1]
    predicted_labels = (probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        prediction.label_ids,
        predicted_labels,
        average="binary",
        pos_label=1,
        zero_division=0,
    )
    metrics: dict[str, object] = {
        "model": model.config.model_type,
        "local_model_path": str(local_model_path),
        "dataset_rows": int(len(records)),
        "train_rows": int(len(train_records)),
        "test_rows": int(len(test_records)),
        "accuracy": round(float(accuracy_score(prediction.label_ids, predicted_labels)), 4),
        "precision_fake": round(float(precision), 4),
        "recall_fake": round(float(recall), 4),
        "f1_fake": round(float(f1), 4),
        "roc_auc": round(float(roc_auc_score(prediction.label_ids, probabilities)), 4),
        "confusion_matrix": confusion_matrix(prediction.label_ids, predicted_labels, labels=[0, 1]).tolist(),
        "labels": {"0": "REAL", "1": "FAKE"},
        "random_state": seed,
        "data_audit": data_audit,
        "near_duplicate_audit": {key: value for key, value in data_audit.items() if key.startswith("near_duplicate_")},
        "source_time_evaluation": "Not implemented for the transformer workflow; its evaluation split only prevents detected near-duplicate groups from crossing partitions.",
        "note": "Optional transformer evaluation on this dataset only; metrics do not establish real-world claim accuracy.",
    }
    destination.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(destination / "model"))
    tokenizer.save_pretrained(str(destination / "model"))
    metrics_path = ARTIFACT_DIR / "model_metrics_transformer.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an optional Hugging Face classifier from local files only.")
    parser.add_argument("--model-path", required=True, help="Local directory containing a DistilBERT/RoBERTa model.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    metrics = train(
        model_path=args.model_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        seed=args.seed,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()