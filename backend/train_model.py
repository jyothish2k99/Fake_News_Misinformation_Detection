from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"
SEED = 42
NEAR_DUPLICATE_SIMILARITY = 0.9


def load_one(name: str, label: int) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Copy Fake.csv and True.csv into backend/data/.")
    df = pd.read_csv(path, on_bad_lines="skip")
    if not {"title", "text"}.issubset(df.columns):
        df = pd.read_csv(path, header=None, on_bad_lines="skip")
        if len(df.columns) >= 2:
            df = df.iloc[:, :4]
            df.columns = ["title", "text", "subject", "date"][:len(df.columns)]
    source_column = next((column for column in ("source", "publisher", "domain") if column in df), None)
    df["source"] = df[source_column] if source_column else ""
    for column in ("title", "text", "source"):
        if column not in df:
            df[column] = ""
        df[column] = df[column].fillna("").astype(str).str.strip()
    if "date" not in df:
        df["date"] = ""
    df["date"] = df["date"].fillna("").astype(str).str.strip()
    df = df[(df.title + " " + df.text).str.len() >= 30].copy()
    df["label"] = label
    return df


def clean_records(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    df = frame.copy()
    df["text_key"] = (df.title + " " + df.text).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df.text_key.str.len() > 0].copy()
    conflict_keys = set(df.groupby("text_key").label.nunique().loc[lambda values: values > 1].index)
    conflicting_rows = int(df.text_key.isin(conflict_keys).sum())
    df = df[~df.text_key.isin(conflict_keys)].copy()
    before_deduplication = len(df)
    df = df.drop_duplicates("text_key").copy()
    exact_duplicate_rows = before_deduplication - len(df)
    df["model_text"] = (df.title + " " + df.text).str.replace(r"\s+", " ", regex=True).str.strip().str.slice(0, 1800)
    df = df.reset_index(drop=True)
    return df, {
        "conflicting_duplicate_rows_excluded": conflicting_rows,
        "exact_duplicate_rows_removed": exact_duplicate_rows,
    }


def near_duplicate_groups(
    texts: list[str], similarity_threshold: float = NEAR_DUPLICATE_SIMILARITY
) -> tuple[np.ndarray, dict[str, int]]:
    """Group char-TF-IDF documents connected by similarity-threshold edges."""
    if not 0 < similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be in (0, 1].")
    if len(texts) < 2:
        return np.arange(len(texts)), {"near_duplicate_group_count": 0, "near_duplicate_rows": 0, "near_duplicate_pairs": 0}

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=100_000, dtype=np.float32)
    matrix = vectorizer.fit_transform(texts)
    neighbors = NearestNeighbors(
        radius=1.0 - similarity_threshold,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    neighbors.fit(matrix)
    graph = neighbors.radius_neighbors_graph(matrix, mode="distance").tocoo()
    parents = list(range(len(texts)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    pairs = set()
    for left, right, distance in zip(graph.row, graph.col, graph.data):
        if left == right or 1.0 - float(distance) < similarity_threshold:
            continue
        pair = (min(int(left), int(right)), max(int(left), int(right)))
        pairs.add(pair)
        left_root, right_root = find(pair[0]), find(pair[1])
        if left_root != right_root:
            parents[right_root] = left_root

    roots = [find(index) for index in range(len(texts))]
    root_to_group = {root: group for group, root in enumerate(sorted(set(roots)))}
    groups = np.asarray([root_to_group[root] for root in roots], dtype=np.int64)
    group_sizes = pd.Series(groups).value_counts()
    duplicate_groups = group_sizes[group_sizes > 1]
    return groups, {
        "near_duplicate_group_count": int(len(duplicate_groups)),
        "near_duplicate_rows": int(duplicate_groups.sum()),
        "near_duplicate_pairs": len(pairs),
    }


def stratified_group_split(
    frame: pd.DataFrame, groups: np.ndarray, seed: int = SEED, n_splits: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    group_count = len(np.unique(groups))
    if group_count < 2:
        raise ValueError("At least two independent duplicate groups are required for evaluation.")
    splitter = StratifiedGroupKFold(n_splits=min(max(2, n_splits), group_count), shuffle=True, random_state=seed)
    for train_indices, test_indices in splitter.split(frame.model_text, frame.label, groups):
        if frame.iloc[train_indices].label.nunique() == 2 and frame.iloc[test_indices].label.nunique() == 2:
            return train_indices, test_indices
    raise ValueError("Could not create a group-disjoint train/test split containing both classes.")


def combine_group_columns(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    parents = list(range(len(frame)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for column in columns:
        first_seen = {}
        for index, value in enumerate(frame[column].tolist()):
            if pd.isna(value):
                continue
            previous = first_seen.setdefault(str(value), index)
            left_root, right_root = find(index), find(previous)
            if left_root != right_root:
                parents[right_root] = left_root
    roots = [find(index) for index in range(len(frame))]
    root_to_group = {root: group for group, root in enumerate(sorted(set(roots)))}
    return np.asarray([root_to_group[root] for root in roots], dtype=np.int64)


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=20000,
                                  min_df=2, sublinear_tf=True, dtype=np.float32)),
        ("classifier", LogisticRegression(max_iter=100, solver="liblinear", tol=5e-3,
                                           class_weight="balanced", random_state=SEED)),
    ])


def evaluate_split(train: pd.DataFrame, test: pd.DataFrame, method: str) -> dict[str, object]:
    train_sources = train.source.fillna("").astype(str).str.strip() if "source" in train else pd.Series(dtype=str)
    test_sources = test.source.fillna("").astype(str).str.strip() if "source" in test else pd.Series(dtype=str)
    metrics: dict[str, object] = {
        "available": False,
        "method": method,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_sources": int(train_sources[train_sources.ne("")].nunique()),
        "test_sources": int(test_sources[test_sources.ne("")].nunique()),
    }
    if train.empty or test.empty or train.label.nunique() < 2 or test.label.nunique() < 2:
        metrics["reason"] = "Both partitions need REAL and FAKE examples; this dataset split does not provide them."
        return metrics
    pipeline = build_pipeline()
    pipeline.fit(train.model_text, train.label)
    predicted = pipeline.predict(test.model_text)
    probabilities = pipeline.predict_proba(test.model_text)[:, list(pipeline.classes_).index(1)]
    precision, recall, f1, _ = precision_recall_fscore_support(
        test.label, predicted, average="binary", pos_label=1, zero_division=0
    )
    metrics.update({
        "available": True,
        "accuracy": round(float(accuracy_score(test.label, predicted)), 4),
        "precision_fake": round(float(precision), 4),
        "recall_fake": round(float(recall), 4),
        "f1_fake": round(float(f1), 4),
        "roc_auc": round(float(roc_auc_score(test.label, probabilities)), 4),
        "confusion_matrix": confusion_matrix(test.label, predicted, labels=[0, 1]).tolist(),
    })
    return metrics


def source_evaluation(frame: pd.DataFrame) -> dict[str, object]:
    normalized = frame.source.fillna("").astype(str).str.strip().str.lower()
    usable = frame[normalized.ne("")].copy()
    usable["source"] = normalized[normalized.ne("")]
    distinct_sources = int(usable.source.nunique())
    if distinct_sources < 2:
        return {"available": False, "method": "Stratified group split by source", "eligible_rows": int(len(usable)),
                "source_groups": distinct_sources, "missing_source_rows": int(len(frame) - len(usable)),
                "reason": "Source/publisher identifiers are missing or fewer than two distinct values exist."}
    try:
        group_columns = ["source"]
        if "near_duplicate_group" in usable:
            group_columns.append("near_duplicate_group")
        groups = combine_group_columns(usable, group_columns)
        train_indices, test_indices = stratified_group_split(usable, groups)
    except ValueError as exc:
        return {"available": False, "method": "Stratified group split by source", "eligible_rows": int(len(usable)),
            "source_groups": distinct_sources, "missing_source_rows": int(len(frame) - len(usable)),
            "reason": str(exc)}
    result = evaluate_split(usable.iloc[train_indices], usable.iloc[test_indices], "Held-out source groups")
    result.update({"eligible_rows": int(len(usable)), "source_groups": distinct_sources,
                   "missing_source_rows": int(len(frame) - len(usable))})
    return result


def time_evaluation(frame: pd.DataFrame) -> dict[str, object]:
    dates = pd.to_datetime(frame.date, errors="coerce", utc=True)
    valid = frame[dates.notna()].copy()
    valid["parsed_date"] = dates[dates.notna()]
    valid_dates = np.sort(valid.parsed_date.unique())
    missing_rows = int(len(frame) - len(valid))
    if len(valid_dates) < 2:
        return {"available": False, "method": "Chronological holdout using the final 20% of unique dates",
                "eligible_rows": int(len(valid)), "missing_or_invalid_date_rows": missing_rows,
                "reason": "Fewer than two distinct valid publication dates are available."}
    cutoff_index = min(len(valid_dates) - 1, max(1, int(len(valid_dates) * 0.8)))
    cutoff = pd.Timestamp(valid_dates[cutoff_index])
    train = valid[valid.parsed_date < cutoff]
    test = valid[valid.parsed_date >= cutoff]
    duplicate_rows_excluded = 0
    if "near_duplicate_group" in valid:
        test_groups = set(test.near_duplicate_group)
        keep_train = ~train.near_duplicate_group.isin(test_groups)
        duplicate_rows_excluded = int((~keep_train).sum())
        train = train[keep_train]
    result = evaluate_split(train, test, "Train before cutoff; test on/after cutoff")
    result.update({"eligible_rows": int(len(valid)), "missing_or_invalid_date_rows": missing_rows,
                   "cutoff_date_utc": cutoff.isoformat(),
                   "near_duplicate_train_rows_excluded": duplicate_rows_excluded})
    return result


def main():
    start = time.time()
    fake = load_one("Fake.csv", 1)
    real = load_one("True.csv", 0)
    df, exact_audit = clean_records(pd.concat([fake, real], ignore_index=True))
    if len(df) > 20000:
        df = df.groupby("label", group_keys=False).sample(n=10000, random_state=SEED).reset_index(drop=True)

    df["model_text"] = (df.title + " " + df.text).str.replace(r"\s+", " ", regex=True).str.strip().str.slice(0, 1800)
    groups, near_audit = near_duplicate_groups(df.model_text.tolist())
    df["near_duplicate_group"] = groups
    train_indices, test_indices = stratified_group_split(df, groups)
    train = df.iloc[train_indices]
    test = df.iloc[test_indices]

    pipeline = build_pipeline()
    print(f"Training baseline on {len(train):,} rows; testing on {len(test):,} group-disjoint rows")
    pipeline.fit(train.model_text, train.label)
    predicted = pipeline.predict(test.model_text)
    probabilities = pipeline.predict_proba(test.model_text)[:, list(pipeline.classes_).index(1)]
    precision, recall, f1, _ = precision_recall_fscore_support(test.label, predicted, average="binary", pos_label=1, zero_division=0)
    metadata = {
        "model": "TF-IDF + Logistic Regression",
        "dataset_rows": int(len(df)), "train_rows": int(len(train)), "test_rows": int(len(test)),
        "accuracy": round(float(accuracy_score(test.label, predicted)), 4),
        "precision_fake": round(float(precision), 4), "recall_fake": round(float(recall), 4),
        "f1_fake": round(float(f1), 4), "roc_auc": round(float(roc_auc_score(test.label, probabilities)), 4),
        "confusion_matrix": confusion_matrix(test.label, predicted, labels=[0, 1]).tolist(),
        "labels": {"0": "REAL", "1": "FAKE"}, "random_state": SEED,
        "evaluation_method": "Stratified group split; all detected near-duplicate edges at or above the similarity threshold are kept within one partition.",
        "similarity_threshold": NEAR_DUPLICATE_SIMILARITY,
        "data_audit": {**exact_audit, **near_audit},
        "source_based_evaluation": source_evaluation(df),
        "time_based_evaluation": time_evaluation(df),
        "note": "Group separation reduces detected near-duplicate leakage; it does not rule out topic, source, temporal, or dataset-collection leakage.",
    }
    ART.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, ART / "baseline_pipeline.joblib", compress=3)
    corpus = df[[column for column in ("title", "text", "subject", "date", "source", "label") if column in df]].copy()
    corpus.to_csv(ART / "evidence_corpus.csv", index=False)
    (ART / "model_metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Saved artifacts. Elapsed: {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
