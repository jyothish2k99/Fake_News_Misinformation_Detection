from __future__ import annotations
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .model_service import CORPUS_PATH, MODEL_PATH
import joblib

class EvidenceService:
    """Find similar corpus entries; similarity is not a verification verdict."""
    def __init__(self):
        self.df = pd.DataFrame()
        self.vectorizer = None
        self.matrix = None
        self._load()

    def _load(self):
        if not CORPUS_PATH.exists() or not MODEL_PATH.exists():
            return
        self.df = pd.read_csv(CORPUS_PATH).fillna("")
        if not len(self.df):
            return
        bundle = joblib.load(MODEL_PATH)
        pipeline = bundle.get("pipeline") if isinstance(bundle, dict) else bundle
        self.vectorizer = pipeline.named_steps["tfidf"]
        corpus = (self.df.get("title", "").astype(str) + " " + self.df.get("text", "").astype(str)).tolist()
        self.matrix = self.vectorizer.transform(corpus)

    def search(self, query: str, k: int = 3):
        if self.vectorizer is None or self.matrix is None or not query.strip() or k <= 0:
            return []
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        idx = scores.argsort()[::-1]
        results = []
        for i in idx:
            if scores[i] <= 0:
                break
            row = self.df.iloc[int(i)]
            title = str(row.get("title", "Untitled"))
            article_text = str(row.get("text", ""))
            results.append({
                "title": title[:240], "snippet": article_text[:360],
                "dataset_label": "FAKE" if str(row.get("label", "0")) in ("1", "FAKE") else "REAL",
                "similarity": round(float(scores[i]), 4),
                "evidence_status": "RELATED_DATASET_ARTICLE_NOT_VERIFIED",
                "source": "Local training corpus",
                "date": str(row.get("date", "")).strip() or None,
            })
            if len(results) >= k:
                break
        return results
