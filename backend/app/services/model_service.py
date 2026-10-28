from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "baseline_pipeline.joblib"
CORPUS_PATH = ARTIFACT_DIR / "evidence_corpus.csv"


def _is_fake_class(label: object) -> bool:
    if isinstance(label, (int, np.integer)):
        return int(label) == 1
    normalized = str(label).strip().upper().replace("_", "-")
    return normalized in {"FAKE", "FAKE-LIKE", "FAKE NEWS", "FAKENEWS", "MISINFORMATION"}


class ModelService:
    def __init__(self):
        self.pipeline = None
        self.metadata = {}
        self.corpus = None
        self.load()

    @property
    def ready(self) -> bool:
        return self.pipeline is not None

    def load(self):
        if MODEL_PATH.exists():
            bundle = joblib.load(MODEL_PATH)
            if isinstance(bundle, dict):
                self.pipeline = bundle.get("pipeline")
                self.metadata = bundle.get("metadata", {})
            else:
                self.pipeline = bundle
        if CORPUS_PATH.exists():
            import pandas as pd
            self.corpus = pd.read_csv(CORPUS_PATH).fillna("")

    def predict(self, title: str, text: str):
        if not self.ready:
            raise RuntimeError("Model is not trained. Run python train_model.py after placing Fake.csv and True.csv in backend/data/.")
        combined = (title.strip() + " " + text.strip()).strip()
        probabilities = self.pipeline.predict_proba([combined])[0]
        classes = list(self.pipeline.named_steps["classifier"].classes_)
        fake_indices = [index for index, value in enumerate(classes) if _is_fake_class(value)]
        if len(fake_indices) != 1 or len(classes) != 2:
            raise RuntimeError("The loaded model does not expose a supported REAL/FAKE class mapping.")
        fake_idx = fake_indices[0]
        real_idx = 1 - fake_idx
        fake_probability = float(probabilities[fake_idx])
        real_probability = float(probabilities[real_idx])
        label = "FAKE-LIKE" if fake_probability >= 0.5 else "REAL-LIKE"
        confidence = max(fake_probability, 1.0 - fake_probability)
        terms = self.explain(combined, limit=12)
        return {
            "label": label,
            "fake_probability": round(fake_probability, 4),
            "real_probability": round(real_probability, 4),
            "confidence": round(confidence, 4),
            "explanation": terms,
            "explanation_available": bool(terms),
            "explanation_message": "" if terms else "Model feature contributions are unavailable for this prediction.",
            "explanation_method": "TF-IDF feature value × Logistic Regression coefficient",
            "disclaimer": "This is a model estimate based on patterns in the training corpus, not a fact-check or proof of truthfulness."
        }

    def explain(self, text: str, limit: int = 12):
        if not self.ready:
            return []
        try:
            vec = self.pipeline.named_steps["tfidf"]
            clf = self.pipeline.named_steps["classifier"]
            classes = list(clf.classes_)
            fake_indices = [index for index, value in enumerate(classes) if _is_fake_class(value)]
            if len(fake_indices) != 1 or len(classes) != 2:
                return []
            x = vec.transform([text])
            names = vec.get_feature_names_out()
            coefs = clf.coef_[0]
            indices = x.indices
            values = x.data
            fake_is_positive_class = fake_indices[0] == 1
            contributions = [
                (str(names[i]), float(v * coefs[i]) * (1 if fake_is_positive_class else -1))
                for i, v in zip(indices, values)
            ]
            contributions.sort(key=lambda item: abs(item[1]), reverse=True)
            return [{"term": term, "contribution": round(score, 5), "direction": "FAKE" if score > 0 else "REAL"} for term, score in contributions[:limit]]
        except Exception:
            return []
