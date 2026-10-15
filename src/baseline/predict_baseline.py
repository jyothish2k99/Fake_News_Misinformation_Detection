import joblib

from src.config import MODEL_DIR


class BaselinePredictor:
    def __init__(self, version="v1_cleaned", preprocess_fn=None):
        self.vectorizer = joblib.load(MODEL_DIR / version / "tfidf_vectorizer.joblib")
        self.model = joblib.load(MODEL_DIR / version / "logreg_model.joblib")
        self.preprocess_fn = preprocess_fn  # AI/ML-01's clean_text, used for v1

    def predict(self, text: str) -> dict:
        if not isinstance(text, str) or not text.strip():
            return {"label": "Unknown", "confidence": None,
                    "prob_fake": None, "error": "Empty input"}
        if self.preprocess_fn:
            text = self.preprocess_fn(text)
        X = self.vectorizer.transform([text])
        prob_fake = float(self.model.predict_proba(X)[0, 1])
        label = "Fake" if prob_fake >= 0.5 else "Real"
        return {"label": label,
                "confidence": max(prob_fake, 1 - prob_fake),
                "prob_fake": prob_fake}

                