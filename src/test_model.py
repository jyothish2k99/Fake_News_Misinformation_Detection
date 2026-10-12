"""Step 8: try the baseline on any text, with a confidence percentage.

Run: python src/test_model.py
(predict_news() is also what the Streamlit app will import in week 3.)
"""
import joblib

from settings import MODEL_PATH, TFIDF_PATH
from text_utils import preprocess_text

_model = _tfidf = None


def predict_news(text):
    """Return (label, confidence_percent, prob_fake_percent)."""
    global _model, _tfidf
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        _tfidf = joblib.load(TFIDF_PATH)
    cleaned = preprocess_text(text)          # SAME cleaning as training
    probs = _model.predict_proba(_tfidf.transform([cleaned]))[0]
    prob = dict(zip(_model.classes_, probs))
    label = max(prob, key=prob.get)
    return label, round(100 * prob[label], 1), round(100 * prob["FAKE"], 1)


if __name__ == "__main__":
    news = input("\nEnter news text: ")
    if not news.strip():
        raise SystemExit("No text entered.")
    label, confidence, prob_fake = predict_news(news)
    print(f"\nPrediction: {label} ({confidence}% confidence)")
    print(f"Probability of being FAKE: {prob_fake}%")
