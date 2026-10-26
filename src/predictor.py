"""One interface for every model, for Streamlit and the other team members.

    from src.predictor import load_model, predict, predict_batch, available_models

    predict("some article text")                      # default model: distilbert
    predict("some article text", model="baseline_v0")
    predict_batch(["text 1", "text 2"], model="distilbert")

Every result is a dict: {"label": "Fake"/"Real"/"Unknown", "confidence": float or None,
"prob_fake": float or None, "model": name}. Invalid input adds "error". Very short input adds
"warning". In Streamlit, wrap load_model with st.cache_resource so the model loads once.
"""

MODELS = {
    "distilbert": "DistilBERT fine-tuned on raw_text (models/transformer/distilbert)",
    "baseline_v0": "TF-IDF + Logistic Regression on raw_text (models/baseline/v0_raw)",
    "baseline_v1": "TF-IDF + Logistic Regression on AI/ML-01 cleaned text (needs preprocess_fn)",
}
DEFAULT_MODEL = "distilbert"
SHORT_TEXT_WORDS = 30   # arbitrary cut-off: the models were trained on full articles
_cache = {}


def available_models():
    return dict(MODELS)


def load_model(name=DEFAULT_MODEL, preprocess_fn=None):
    """Load a model once and keep it in memory. Raises a clear error if it cannot be loaded."""
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODELS)}")
    if name in _cache:
        return _cache[name]

    if name == "distilbert":
        from src.transformer.predict_transformer import TransformerPredictor
        model = TransformerPredictor()
    elif name == "baseline_v0":
        from src.baseline.predict_baseline import BaselinePredictor
        model = BaselinePredictor(version="v0_raw")
    else:  # baseline_v1
        if preprocess_fn is None:
            raise ValueError("baseline_v1 needs AI/ML-01's clean_text function: "
                             "load_model('baseline_v1', preprocess_fn=clean_text)")
        from src.baseline.predict_baseline import BaselinePredictor
        model = BaselinePredictor(version="v1_cleaned", preprocess_fn=preprocess_fn)

    _cache[name] = model
    return model


def _finish(result, name, text):
    result = dict(result)
    result["model"] = name
    if (isinstance(text, str) and "error" not in result
            and len(text.split()) < SHORT_TEXT_WORDS):
        result["warning"] = (f"Very short text (under {SHORT_TEXT_WORDS} words). The models were "
                             "trained on full news articles, so this result is unreliable.")
    return result


def _error(name, msg):
    return {"label": "Unknown", "confidence": None, "prob_fake": None,
            "model": name, "error": msg}


def predict(text, model=DEFAULT_MODEL):
    return predict_batch([text], model=model)[0]


def predict_batch(texts, model=DEFAULT_MODEL):
    try:
        m = load_model(model)
    except Exception as e:  # noqa: BLE001  (missing files, unknown name, ...)
        return [_error(model, f"Model could not be loaded: {e}") for _ in texts]
    try:
        if hasattr(m, "predict_batch"):
            results = m.predict_batch(list(texts))
        else:
            results = [m.predict(t) for t in texts]
    except Exception as e:  # noqa: BLE001
        return [_error(model, f"Prediction failed: {e}") for _ in texts]
    return [_finish(r, model, t) for r, t in zip(results, texts)]