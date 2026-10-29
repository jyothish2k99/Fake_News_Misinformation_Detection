"""DistilBERT inference service.

Loads the fine-tuned DistilBertForSequenceClassification model from
artifacts/transformer/distilbert and exposes the same predict() interface
shape as ModelService so the API endpoint can call both in parallel.
"""
from __future__ import annotations
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoids Windows OpenMP crash

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"
DISTILBERT_PATH = ARTIFACT_DIR / "transformer" / "distilbert"
MAX_LEN = 256

# Lazy-loaded singletons so imports don't block startup
_tokenizer = None
_model = None
_device = None
_load_error: str | None = None


def _try_load():
    global _tokenizer, _model, _device, _load_error
    if _tokenizer is not None or _load_error is not None:
        return
    if not (DISTILBERT_PATH / "model.safetensors").exists():
        _load_error = (
            f"DistilBERT artifact not found at {DISTILBERT_PATH}. "
            "Copy the trained model files into backend/artifacts/transformer/distilbert/."
        )
        return
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(str(DISTILBERT_PATH))
        model = (
            AutoModelForSequenceClassification.from_pretrained(str(DISTILBERT_PATH))
            .to(device)
            .eval()
        )
        _tokenizer = tokenizer
        _model = model
        _device = device
    except Exception as exc:  # noqa: BLE001
        _load_error = f"Failed to load DistilBERT model: {exc}"


class TransformerService:
    """Wraps the fine-tuned DistilBERT model for sequence prediction."""

    def __init__(self):
        _try_load()

    @property
    def ready(self) -> bool:
        _try_load()
        return _model is not None and _load_error is None

    @property
    def load_error(self) -> str | None:
        return _load_error

    def predict(self, title: str, text: str) -> dict:
        """Return a prediction dict compatible with the LR model's output shape."""
        if not self.ready:
            return {
                "label": "UNAVAILABLE",
                "fake_probability": None,
                "real_probability": None,
                "confidence": None,
                "error": _load_error or "DistilBERT model not loaded.",
                "disclaimer": "DistilBERT prediction unavailable.",
                "model": "DistilBERT",
            }
        combined = (title.strip() + " " + text.strip()).strip()
        if not combined:
            return {
                "label": "UNAVAILABLE",
                "fake_probability": None,
                "real_probability": None,
                "confidence": None,
                "error": "Empty input.",
                "disclaimer": "",
                "model": "DistilBERT",
            }
        try:
            import torch

            enc = _tokenizer(
                combined,
                truncation=True,
                max_length=MAX_LEN,
                padding=True,
                return_tensors="pt",
            ).to(_device)
            with torch.no_grad():
                logits = _model(**enc).logits.float()
                probs = torch.softmax(logits, dim=1)[0].cpu().tolist()

            # config.json label mapping: id2label expected as {0: "REAL", 1: "FAKE"}
            # The model was trained with prob index 1 = FAKE (matches TransformerPredictor)
            prob_fake = float(probs[1])
            prob_real = float(probs[0])
            label = "FAKE-LIKE" if prob_fake >= 0.5 else "REAL-LIKE"
            confidence = max(prob_fake, prob_real)
            return {
                "label": label,
                "fake_probability": round(prob_fake, 4),
                "real_probability": round(prob_real, 4),
                "confidence": round(confidence, 4),
                "disclaimer": (
                    "DistilBERT prediction is based on fine-tuned patterns in the training "
                    "corpus. It is not a fact-check or proof of truthfulness."
                ),
                "model": "DistilBERT",
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "label": "UNAVAILABLE",
                "fake_probability": None,
                "real_probability": None,
                "confidence": None,
                "error": f"Prediction failed: {exc}",
                "disclaimer": "",
                "model": "DistilBERT",
            }
