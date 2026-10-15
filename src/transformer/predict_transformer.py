"""Reusable DistilBERT predictor. Returns the same dict format as BaselinePredictor.

Usage:
    from src.transformer.predict_transformer import TransformerPredictor
    pred = TransformerPredictor()
    pred.predict("some article text")
    pred.predict_batch(["text 1", "text 2"])
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # avoids a Windows OpenMP crash

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import ROOT

DEFAULT_PATH = ROOT / "models" / "transformer" / "distilbert"
MAX_LEN = 256


class TransformerPredictor:
    def __init__(self, model_path=DEFAULT_PATH, preprocess_fn=None, max_len=MAX_LEN):
        if not (model_path / "model.safetensors").exists():
            raise FileNotFoundError(
                f"No trained model in {model_path}. "
                "Download the DistilBERT files into models/transformer/distilbert/.")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = (AutoModelForSequenceClassification
                      .from_pretrained(model_path).to(self.device).eval())
        self.preprocess_fn = preprocess_fn   # None = use the text as is (how the model was trained)
        self.max_len = max_len

    @staticmethod
    def _error(msg):
        return {"label": "Unknown", "confidence": None, "prob_fake": None, "error": msg}

    @staticmethod
    def _result(prob_fake):
        return {"label": "Fake" if prob_fake >= 0.5 else "Real",
                "confidence": max(prob_fake, 1 - prob_fake),
                "prob_fake": prob_fake}

    def predict(self, text):
        return self.predict_batch([text])[0]

    def predict_batch(self, texts, batch_size=16):
        results = [None] * len(texts)
        valid = []                                    # (position, text) pairs
        for i, t in enumerate(texts):
            if not isinstance(t, str) or not t.strip():
                results[i] = self._error("Empty input")
            else:
                valid.append((i, self.preprocess_fn(t) if self.preprocess_fn else t))

        try:
            with torch.no_grad():
                for s in range(0, len(valid), batch_size):
                    chunk = valid[s:s + batch_size]
                    enc = self.tokenizer([t for _, t in chunk], truncation=True,
                                         max_length=self.max_len, padding=True,
                                         return_tensors="pt").to(self.device)
                    probs = torch.softmax(self.model(**enc).logits.float(), dim=1)[:, 1]
                    for (pos, _), p in zip(chunk, probs.cpu().tolist()):
                        results[pos] = self._result(float(p))
        except Exception as e:  # noqa: BLE001
            for pos, _ in valid:
                if results[pos] is None:
                    results[pos] = self._error(f"Prediction failed: {e}")
        return results