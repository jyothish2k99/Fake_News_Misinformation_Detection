"""Project-wide paths and constants. Always run scripts from the repo root:
    python src/preprocessing.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

RAW_TRUE = DATA_DIR / "True.csv"
RAW_FAKE = DATA_DIR / "Fake.csv"
PROCESSED_CSV = DATA_DIR / "processed_news.csv"
FEATURES_CSV = DATA_DIR / "features_news.csv"

MODEL_PATH = MODELS_DIR / "model.pkl"
TFIDF_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"

RANDOM_STATE = 42

for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
