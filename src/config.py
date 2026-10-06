from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
SPLIT_DIR = ROOT / "data" / "splits"
MODEL_DIR = ROOT / "models" / "baseline"
RESULTS_DIR = ROOT / "results" / "baseline"

RANDOM_STATE = 42
FAKE_LABEL, REAL_LABEL = 1, 0