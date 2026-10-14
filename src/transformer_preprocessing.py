from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

# Project paths
ROOT = Path(__file__).resolve().parents[1]

DATA_PATHS = [
    ROOT / "data" / "processed_news.csv",
    ROOT.parent / "data" / "processed_news.csv",
]

DATA_PATH = next(
    (path for path in DATA_PATHS if path.exists()),
    None
)

if DATA_PATH is None:
    raise FileNotFoundError(
        "processed_news.csv not found. Check the data folder."
    )

# Load and clean dataset
df = pd.read_csv(DATA_PATH)
df = df[["cleaned_text", "label"]].dropna().copy()

df["label"] = (
    df["label"]
    .str.upper()
    .map({"FAKE": 0, "REAL": 1})
)

df = df.dropna(subset=["label"]).reset_index(drop=True)
df["label"] = df["label"].astype(np.int32)

# Load DistilBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

max_length = 256
batch_size = 512
num_articles = len(df)

# Prepare arrays for tokenized data
input_ids = np.zeros(
    (num_articles, max_length), dtype=np.int32
)
attention_masks = np.zeros(
    (num_articles, max_length), dtype=np.int32
)

# Tokenize in batches
texts = df["cleaned_text"].astype(str).tolist()

for start in range(0, num_articles, batch_size):
    end = min(start + batch_size, num_articles)

    encoded = tokenizer(
        texts[start:end],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    input_ids[start:end] = np.asarray(
        encoded["input_ids"], dtype=np.int32
    )

    attention_masks[start:end] = np.asarray(
        encoded["attention_mask"], dtype=np.int32
    )

    print(f"Processed {end}/{num_articles} articles")

# Save tokenized data outside the code repository
output_dir = ROOT.parent / "data"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "transformer_tokenized.npz"

np.savez_compressed(
    output_file,
    input_ids=input_ids,
    attention_mask=attention_masks,
    labels=df["label"].to_numpy(dtype=np.int32),
)

print("\nTransformer preprocessing completed!")
print("Total articles:", num_articles)
print("Input IDs shape:", input_ids.shape)
print("Attention mask shape:", attention_masks.shape)
print("Labels shape:", df["label"].shape)
print("Saved file:", output_file)
print("Label mapping: FAKE = 0, REAL = 1")