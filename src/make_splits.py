import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import INTERIM_DIR, SPLIT_DIR, RANDOM_STATE

df = pd.read_csv(INTERIM_DIR / "master.csv", usecols=["row_id", "label"])

# 1) hold out 15% as test
train_val, test = train_test_split(
    df, test_size=0.15, stratify=df["label"], random_state=RANDOM_STATE)

# 2) from the remaining 85%, take 15/85 as validation (= 15% of the total)
train, val = train_test_split(
    train_val, test_size=0.15 / 0.85, stratify=train_val["label"],
    random_state=RANDOM_STATE)

out = pd.concat([
    train.assign(split="train"),
    val.assign(split="val"),
    test.assign(split="test"),
])[["row_id", "split"]].sort_values("row_id")

SPLIT_DIR.mkdir(parents=True, exist_ok=True)
out.to_csv(SPLIT_DIR / "split.csv", index=False)

print(out["split"].value_counts())
print(df.merge(out, on="row_id").groupby("split")["label"].mean())