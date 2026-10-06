import pandas as pd
from src.config import RAW_DIR, INTERIM_DIR, FAKE_LABEL, REAL_LABEL


def build_master():
    fake = pd.read_csv(RAW_DIR / "Fake.csv")
    fake["label"] = FAKE_LABEL
    real = pd.read_csv(RAW_DIR / "True.csv")
    real["label"] = REAL_LABEL
    df = pd.concat([fake, real], ignore_index=True)
    print("Raw rows:", len(df))

    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")

    # drop articles with an empty body (almost all are Fake -> shortcut risk)
    df = df[df["text"].str.strip().str.len() > 0]
    print("After removing blank text:", len(df))

    df["raw_text"] = (df["title"] + ". " + df["text"]).str.strip()

    # drop duplicates; also drop any article that appears with BOTH labels
    conflict = df.groupby("raw_text")["label"].nunique()
    conflict_texts = conflict[conflict > 1].index
    print("Texts labelled both Fake and Real:", len(conflict_texts))
    df = df[~df["raw_text"].isin(conflict_texts)]

    df = df.drop_duplicates(subset="raw_text").reset_index(drop=True)
    print("After removing duplicates:", len(df))

    df.insert(0, "row_id", range(len(df)))

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(INTERIM_DIR / "master.csv", index=False)
    print(df["label"].value_counts())


if __name__ == "__main__":
    build_master()