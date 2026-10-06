import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Download required NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


# Load NLP tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# Text preprocessing function
def preprocess_text(text):

    # Handle missing or non-text values
    if not isinstance(text, str):
        return ""

    # Convert text to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # Lemmatize
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    # Join words back into text
    cleaned_text = " ".join(tokens)

    return cleaned_text


# ==============================
# LOAD DATASETS
# ==============================

true_data = pd.read_csv("data/True.csv")
fake_data = pd.read_csv("data/Fake.csv")


# Add labels
true_data["label"] = "REAL"
fake_data["label"] = "FAKE"


# Combine both datasets
data = pd.concat(
    [true_data, fake_data],
    ignore_index=True
)


# Remove duplicate rows
data = data.drop_duplicates()


# ==============================
# TEXT PREPROCESSING
# ==============================

data["cleaned_text"] = data["text"].apply(preprocess_text)


# ==============================
# CHECK RESULTS
# ==============================

print("Dataset shape:", data.shape)

print("\nLabel distribution:")
print(data["label"].value_counts())

print("\nOriginal text:")
print(data["text"].iloc[0])

print("\nCleaned text:")
print(data["cleaned_text"].iloc[0])


# ==============================
# SAVE PROCESSED DATASET
# ==============================

data.to_csv(
    "data/processed_news.csv",
    index=False
)

print("\nProcessed dataset saved successfully!")
