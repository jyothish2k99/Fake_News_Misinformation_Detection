import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    text = re.sub(
        r"\(reuters\)",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    tokens = word_tokenize(text)

    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)


# Load model and TF-IDF
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")


# Get news from user
news = input("\nEnter news text: ")


# Apply same preprocessing used during training
cleaned_news = preprocess_text(news)


# Convert to TF-IDF
news_tfidf = tfidf.transform([cleaned_news])


# Predict
prediction = model.predict(news_tfidf)[0]


print("\nPrediction:", prediction)

