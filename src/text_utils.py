"""Single source of truth for text cleaning.

Used by preprocessing.py (training data), feature_extraction.py and
test_model.py (prediction), so the model always sees text cleaned exactly the
same way it was trained on.
"""
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

for _res in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
    nltk.download(_res, quiet=True)

# Keep negations: "not", "no" carry meaning for fake/real claims.
KEEP_WORDS = {"no", "not", "nor", "against"}
STOP_WORDS = set(stopwords.words("english")) - KEEP_WORDS
LEMMATIZER = WordNetLemmatizer()

WEEKDAYS = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
# Layout words of fake sites ("WATCH:", "(VIDEO)", "via Twitter"), not content
STYLE_WORDS = {"video", "watch", "via"}

# ---------------------------------------------------------------------------
# 1. Source markers (these identify the PUBLISHER, not whether a claim is true).
#    If left in, the model just learns "Reuters style = real".
# ---------------------------------------------------------------------------
_DATELINE = re.compile(r"^[^()\n]{0,80}\(\s*reuters\s*\)\s*[-\u2013\u2014]*\s*", re.I)
# Dateline anywhere in the text: "SEATTLE/WASHINGTON (Reuters) -"
_DATELINE_ANY = re.compile(
    r"[A-Z][A-Z/ .,'\-]{2,60}\(\s*(?i:reuters)\s*\)\s*[-\u2013\u2014]*\s*"
)
_REUTERS_WORD = re.compile(r"\bthomson\s+reuters\b|\breuters\b", re.I)
_REUTERS_FOOTER = re.compile(
    r"our\s+standards:?\s*the\s+(?:thomson\s+reuters\s+)?trust\s+principles\.?"
    r"|\b(?:additional\s+)?(?:reporting|writing|editing)\s+by\s+[^\n.]{0,150}",
    re.I,
)
# Typical fake-site boilerplate: image credits, "read more", etc.
_FAKE_FOOTER = re.compile(
    r"featured\s+image(?:\s+via)?[^\n.]{0,120}"
    r"|\b(?:image|photo|screenshot|video|screengrab)\s+(?:via|by|courtesy\s+of|credit)\b[^\n.]{0,80}"
    r"|\bread\s+more\b:?"
    r"|\b21st\s+century\s+wire\b"
    r"|\bpic\.twitter\.com/\S+"
    r"|\bgetty\s+images\b",
    re.I,
)
# "don't" / "don t" (the dataset often drops the apostrophe) -> "do not"
_WONT = re.compile(r"\bwon(?:['\u2019]|\s)t\b")
_CANT = re.compile(r"\bcan(?:['\u2019]|\s)t\b")
_NT = re.compile(r"\b(\w+)n(?:['\u2019]|\s)t\b")
_URL = re.compile(r"https?://\S+|www\.\S+", re.I)


def remove_source_markers(text):
    """Strip publisher fingerprints from RAW text (keeps original casing and
    punctuation so linguistic features such as capital ratio still work)."""
    if not isinstance(text, str):
        return ""
    text = _DATELINE.sub("", text, count=1)      # "Washington (Reuters) - " at start
    text = _DATELINE_ANY.sub(" ", text)          # "WASHINGTON (Reuters) - " anywhere
    text = _REUTERS_FOOTER.sub(" ", text)
    text = _REUTERS_WORD.sub(" ", text)
    text = _FAKE_FOOTER.sub(" ", text)
    text = _URL.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# 2. Full NLP preprocessing for the model
# ---------------------------------------------------------------------------
def _tokenize(text):
    try:
        return word_tokenize(text)
    except LookupError:          # punkt not available -> safe fallback
        return text.split()


def preprocess_text(text, remove_weekdays=True):
    """raw text -> cleaned string ready for TF-IDF."""
    text = remove_source_markers(text)
    text = text.lower()
    text = _WONT.sub("will not", text)
    text = _CANT.sub("can not", text)
    text = _NT.sub(r"\1 not", text)
    if remove_weekdays:           # "said on Tuesday" is Reuters house style
        text = re.sub(rf"\b{WEEKDAYS}\b", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = _tokenize(text)
    tokens = [t for t in tokens if len(t) > 1 and t not in STOP_WORDS and t not in STYLE_WORDS]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)
