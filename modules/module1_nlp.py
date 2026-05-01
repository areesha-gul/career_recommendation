"""Module 1: NLTK text preprocessing and MBTI dimension posteriors (MNB)."""

from __future__ import annotations

import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import ToktokTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_M1_MODEL_PATH = ROOT_DIR / "models" / "module1_mnb.pkl"
DIMENSION_PAIRS: list[tuple[str, str]] = [("E", "I"), ("N", "S"), ("T", "F"), ("J", "P")]
NLTK_RESOURCES = ["stopwords", "wordnet", "omw-1.4"]
_NLTK_READY = False
_LEMMATIZER = WordNetLemmatizer()
_TOKENIZER = ToktokTokenizer()
_STOPWORDS: set[str] = set()

MBTI_LABEL_PATTERN = re.compile(
    r"\b(infj|infp|intj|intp|isfj|isfp|istj|istp|enfj|enfp|entj|entp|esfj|esfp|estj|estp)\b",
    flags=re.IGNORECASE,
)


def _ensure_nltk_resources() -> None:
    """Download required NLTK resources if they are missing."""
    global _NLTK_READY, _STOPWORDS
    if _NLTK_READY:
        return

    for resource in NLTK_RESOURCES:
        try:
            if resource == "stopwords":
                nltk.data.find("corpora/stopwords")
            elif resource == "wordnet":
                nltk.data.find("corpora/wordnet")
            elif resource == "omw-1.4":
                nltk.data.find("corpora/omw-1.4")
        except LookupError:
            nltk.download(resource, quiet=True)

    _STOPWORDS = set(stopwords.words("english"))
    _NLTK_READY = True


def _normalize_text(text: str) -> str:
    """Normalize raw text by stripping URL/noise and removing MBTI label leakage."""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.replace("|||", " ")
    text = MBTI_LABEL_PATTERN.sub(" ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text: str) -> dict[str, Any]:
    """Preprocess free text into cleaned text, tokens, and keywords.

    Pipeline:
    1. Normalize and remove leakage/noise.
    2. Tokenize with NLTK.
    3. Remove stopwords and short tokens.
    4. Lemmatize and build keyword vector inputs.
    """
    if not text or not text.strip():
        return {"cleaned_text": "", "tokens": [], "keywords": []}

    _ensure_nltk_resources()

    normalized = _normalize_text(text)
    raw_tokens = _TOKENIZER.tokenize(normalized)
    alpha_tokens = [token for token in raw_tokens if token.isalpha()]
    tokens = [
        _LEMMATIZER.lemmatize(token)
        for token in alpha_tokens
        if token not in _STOPWORDS and len(token) > 2
    ]
    keywords = sorted(set(tokens))
    cleaned_text = " ".join(tokens)

    return {
        "cleaned_text": cleaned_text,
        "tokens": tokens,
        "keywords": keywords,
    }


def _default_posteriors() -> dict[str, float]:
    return {"E": 0.5, "I": 0.5, "N": 0.5, "S": 0.5, "T": 0.5, "F": 0.5, "J": 0.5, "P": 0.5}


def _mbti_dimension_targets(mbti_type: str) -> dict[tuple[str, str], str]:
    """Convert MBTI type into labels for each binary dimension pair."""
    mbti = str(mbti_type).upper().strip()
    if len(mbti) != 4:
        raise ValueError(f"Invalid MBTI label: {mbti_type}")

    return {
        ("E", "I"): mbti[0],
        ("N", "S"): mbti[1],
        ("T", "F"): mbti[2],
        ("J", "P"): mbti[3],
    }


def train_mnb_dimension_models(
    df: pd.DataFrame,
    min_df: int = 2,
    max_features: int = 10000,
) -> dict[str, Any]:
    """Train 4 binary MNB models and return a serializable bundle."""
    required_columns = {"mbti_type", "text"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Training dataframe must include columns: {required_columns}")

    cleaned_series = df["text"].astype(str).map(lambda item: preprocess_text(item)["cleaned_text"])
    vectorizer = CountVectorizer(min_df=min_df, max_features=max_features)
    x_matrix = vectorizer.fit_transform(cleaned_series)

    models: dict[str, MultinomialNB] = {}
    for pair in DIMENSION_PAIRS:
        y = df["mbti_type"].astype(str).map(lambda label: _mbti_dimension_targets(label)[pair])
        model = MultinomialNB(alpha=1.0)
        model.fit(x_matrix, y)
        models[f"{pair[0]}{pair[1]}"] = model

    return {
        "vectorizer": vectorizer,
        "models": models,
        "dimension_pairs": DIMENSION_PAIRS,
    }


def save_m1_bundle(bundle: dict[str, Any], path: Path = DEFAULT_M1_MODEL_PATH) -> Path:
    """Persist trained Module 1 bundle to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(bundle, file)
    return path


def load_m1_bundle(path: Path = DEFAULT_M1_MODEL_PATH) -> dict[str, Any] | None:
    """Load Module 1 bundle if it exists."""
    if not path.exists():
        return None
    with path.open("rb") as file:
        return pickle.load(file)


def compute_dimension_posteriors(cleaned_text: str, bundle: dict[str, Any] | None) -> dict[str, float]:
    """Compute posterior probabilities for E/I, N/S, T/F, J/P dimensions."""
    if not cleaned_text or bundle is None:
        return _default_posteriors()

    vectorizer: CountVectorizer = bundle["vectorizer"]
    models: dict[str, MultinomialNB] = bundle["models"]
    x_input = vectorizer.transform([cleaned_text])

    posteriors: dict[str, float] = {}
    for left, right in DIMENSION_PAIRS:
        model = models.get(f"{left}{right}")
        if model is None:
            posteriors[left] = 0.5
            posteriors[right] = 0.5
            continue

        proba = model.predict_proba(x_input)[0]
        class_prob = dict(zip(model.classes_, proba))
        posteriors[left] = float(class_prob.get(left, 0.5))
        posteriors[right] = float(class_prob.get(right, 0.5))

    return posteriors


def analyze_personality_text(text: str, bundle: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return preprocessed NLP features and MBTI dimension posterior scores.

    Contract:
    - Input: raw free text.
    - Output keys:
      - keywords: unique keyword list
      - keyword_vector: token-frequency dictionary
      - mbti_dimension_posteriors: probabilities for E/I, N/S, T/F, J/P
      - confidence: average max confidence across 4 dimensions
    """
    if bundle is None:
        bundle = load_m1_bundle()

    processed = preprocess_text(text)
    token_counts = Counter(processed["tokens"])
    posteriors = compute_dimension_posteriors(processed["cleaned_text"], bundle)

    pair_confidences = []
    for left, right in DIMENSION_PAIRS:
        pair_confidences.append(max(posteriors[left], posteriors[right]))
    confidence = float(sum(pair_confidences) / len(pair_confidences)) if pair_confidences else 0.0

    return {
        "cleaned_text": processed["cleaned_text"],
        "tokens": processed["tokens"],
        "keywords": processed["keywords"],
        "keyword_vector": dict(token_counts),
        "mbti_dimension_posteriors": posteriors,
        "confidence": confidence,
    }
