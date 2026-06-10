"""Shared NLP preprocessing utilities for the Amazon Review Intelligence project.

These functions are the single source of truth for text cleaning: the notebooks
and the dashboard import them, so inference-time preprocessing always matches
training. `models/final_sentiment_model.joblib` is pickled with a
`FunctionTransformer(clean_batch)` step that references `nlp_utils.clean_batch`,
so loading it only requires this module to be importable.
"""

import re
from html import unescape

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def clean_review_text(text):
    """Lowercase, expand contractions, strip html/urls/punctuation, normalize spaces."""
    if pd.isna(text):
        return ""

    text = unescape(str(text))
    text = text.lower()

    # contractions
    text = re.sub(r"\bcan't\b", "can not", text)
    text = re.sub(r"\bwon't\b", "will not", text)

    text = re.sub(r"\bit's\b", "it is", text)
    text = re.sub(r"\bthat's\b", "that is", text)
    text = re.sub(r"\bthere's\b", "there is", text)
    text = re.sub(r"\bhere's\b", "here is", text)
    text = re.sub(r"\bwhat's\b", "what is", text)

    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"'re\b", " are", text)
    text = re.sub(r"'d\b", " would", text)
    text = re.sub(r"'ll\b", " will", text)
    text = re.sub(r"'ve\b", " have", text)
    text = re.sub(r"'m\b", " am", text)

    # remove possessive 's
    text = re.sub(r"'s\b", "", text)

    # remove urls/html
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)

    # remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_batch(texts):
    """Batch wrapper used by the saved sentiment Pipeline's FunctionTransformer."""
    return [clean_review_text(t) for t in texts]


# Stopwords: keep "not" (matters for sentiment), drop contraction leftovers.
stop_words = set(ENGLISH_STOP_WORDS) - {"not"}
extra_stop_words = {"t", "s", "m", "don", "ve", "ll", "re", "d"}


def remove_stopwords(text):
    """Drop stopwords, contraction leftovers and 1-char tokens (for topic modeling)."""
    words = text.split()
    words = [
        word
        for word in words
        if word not in stop_words
        and word not in extra_stop_words
        and len(word) > 1
    ]
    return " ".join(words)


def build_full_text(title, text):
    """Combine title + text the same way the notebook did before cleaning."""
    title = "" if pd.isna(title) else str(title)
    text = "" if pd.isna(text) else str(text)
    return f"{title}. {text}"
