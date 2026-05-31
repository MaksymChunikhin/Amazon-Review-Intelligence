"""Shared NLP preprocessing utilities for the Amazon Review Intelligence project.

These functions mirror the cleaning logic used inside the notebook so that the
dashboard (and the saved end-to-end sentiment pipeline) preprocess text exactly
the same way at inference time as during training.

IMPORTANT: `models/final_sentiment_model.joblib` was pickled with a
`FunctionTransformer(clean_batch)` step whose function reference points at
`__main__.clean_batch`. Any script that loads that model must expose
`clean_review_text` and `clean_batch` in `__main__`. Use `register_main()` for that.
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


def register_main():
    """Expose cleaning fns in __main__ so the pickled sentiment model can unpickle."""
    import __main__

    __main__.clean_review_text = clean_review_text
    __main__.clean_batch = clean_batch
