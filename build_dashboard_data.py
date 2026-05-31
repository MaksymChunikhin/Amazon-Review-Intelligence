"""Build an enriched parquet for the Streamlit dashboard.

The raw `data/reviews_processed.parquet` only holds the ingested columns. The
notebook derives sentiment labels, cleaned text, model predictions and LDA topics
at runtime. Recomputing those for 500k rows on every dashboard load would be slow,
so we precompute them once here and save a slim `data/reviews_dashboard.parquet`
that the dashboard can load instantly.

Predictions use the fine-tuned DistilBERT (the best model), so run on a machine
with the GPU torch build. On Windows run from PowerShell:
    python build_dashboard_data.py
"""

import time
import warnings

import joblib
import numpy as np
import pandas as pd

import distilbert_utils
import nlp_utils

warnings.filterwarnings("ignore")

RAW_PATH = "data/reviews_processed.parquet"
OUT_PATH = "data/reviews_dashboard.parquet"

LDA_MODEL = "models/lda_model.joblib"
TOPIC_VECTORIZER = "models/topic_vectorizer.joblib"

# Business labels for LDA topics. Mirrors the mapping fixed in the notebook by
# inspecting print_topics(); if LDA is retrained, this must be re-verified.
TOPIC_LABELS = {
    0: "Ease of Use & Cleaning",
    1: "Bedding & Sleep",
    2: "Bath & Soft Furnishings",
    3: "Home Appliances (Fans & Vacuums)",
    4: "Price, Value & Delivery Issues",
    5: "Furniture & Cutlery",
    6: "Size, Fit & Storage",
    7: "Cookware & Kitchen",
    8: "Drinkware & Beverages",
    9: "Gifts & Recommendations",
    10: "Appearance & Design",
    11: "Complaints & Malfunctions",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log(f"Loading raw parquet: {RAW_PATH}")
    df = pd.read_parquet(RAW_PATH)
    log(f"  raw shape: {df.shape}")

    # Cleaning, mirrors notebook section 2. Drop the `images` column first: it
    # holds numpy arrays (unhashable), which would break drop_duplicates().
    df = df.drop(columns=[c for c in ["images"] if c in df.columns])
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset="text")
    df = df.reset_index(drop=True)
    df["text_length"] = df["text"].str.len()
    df = df[df["text_length"] >= 20].drop(columns="text_length").reset_index(drop=True)
    log(f"  after dedup + short-review filter: {df.shape}")

    # Build the raw title+text input and the cleaned variant used for topics
    log("Building full_text (title + text) and cleaning...")
    full_text = (
        df["title"].fillna("").astype(str) + ". " + df["text"].fillna("").astype(str)
    )
    df["clean_text"] = full_text.map(nlp_utils.clean_review_text)
    df["clean_text_no_stopwords"] = df["clean_text"].map(nlp_utils.remove_stopwords)

    # Ground-truth sentiment derived from the star rating (notebook section 5)
    conditions = [
        df["rating"].isin([1, 2]),
        df["rating"].eq(3),
        df["rating"].isin([4, 5]),
    ]
    df["sentiment_label"] = np.select(
        conditions, ["negative", "neutral", "positive"], default="unknown"
    )
    assert (df["sentiment_label"] == "unknown").sum() == 0

    # Predictions from the fine-tuned DistilBERT (the best model), on raw
    # title+text. Runs on GPU if available; this is the slow part.
    log("Predicting sentiment + confidence with DistilBERT...")
    labels, confidence = distilbert_utils.predict(full_text.tolist())
    df["predicted_label"] = labels
    df["confidence"] = confidence
    log("  predictions done")

    # Topic modeling: LDA on the stopword-free cleaned text
    log("Loading LDA + vectorizer")
    lda = joblib.load(LDA_MODEL)
    vectorizer = joblib.load(TOPIC_VECTORIZER)
    log("Vectorizing + LDA transform...")
    counts = vectorizer.transform(df["clean_text_no_stopwords"])
    topic_dist = lda.transform(counts)
    df["dominant_topic"] = topic_dist.argmax(axis=1)
    df["topic_label"] = df["dominant_topic"].map(TOPIC_LABELS)
    log("  topics assigned")

    # Year-month helper for the time-series charts
    df["year_month"] = df["timestamp"].dt.to_period("M").astype(str)

    # Keep only the columns the dashboard needs
    keep = [
        "rating",
        "title",
        "text",
        "timestamp",
        "year_month",
        "helpful_vote",
        "verified_purchase",
        "sentiment_label",
        "predicted_label",
        "confidence",
        "dominant_topic",
        "topic_label",
    ]
    out = df[keep].copy()
    out.to_parquet(OUT_PATH, index=False)
    log(f"Saved dashboard parquet: {OUT_PATH}  shape={out.shape}")

    # Quick sanity summary
    log("Sentiment (ground truth) distribution:")
    print(out["sentiment_label"].value_counts(normalize=True).round(3).to_string())
    log("Top topics:")
    print(out["topic_label"].value_counts().head().to_string())


if __name__ == "__main__":
    main()
