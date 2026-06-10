"""Fit BERTopic on a sample of reviews to compare with the existing LDA topics.

This is additive: the LDA pipeline and the dashboard are left untouched. BERTopic
builds topics from sentence embeddings (semantics) instead of word counts, finds the
number of topics automatically, and can name them from their distinctive words.

Steps: embed reviews on the GPU (MiniLM) -> UMAP + HDBSCAN clustering (inside
BERTopic) -> c-TF-IDF topic words. Saves the fitted model and topic tables so the
notebook can present results without re-running the heavy fit.

Run from PowerShell (uses GPU embeddings):
    python build_bertopic.py
"""

import re
import time
import warnings
from html import unescape

import pandas as pd
import torch
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore")


def light_clean(text):
    # Strip HTML (notably <br>) so it doesn't leak into topic words; keep case
    # and punctuation, which the embedding model benefits from.
    text = unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()

DATA = "data/reviews_dashboard.parquet"
EMBED_MODEL = "all-MiniLM-L6-v2"
SAMPLE = 50_000
REDUCED_COUNT = 25  # a practical middle ground, still far finer than LDA's 12

MODEL_DIR = "models/bertopic_model"
INFO_AUTO = "models/bertopic_topics_auto.csv"
INFO_REDUCED = "models/bertopic_topics_reduced25.csv"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")

    df = pd.read_parquet(DATA, columns=["title", "text"])
    docs = (
        df["title"].fillna("").astype(str) + ". " + df["text"].fillna("").astype(str)
    )
    docs = docs.sample(n=min(SAMPLE, len(docs)), random_state=42).tolist()
    docs = [light_clean(d) for d in docs]
    log(f"Sampled {len(docs):,} reviews")

    log("Embedding reviews on GPU (MiniLM)...")
    embedder = SentenceTransformer(EMBED_MODEL, device=device)
    embeddings = embedder.encode(docs, batch_size=256, show_progress_bar=True)

    # English stopwords + bigrams give readable, distinctive topic words.
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=10)
    topic_model = BERTopic(
        embedding_model=embedder,
        vectorizer_model=vectorizer,
        min_topic_size=150,
        calculate_probabilities=False,
        verbose=True,
    )

    log("Fitting BERTopic (UMAP + HDBSCAN + c-TF-IDF)...")
    t0 = time.time()
    topics, _ = topic_model.fit_transform(docs, embeddings)
    log(f"Fit done in {(time.time() - t0) / 60:.1f} min")

    info_auto = topic_model.get_topic_info()
    n_topics = (info_auto["Topic"] >= 0).sum()
    n_outliers = int(info_auto.loc[info_auto["Topic"] == -1, "Count"].sum() or 0)
    log(f"Topics found automatically: {n_topics} (+ {n_outliers:,} outlier docs)")
    info_auto.to_csv(INFO_AUTO, index=False)

    log("Top automatically-found topics:")
    print(info_auto.head(20).to_string(index=False))

    log(f"Reducing to ~{REDUCED_COUNT} topics; outliers kept as a separate 'Other' group...")
    topic_model.reduce_topics(docs, nr_topics=REDUCED_COUNT)
    info_reduced = topic_model.get_topic_info()
    n_red = (info_reduced["Topic"] >= 0).sum()
    red_out = int(info_reduced.loc[info_reduced["Topic"] == -1, "Count"].sum() or 0)
    log(f"Reduced topics: {n_red} (+ {red_out:,} outliers shown as 'Other')")
    info_reduced.to_csv(INFO_REDUCED, index=False)
    print(info_reduced.to_string(index=False))

    topic_model.save(
        MODEL_DIR, serialization="safetensors", save_embedding_model=False
    )
    log(f"Saved model -> {MODEL_DIR} | tables -> {INFO_AUTO}, {INFO_REDUCED}")


if __name__ == "__main__":
    main()
