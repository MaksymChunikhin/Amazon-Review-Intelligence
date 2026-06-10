"""Tune the LinearSVC baseline with RandomizedSearchCV (C + TF-IDF params).

The original baseline used a fixed, *untuned* LinearSVC (default C=1.0) on a fixed
TF-IDF (uni+bigrams, 50k features). This script searches over C and the main TF-IDF
knobs, optimising macro-F1 (the metric that actually matters on this imbalanced data),
to show how high a *tuned* classical baseline can reach vs the untuned one and DistilBERT.

Fairness: same rows, same label mapping, same train/test split (random_state=42,
stratified), same cleaned-text input (clean_text) as the notebook baselines.

Run:  python tune_linearsvc.py
"""

import json
import time
import warnings

import joblib
import pandas as pd
from scipy.stats import loguniform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import nlp_utils

warnings.filterwarnings("ignore")

DATA = "data/reviews_dashboard.parquet"
OUT_MODEL = "models/svc_tuned_pipeline.joblib"
METRICS_PATH = "models/svc_tuned_metrics.json"

SEARCH_SAMPLE = None  # None = run the search on the FULL train set (no subsample)
N_ITER = 15
CV = 3
LABELS = ["negative", "neutral", "positive"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log(f"Loading {DATA}")
    df = pd.read_parquet(DATA, columns=["title", "text", "sentiment_label"])
    full = (
        df["title"].fillna("").astype(str) + ". " + df["text"].fillna("").astype(str)
    )
    # Baseline used clean_text -> reconstruct it with the same cleaner.
    X = full.map(nlp_utils.clean_review_text)
    y = df["sentiment_label"]

    # Identical split to the baselines.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log(f"train={len(X_train):,} test={len(X_test):,}")

    # Search on the full train set (optionally subsample if SEARCH_SAMPLE is set).
    if SEARCH_SAMPLE and len(X_train) > SEARCH_SAMPLE:
        X_search, _, y_search, _ = train_test_split(
            X_train,
            y_train,
            train_size=SEARCH_SAMPLE,
            random_state=42,
            stratify=y_train,
        )
    else:
        X_search, y_search = X_train, y_train
    log(f"search set={len(X_search):,} (full train)")

    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LinearSVC(class_weight="balanced", random_state=42)),
        ]
    )
    param_dist = {
        "tfidf__max_features": [20_000, 50_000, 100_000],
        "tfidf__min_df": [2, 5, 10],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__sublinear_tf": [True, False],
        "clf__C": loguniform(1e-2, 1e1),
    }

    search = RandomizedSearchCV(
        pipe,
        param_dist,
        n_iter=N_ITER,
        scoring="f1_macro",
        cv=CV,
        n_jobs=4,
        random_state=42,
        verbose=2,
    )
    log("Running RandomizedSearchCV (macro-F1)...")
    t0 = time.time()
    search.fit(X_search, y_search)
    log(f"Search done in {(time.time() - t0) / 60:.1f} min")
    log(f"Best CV macro-F1 (full train): {search.best_score_:.4f}")
    log(f"Best params: {search.best_params_}")

    # Refit the best configuration on the FULL train set, evaluate on FULL test.
    best = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LinearSVC(class_weight="balanced", random_state=42)),
        ]
    ).set_params(**search.best_params_)
    log("Refitting best config on full train...")
    best.fit(X_train, y_train)
    y_pred = best.predict(X_test)

    report_txt = classification_report(y_test, y_pred, target_names=LABELS, digits=3)
    print(report_txt)

    summary = {
        "best_params": {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in search.best_params_.items()
        },
        "cv_macro_f1": float(search.best_score_),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "test_weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
        "report": classification_report(
            y_test, y_pred, target_names=LABELS, output_dict=True
        ),
    }
    log(
        f"TUNED TEST  acc={summary['test_accuracy']:.3f}  "
        f"macro_f1={summary['test_macro_f1']:.3f}  "
        f"weighted_f1={summary['test_weighted_f1']:.3f}  "
        f"neutral_recall={summary['report']['neutral']['recall']:.3f}"
    )

    joblib.dump(best, OUT_MODEL)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"Saved model -> {OUT_MODEL} | metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
