"""Experiment: fine-tune DistilBERT for 3-class sentiment and compare to the
sklearn baselines (Logistic Regression / LinearSVC / LightGBM).

Fairness: uses the SAME rows, SAME label mapping (1-2/3/4-5 -> neg/neu/pos) and
the SAME train/test split (test_size=0.2, random_state=42, stratified) as the
notebook baselines. The split is reproduced from data/reviews_dashboard.parquet,
which holds the cleaned reviews in the same order the notebook had at split time.

Run via PowerShell (torch+CUDA segfaults under git-bash):
    python experiment_distilbert.py --smoke   # tiny/fast sanity run, nothing saved
    python experiment_distilbert.py           # full fine-tune on GPU, saves model

Notes:
- DistilBERT is fed RAW text (title + ". " + text); it has its own tokenizer and
  benefits from casing/punctuation, unlike the TF-IDF baselines that used cleaned
  text. Same rows/labels keep the comparison fair on the prediction target.
- A class-weighted cross-entropy counters the ~82/11/7 imbalance to give the weak
  `neutral` class a fair chance (its baseline recall was only ~0.17).
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

DATA = "data/reviews_dashboard.parquet"
MODEL_NAME = "distilbert-base-uncased"
OUT_DIR = "models/distilbert_sentiment"
METRICS_PATH = "models/distilbert_metrics.json"

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {lbl: i for i, lbl in enumerate(LABELS)}
ID2LABEL = {i: lbl for lbl, i in LABEL2ID.items()}

TRAIN_CAP = None  # None = use the full ~368k train split; set an int to subsample
MAX_LEN = 128


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_split(smoke):
    """Return train_df, test_df with columns input_text, label — identical split to baselines."""
    df = pd.read_parquet(DATA, columns=["title", "text", "sentiment_label"])
    df["input_text"] = (
        df["title"].fillna("").astype(str) + ". " + df["text"].fillna("").astype(str)
    )
    df["label"] = df["sentiment_label"].map(LABEL2ID)

    # Reproduce the baseline partition exactly: same n, seed, stratify order.
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        random_state=42,
        stratify=df["sentiment_label"],
    )
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    if smoke:
        train_df = train_df.groupby("label", group_keys=False).sample(
            n=400, random_state=42
        )
        test_df = test_df.groupby("label", group_keys=False).sample(
            n=200, random_state=42
        )
    elif TRAIN_CAP and len(train_df) > TRAIN_CAP:
        frac = TRAIN_CAP / len(train_df)
        train_df = train_df.groupby("label", group_keys=False).sample(
            frac=frac, random_state=42
        )

    return train_df, test_df


def class_weights(train_df):
    counts = train_df["label"].value_counts().sort_index().to_numpy()
    w = counts.sum() / (len(counts) * counts)  # inverse-frequency
    return torch.tensor(w, dtype=torch.float)


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy to help the rare neutral class."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits,
            labels,
            weight=self._class_weights.to(outputs.logits.device),
        )
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    preds = eval_pred.predictions.argmax(-1)
    labels = eval_pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }


def main(smoke):
    log(f"CUDA available: {torch.cuda.is_available()} | smoke={smoke}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    train_df, test_df = load_split(smoke)
    log(f"train={len(train_df):,} test={len(test_df):,}")
    log(f"train label dist: {train_df['label'].value_counts().sort_index().to_dict()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tok_fn(batch):
        return tokenizer(batch["input_text"], truncation=True, max_length=MAX_LEN)

    train_ds = Dataset.from_pandas(
        train_df[["input_text", "label"]], preserve_index=False
    ).map(tok_fn, batched=True, remove_columns=["input_text"])
    test_ds = Dataset.from_pandas(
        test_df[["input_text", "label"]], preserve_index=False
    ).map(tok_fn, batched=True, remove_columns=["input_text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )

    args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=1 if smoke else 2,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=128,
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights(train_df),
    )

    log("Training...")
    t0 = time.time()
    trainer.train()
    log(f"Training done in {(time.time() - t0) / 60:.1f} min")

    pred = trainer.predict(test_ds)
    y_pred = pred.predictions.argmax(-1)
    y_true = pred.label_ids
    report_txt = classification_report(y_true, y_pred, target_names=LABELS, digits=3)
    print(report_txt)

    summary = {
        "model": MODEL_NAME,
        "smoke": smoke,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "report": classification_report(
            y_true, y_pred, target_names=LABELS, output_dict=True
        ),
    }
    log(
        f"FINAL  acc={summary['accuracy']:.3f}  "
        f"macro_f1={summary['macro_f1']:.3f}  "
        f"weighted_f1={summary['weighted_f1']:.3f}"
    )

    if not smoke:
        trainer.save_model(OUT_DIR)
        tokenizer.save_pretrained(OUT_DIR)
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log(f"Saved model -> {OUT_DIR} | metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny fast sanity run")
    main(parser.parse_args().smoke)
