"""Inference helpers for the fine-tuned DistilBERT sentiment model.

Shared by the dashboard (app.py) and the data-build step (build_dashboard_data.py)
so both score reviews with the same transformer. The model is loaded once and
reused; it runs on the GPU when one is available, otherwise on CPU.
"""

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "models/distilbert_sentiment"

_model = None
_tokenizer = None


def load():
    """Load (once) the tokenizer and model, move the model to GPU if available."""
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        _model.eval()
        if torch.cuda.is_available():
            _model.to("cuda")
    return _model, _tokenizer


def label_list():
    """Class names ordered by the model's label ids (e.g. negative/neutral/positive)."""
    model, _ = load()
    return [model.config.id2label[i] for i in range(model.config.num_labels)]


def predict_proba(texts, batch_size=64, max_length=128):
    """Return an (n, 3) array of class probabilities for a list of raw texts."""
    model, tokenizer = load()
    device = next(model.parameters()).device
    probs = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(
                batch,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(probs)


def predict(texts, batch_size=64, max_length=128):
    """Return (labels, confidences) for a list of raw texts."""
    proba = predict_proba(texts, batch_size=batch_size, max_length=max_length)
    labels = label_list()
    ids = proba.argmax(axis=1)
    return [labels[i] for i in ids], proba.max(axis=1)
