"""
FastText Quality Classification Module for Tier 2 (L2: Selection).

Part of the L0-L4 Tiered Data Management framework (arXiv:2602.09003).
Maps to paper methodology: Section 3 — High-throughput linear text classifiers
(e.g., FastText / CCNet style classifiers) widely deployed in production web
curation pipelines to filter high-quality educational/scientific documents.

Provides an alternative to the TF-IDF + LogisticRegression baseline.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def _check_fasttext() -> Any:
    """Import fasttext dynamically, raising clear installation instructions on failure."""
    try:
        import fasttext
        return fasttext
    except ImportError as err:
        raise ImportError(
            f"fasttext package is required for FastText quality classification ({err}).\n"
            "To install it, run:\n"
            "    pip install -r requirements-ml.txt\n"
            "or on Linux/Colab:\n"
            "    pip install fasttext\n"
            "or on Windows:\n"
            "    pip install fasttext-wheel"
        ) from err


def train_fasttext_classifier(
    train_texts: list[str],
    train_labels: list[int] | np.ndarray,
    epochs: int = 10,
    lr: float = 0.5,
    word_ngrams: int = 2,
    dim: int = 50,
) -> Any:
    """
    Train a supervised FastText quality classifier on labeled text samples.

    Creates a temporary file formatted for FastText supervised training:
        __label__<0|1> <single_line_normalized_text>

    Args:
        train_texts: List of document text strings.
        train_labels: Binary labels (0 = generic/low-info, 1 = high educational density).
        epochs: Number of training epochs.
        lr: Learning rate.
        word_ngrams: Max length of word n-gram features.
        dim: Embedding dimension.

    Returns:
        Trained FastText model object.
    """
    ft = _check_fasttext()

    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".txt") as tmp:
        tmp_path = tmp.name
        for text, label in zip(train_texts, train_labels):
            # FastText requires one line per document with no internal newlines
            clean_line = " ".join(str(text).split())
            if clean_line:
                tmp.write(f"__label__{int(label)} {clean_line}\n")

    try:
        model = ft.train_supervised(
            input=tmp_path,
            epoch=epochs,
            lr=lr,
            wordNgrams=word_ngrams,
            dim=dim,
            verbose=0,
        )
        return model
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def score_with_fasttext(model: Any, texts: list[str]) -> np.ndarray:
    """
    Score documents with a trained FastText classifier.

    Computes the predicted probability of the positive class (__label__1).

    Args:
        model: Trained FastText model.
        texts: List of document text strings.

    Returns:
        1D numpy array of probabilities [0.0 - 1.0] for class 1.
    """
    scores = []
    for text in texts:
        clean_line = " ".join(str(text).split())
        if not clean_line:
            scores.append(0.0)
            continue

        labels, probs = model.predict(clean_line, k=2)
        prob_dict = dict(zip(labels, probs))

        # Retrieve probability of class 1
        p1 = float(prob_dict.get("__label__1", 0.0))
        scores.append(round(p1, 4))

    return np.array(scores, dtype=float)
