"""
Tier 2 (L2: Selection) pure logic and modeling module.

Supports Phase 2: Model-Driven Selection.
Maps to paper methodology: selecting high-quality educational/informative tokens
using lightweight models trained with weak supervision.
All preprocessing is fitted strictly on the training split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True, slots=True)
class L2Config:
    """Configuration for L2 selection."""
    selection_threshold: float = 0.5
    top_k_ratio: float = 0.5
    test_size: float = 0.3
    random_state: int = 42
    max_tfidf_features: int = 300


# Informative domain keyword indicators for demo weak labeling
_INFORMATIVE_KEYWORDS = frozenset([
    "architecture", "model", "learning", "network", "system", "process",
    "analysis", "method", "algorithm", "representation", "data", "parameter",
    "structure", "function", "hypothesis", "transformer", "optimization",
])


def assign_weak_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Generate deterministic weak supervision labels for demo training.

    Labels are generated based on a composite heuristic of informational density:
    - Sufficient length (words >= 40)
    - High alpha ratio (>= 0.70)
    - Low symbol ratio (<= 0.15)
    - Presence of at least 2 informative educational/technical keywords

    These labels serve as proxy ground-truth for training the L2 lightweight selector.
    """
    labels = []
    for _, row in df.iterrows():
        text = str(row.get("text_clean", "")).lower()
        word_count = row.get("word_count", len(text.split()))
        alpha_ratio = row.get("alpha_ratio", 0.0)
        symbol_ratio = row.get("symbol_ratio", 0.0)

        keyword_hits = sum(1 for kw in _INFORMATIVE_KEYWORDS if kw in text)

        is_high_quality = (
            word_count >= 40
            and alpha_ratio >= 0.70
            and symbol_ratio <= 0.15
            and keyword_hits >= 2
        )
        labels.append(1 if is_high_quality else 0)

    return np.array(labels, dtype=int)


def build_feature_pipeline(max_tfidf_features: int = 300) -> ColumnTransformer:
    """
    Construct preprocessing pipeline for text and numeric features.
    
    Numeric features are scaled; text is vectorised with TF-IDF.
    """
    numeric_features = ["char_len", "word_count", "alpha_ratio", "symbol_ratio"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=max_tfidf_features, stop_words="english", ngram_range=(1, 2)), "text_clean"),
            ("numeric", StandardScaler(), numeric_features),
        ],
        remainder="drop",
    )
    return preprocessor


def train_l2_selector(
    df: pd.DataFrame,
    labels: np.ndarray,
    config: L2Config,
) -> tuple[Pipeline, dict[str, float]]:
    """
    Train L2 selection model.

    CRITICAL: Preprocessing is fit strictly on the training split only to avoid data leakage.
    
    Returns:
        (fitted_pipeline, train_test_metrics)
    """
    # Ensure all required numeric features exist
    df_work = df.copy()
    if "char_len" not in df_work.columns:
        df_work["char_len"] = df_work["text_clean"].str.len()
    if "word_count" not in df_work.columns:
        df_work["word_count"] = df_work["text_clean"].apply(lambda t: len(str(t).split()))
    if "alpha_ratio" not in df_work.columns:
        df_work["alpha_ratio"] = df_work["text_clean"].apply(
            lambda t: sum(1 for c in str(t) if c.isalpha()) / max(len(str(t)), 1)
        )
    if "symbol_ratio" not in df_work.columns:
        df_work["symbol_ratio"] = df_work["text_clean"].apply(
            lambda t: sum(1 for c in str(t) if not c.isalnum() and not c.isspace()) / max(len(str(t)), 1)
        )

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df_work,
        labels,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=labels if len(np.unique(labels)) > 1 and min(np.bincount(labels)) >= 2 else None,
    )

    preprocessor = build_feature_pipeline(config.max_tfidf_features)
    classifier = LogisticRegression(random_state=config.random_state, max_iter=1000)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    # Fit pipeline ONLY on the training split
    pipeline.fit(X_train, y_train)

    # Evaluate on both splits
    train_acc = float(pipeline.score(X_train, y_train))
    test_acc = float(pipeline.score(X_test, y_test)) if len(X_test) > 0 else train_acc

    metrics = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "positive_label_ratio": float(np.mean(labels)),
    }

    return pipeline, metrics


def score_documents(pipeline: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """
    Score documents using the trained L2 pipeline.
    
    Appends 'quality_score' (predicted probability of positive class) to records.
    """
    df_scored = df.copy()
    if "char_len" not in df_scored.columns:
        df_scored["char_len"] = df_scored["text_clean"].str.len()
    if "word_count" not in df_scored.columns:
        df_scored["word_count"] = df_scored["text_clean"].apply(lambda t: len(str(t).split()))
    if "alpha_ratio" not in df_scored.columns:
        df_scored["alpha_ratio"] = df_scored["text_clean"].apply(
            lambda t: sum(1 for c in str(t) if c.isalpha()) / max(len(str(t)), 1)
        )
    if "symbol_ratio" not in df_scored.columns:
        df_scored["symbol_ratio"] = df_scored["text_clean"].apply(
            lambda t: sum(1 for c in str(t) if not c.isalnum() and not c.isspace()) / max(len(str(t)), 1)
        )

    # Predict probabilities; handle single-class edge case gracefully
    if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
        probs = pipeline.predict_proba(df_scored)
        # Class 1 probability if 2 classes, else 1.0
        if probs.shape[1] > 1:
            scores = probs[:, 1]
        else:
            scores = probs[:, 0]
    else:
        scores = pipeline.decision_function(df_scored)

    df_scored["quality_score"] = np.round(scores, 4)
    return df_scored


def select_top_documents(df_scored: pd.DataFrame, config: L2Config) -> pd.DataFrame:
    """
    Filter scored documents to the top candidates based on score threshold or top_k_ratio.
    """
    # Sort by quality score descending
    sorted_df = df_scored.sort_values(by="quality_score", ascending=False).reset_index(drop=True)
    
    # Selection by threshold with top_k fallback
    passing = sorted_df[sorted_df["quality_score"] >= config.selection_threshold]
    
    min_keep = max(1, int(len(sorted_df) * config.top_k_ratio))
    if len(passing) < min_keep:
        selected_df = sorted_df.head(min_keep).copy()
    else:
        selected_df = passing.copy()

    selected_df["tier"] = "l2_selected"
    return selected_df
