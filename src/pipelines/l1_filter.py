"""
Pure filter and deduplication utilities for L1 (Tier 1: Clean).

Maps to paper methodology: heuristic filtering + exact deduplication.
Functions are pure — same input = same output, no side effects.
"""

import hashlib
import re
from dataclasses import dataclass
from typing import Final

# Heuristic filter thresholds (config-driven via l1_run.py)
# These are starter values — NOT cited from paper.pdf

# Regex for word tokenization (simple, Unicode-aware)
_WORD_RE: Final[re.Pattern[str]] = re.compile(r"\b\w+\b", re.UNICODE)

# Symbol characters (non-alphanumeric, non-whitespace)
_SYMBOL_RE: Final[re.Pattern[str]] = re.compile(r"[^\w\s]", re.UNICODE)


@dataclass(frozen=True, slots=True)
class FilterConfig:
    """Immutable filter thresholds for L1 heuristic filtering."""
    min_chars: int
    min_words: int
    min_alpha_ratio: float
    max_symbol_ratio: float


@dataclass(frozen=True, slots=True)
class FilterResult:
    """Result of heuristic filtering."""
    passes: bool
    reason: str  # "pass" | "too_short" | "too_few_words" | "low_alpha" | "high_symbol"


def count_words(text: str) -> int:
    """Count words using simple Unicode-aware regex."""
    return len(_WORD_RE.findall(text))


def count_alpha_chars(text: str) -> int:
    """Count alphabetic characters (Unicode letters)."""
    return sum(1 for ch in text if ch.isalpha())


def count_symbol_chars(text: str) -> int:
    """Count symbol characters (non-alphanumeric, non-whitespace)."""
    return len(_SYMBOL_RE.findall(text))


def passes_filters(text: str, config: FilterConfig) -> FilterResult:
    """
    Apply L1 heuristic filters to cleaned text.

    Filters (in order):
    1. Minimum character count
    2. Minimum word count
    3. Minimum alphabetic character ratio
    4. Maximum symbol character ratio

    Args:
        text: Cleaned text (output of clean_text).
        config: FilterConfig with thresholds.

    Returns:
        FilterResult with passes flag and reason.
    """
    if not text:
        return FilterResult(passes=False, reason="empty")

    char_len = len(text)
    if char_len < config.min_chars:
        return FilterResult(passes=False, reason="too_short")

    word_count = count_words(text)
    if word_count < config.min_words:
        return FilterResult(passes=False, reason="too_few_words")

    alpha_count = count_alpha_chars(text)
    alpha_ratio = alpha_count / char_len if char_len > 0 else 0.0
    if alpha_ratio < config.min_alpha_ratio:
        return FilterResult(passes=False, reason="low_alpha")

    symbol_count = count_symbol_chars(text)
    symbol_ratio = symbol_count / char_len if char_len > 0 else 0.0
    if symbol_ratio > config.max_symbol_ratio:
        return FilterResult(passes=False, reason="high_symbol")

    return FilterResult(passes=True, reason="pass")


def hash_doc(text: str) -> str:
    """
    Compute SHA-256 hash of text for exact deduplication.

    Pure function — deterministic, no side effects.
    Used to identify exact duplicate documents at L1.

    Args:
        text: Cleaned text to hash.

    Returns:
        Hexadecimal SHA-256 digest (64 chars).
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_stats(text: str) -> dict:
    """
    Compute basic text statistics for L1 output record.

    Pure function — no side effects.

    Args:
        text: Cleaned text.

    Returns:
        Dict with char_len, word_count, alpha_ratio, symbol_ratio.
    """
    char_len = len(text)
    word_count = count_words(text)
    alpha_count = count_alpha_chars(text)
    symbol_count = count_symbol_chars(text)

    return {
        "char_len": char_len,
        "word_count": word_count,
        "alpha_ratio": alpha_count / char_len if char_len > 0 else 0.0,
        "symbol_ratio": symbol_count / char_len if char_len > 0 else 0.0,
    }