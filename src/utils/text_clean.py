"""
Pure text cleaning utilities for Tier 1 (L1: Clean).

Part of the L0-L4 Tiered Data Management framework (arXiv:2602.09003).
Maps to paper methodology: deterministic text normalization and boilerplate removal
prior to model-driven selection.
Functions are pure — deterministic, with no side effects or I/O.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

# Boilerplate patterns to strip (navigation, cookie banners, etc.)
# Kept minimal and conservative for tiny demo.
BOILERPLATE_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"(?i)\b(cookie|privacy policy|terms of service|accept all|reject all)\b.*"),
    re.compile(r"(?i)\b(sign in|log in|register|subscribe|newsletter)\b.*"),
    re.compile(r"(?i)\b(home|about|contact|menu|navigation|skip to content)\b.*"),
    re.compile(r"(?i)\b(advertisement|sponsored|promoted)\b.*"),
    re.compile(r"(?i)\b(share this|follow us|social media)\b.*"),
]

# Whitespace normalization: collapse multiple spaces/newlines/tabs to single space
_WS_RE: Final[re.Pattern[str]] = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """
    Normalize text for L1 cleaning.

    Steps (per technical-domain.md: fix_encoding → normalize → remove_boilerplate):
    1. Unicode NFKC normalization (fixes encoding artifacts, composes chars)
    2. Collapse all whitespace (spaces, newlines, tabs) to single space
    3. Strip leading/trailing whitespace

    Does NOT lowercase — preserves original case per user requirement.

    Args:
        text: Raw input text.

    Returns:
        Normalized text string.
    """
    if not text:
        return ""

    # 1. Unicode NFKC normalization
    normalized = unicodedata.normalize("NFKC", text)

    # 2. Collapse whitespace
    normalized = _WS_RE.sub(" ", normalized)

    # 3. Strip
    return normalized.strip()


def remove_boilerplate(text: str) -> str:
    """
    Remove common boilerplate/navigation text from normalized text.

    Applies a small set of conservative regex patterns to strip
    cookie banners, nav menus, sign-in prompts, etc.

    Args:
        text: Already-normalized text (output of normalize_text).

    Returns:
        Text with boilerplate patterns removed.
    """
    if not text:
        return ""

    cleaned = text
    for pattern in BOILERPLATE_PATTERNS:
        cleaned = pattern.sub("", cleaned)

    # Re-collapse whitespace after removals
    cleaned = _WS_RE.sub(" ", cleaned)
    return cleaned.strip()


def clean_text(text: str) -> str:
    """
    Full L1 deterministic cleaning pipeline: normalize → remove_boilerplate.

    This is the main entry point for text cleaning at L1.
    Pure function — no side effects, no I/O.

    Args:
        text: Raw input text.

    Returns:
        Fully cleaned text ready for heuristic filtering.
    """
    return remove_boilerplate(normalize_text(text))