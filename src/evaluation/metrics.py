"""
Evaluation and comparison metrics for tiered data pipelines.

Supports Phase 4 (Evaluation & Comparison) of the L0-L4 Tiered Data Management
framework (arXiv:2602.09003). Computes tier-by-tier metrics:
- Document count
- Average character length and word count
- Alphabetic character ratio and symbol ratio
- Exact duplicate counts
- Tier-to-tier retention, selection, and validation rates
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def compute_doc_stats(text: str) -> dict[str, float]:
    """Compute basic character, word, alpha, and symbol statistics on a string."""
    text_str = str(text) if text is not None else ""
    char_len = len(text_str)
    words = text_str.split()
    word_count = len(words)
    alpha_count = sum(1 for c in text_str if c.isalpha())
    symbol_count = sum(1 for c in text_str if not c.isalnum() and not c.isspace())

    return {
        "char_len": float(char_len),
        "word_count": float(word_count),
        "alpha_ratio": alpha_count / max(char_len, 1),
        "symbol_ratio": symbol_count / max(char_len, 1),
    }


def compute_tier_metrics(df: pd.DataFrame | None, tier_name: str, text_col: str = "text") -> dict[str, Any]:
    """
    Compute comprehensive metrics for a given tier dataframe.

    Pure function — deterministic and side-effect free. Handles empty or None dataframes gracefully.
    """
    if df is None or len(df) == 0:
        return {
            "tier": tier_name,
            "doc_count": 0,
            "avg_char_len": 0.0,
            "avg_word_count": 0.0,
            "avg_alpha_ratio": 0.0,
            "avg_symbol_ratio": 0.0,
            "duplicate_count": 0,
            "unique_doc_count": 0,
        }

    # Identify primary text column
    actual_col = text_col
    if actual_col not in df.columns:
        candidates = ["text_clean", "text_refined", "text", "content"]
        for cand in candidates:
            if cand in df.columns:
                actual_col = cand
                break

    stats_list = [compute_doc_stats(t) for t in df[actual_col]]
    stats_df = pd.DataFrame(stats_list)

    # Hash for duplicates
    hashes = [hashlib.sha256(str(t).encode("utf-8")).hexdigest() for t in df[actual_col]]
    unique_hashes = len(set(hashes))
    duplicate_count = len(hashes) - unique_hashes

    return {
        "tier": tier_name,
        "doc_count": int(len(df)),
        "avg_char_len": round(float(stats_df["char_len"].mean()), 1),
        "avg_word_count": round(float(stats_df["word_count"].mean()), 1),
        "avg_alpha_ratio": round(float(stats_df["alpha_ratio"].mean()), 4),
        "avg_symbol_ratio": round(float(stats_df["symbol_ratio"].mean()), 4),
        "duplicate_count": int(duplicate_count),
        "unique_doc_count": int(unique_hashes),
    }


def generate_markdown_report(
    tier_metrics: list[dict[str, Any]],
    rates: dict[str, float],
    source_info: str = "Local Substitute Data (or openbmb/Ultra-FineWeb real sample)",
) -> str:
    """
    Generate GitHub-flavored Markdown report comparing tiers across the pipeline.

    Includes disclaimer banner, prominent data source indicator, and formatted metric tables.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# Pipeline Evaluation & Tier Comparison Report",
        "",
        "> [!NOTE]",
        "> **Notice:** This report reflects a small-scale, offline demo reproduction of the methodology from",
        "> *\"Data Science and Technology Towards AGI Part I: Tiered Data Management\"* (arXiv:2602.09003).",
        "> Heuristics, thresholds, and mock LLM synthesizers are lightweight starter implementations and do not claim paper-scale results.",
        "",
        f"**Generated:** {timestamp}  ",
        f"**Data Source:** `{source_info}`  ",
        "**Environment:** Colab / Linux / Local agnostic  ",
        "",
        "## 1. Tier-by-Tier Quality Progression",
        "",
        "| Tier | Description | Doc Count | Avg Chars | Avg Words | Alpha Ratio | Symbol Ratio | Duplicates |",
        "|---|---|---|---|---|---|---|---|",
    ]

    tier_descriptions = {
        "L0_Raw": "Unfiltered raw web corpus (real sample or substitute)",
        "L1_Filtered": "Heuristic filtered & exact deduped (clean text)",
        "L2_Selected": "Model-selected informative tokens (educational)",
        "L3_Refined": "Mock LLM refined, Q&A pairs & textbook chapters",
        "L4_Organized": "Validated structured knowledge units with provenance",
    }

    for m in tier_metrics:
        name = m.get("tier", "Unknown")
        desc = tier_descriptions.get(name, "Data tier")
        doc_count = m.get("doc_count", 0)
        avg_chars = m.get("avg_char_len", 0.0)
        avg_words = m.get("avg_word_count", 0.0)
        alpha_ratio = m.get("avg_alpha_ratio", 0.0)
        symbol_ratio = m.get("avg_symbol_ratio", 0.0)
        dup_count = m.get("duplicate_count", 0)

        lines.append(
            f"| `{name}` | {desc} | {doc_count:,} | {avg_chars:,.1f} | "
            f"{avg_words:,.1f} | {alpha_ratio:.4f} | {symbol_ratio:.4f} | {dup_count:,} |"
        )

    lines.extend([
        "",
        "## 2. Retention & Transition Rates",
        "",
        f"- **L0 → L1 Filter Retention Rate:** `{rates.get('l1_retention', 0.0) * 100:.1f}%`",
        f"- **L1 → L2 Model Selection Rate:** `{rates.get('l2_selection_rate', 0.0) * 100:.1f}%`",
        f"- **L2 → L3 Refinement Rate:** `{rates.get('l3_refinement_rate', 0.0) * 100:.1f}%`",
        f"- **L3 → L4 Validation Pass Rate:** `{rates.get('l4_pass_rate', 0.0) * 100:.1f}%`",
        f"- **End-to-End Retention (L0 → L4):** `{rates.get('end_to_end_rate', 0.0) * 100:.1f}%`",
        "",
        "## 3. Artifact Locations",
        "",
        "- `data/l0_raw/l0_real_sample.jsonl` (or `data/l0_raw/l0_expanded_SUBSTITUTE.jsonl`)",
        "- `data/l1_filtered/l1_expanded_SUBSTITUTE.parquet`",
        "- `data/l2_selected/l2_selected.parquet`",
        "- `data/l3_refined/l3_refined.parquet`",
        "- `data/l4_organized/l4_organized.parquet`",
        "- `reports/tier_stats.csv`",
        "- `reports/tier_stats.json`",
        "",
    ])

    return "\n".join(lines)
