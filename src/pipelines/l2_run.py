"""
Tier 2 (L2: Selection) Orchestration Module.

Part of the L0-L4 Tiered Data Management framework (arXiv:2602.09003).
Coordinates Phase 2: Loads cleaned data (L1), generates weak supervision labels,
trains a lightweight TF-IDF + numeric feature selector (strictly on train split),
scores documents, and saves high-value educational tokens (L2) to Parquet and JSONL.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import yaml

from src.pipelines.l2_select import (
    L2Config,
    assign_weak_labels,
    score_documents,
    select_top_documents,
    train_l2_selector,
)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration."""
    path = Path(config_path)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_l1_data(input_path: str | Path) -> pd.DataFrame:
    """Load L1 filtered records from Parquet or JSONL file."""
    path = Path(input_path)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(
            f"L1 input file not found: {path.as_posix()}. "
            "Run Phase 1 first via 'python scripts/run_phase1.py' or 'python scripts/run_l1.py' to generate it."
        )

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix == ".jsonl":
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return pd.DataFrame(records)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Expected .parquet or .jsonl")


def run_l2(config_path: str | Path) -> dict[str, Any]:
    """
    Execute Phase 2 (L2 Selection) pipeline.

    1. Load L1 filtered records
    2. Assign weak demo supervision labels
    3. Train lightweight selector (preprocessing fitted ONLY on train split)
    4. Score all records and save to data/l2_scores/
    5. Filter top records and save to data/l2_selected/
    
    Returns:
        Summary metrics dictionary.
    """
    config = load_config(config_path)
    raw_inp = Path(config["input"]["path"])
    input_path = raw_inp if raw_inp.is_absolute() else _PROJECT_ROOT / raw_inp
    
    l2_cfg = L2Config(
        selection_threshold=config.get("selection", {}).get("threshold", 0.5),
        top_k_ratio=config.get("selection", {}).get("top_k_ratio", 0.5),
        test_size=config.get("model", {}).get("test_size", 0.3),
        random_state=config.get("model", {}).get("random_state", 42),
        max_tfidf_features=config.get("model", {}).get("max_tfidf_features", 300),
    )

    df_l1 = load_l1_data(input_path)
    input_count = len(df_l1)
    if input_count == 0:
        raise ValueError("L1 input dataset is empty. Run Phase 1 first via 'python scripts/run_phase1.py'.")

    # Generate weak supervision labels
    weak_labels = assign_weak_labels(df_l1)

    # Train selector (train/test split, preprocessor fit on train split only)
    pipeline, model_metrics = train_l2_selector(df_l1, weak_labels, l2_cfg)

    # Score all records
    df_scored = score_documents(pipeline, df_l1)

    # Select top documents
    df_selected = select_top_documents(df_scored, l2_cfg)
    selected_count = len(df_selected)

    # Output paths setup
    raw_scores_dir = Path(config["output"].get("scores_dir", "data/l2_scores"))
    scores_dir = raw_scores_dir if raw_scores_dir.is_absolute() else _PROJECT_ROOT / raw_scores_dir
    raw_selected_dir = Path(config["output"].get("selected_dir", "data/l2_selected"))
    selected_dir = raw_selected_dir if raw_selected_dir.is_absolute() else _PROJECT_ROOT / raw_selected_dir
    basename = config["output"].get("basename", "l2_selected")
    formats = config["output"].get("formats", ["parquet", "jsonl"])

    scores_dir.mkdir(parents=True, exist_ok=True)
    selected_dir.mkdir(parents=True, exist_ok=True)

    scores_parquet = (scores_dir / f"{basename}_all_scored.parquet").as_posix()
    scores_jsonl = (scores_dir / f"{basename}_all_scored.jsonl").as_posix()
    selected_parquet = (selected_dir / f"{basename}.parquet").as_posix()
    selected_jsonl = (selected_dir / f"{basename}.jsonl").as_posix()

    if "parquet" in formats:
        df_scored.to_parquet(scores_parquet, index=False)
        df_selected.to_parquet(selected_parquet, index=False)

    if "jsonl" in formats:
        df_scored.to_json(scores_jsonl, orient="records", lines=True, force_ascii=False)
        df_selected.to_json(selected_jsonl, orient="records", lines=True, force_ascii=False)

    return {
        "input_count": input_count,
        "selected_count": selected_count,
        "selection_rate": round(selected_count / max(input_count, 1), 4),
        "mean_quality_score": round(float(df_scored["quality_score"].mean()), 4),
        "mean_selected_score": round(float(df_selected["quality_score"].mean()), 4),
        "train_accuracy": round(model_metrics["train_accuracy"], 4),
        "test_accuracy": round(model_metrics["test_accuracy"], 4),
        "output_scores_parquet": scores_parquet if "parquet" in formats else None,
        "output_scores_jsonl": scores_jsonl if "jsonl" in formats else None,
        "output_selected_parquet": selected_parquet if "parquet" in formats else None,
        "output_selected_jsonl": selected_jsonl if "jsonl" in formats else None,
    }
