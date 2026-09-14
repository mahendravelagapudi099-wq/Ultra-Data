"""
Tier 4 (L4: Organized) Knowledge Export Orchestration Module.

Part of the L0-L4 Tiered Data Management framework (arXiv:2602.09003).
Coordinates Optional Phase 3.5: Validates synthesized records (L3) against
structural quality gates (text length, QA formatting, chapter headers) and exports
production-grade structured knowledge units with provenance and UTC timestamps.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration."""
    path = Path(config_path)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_l3_record(record: dict[str, Any]) -> tuple[bool, str]:
    """
    Validate a synthesized L3 record against structural quality gates.

    Checks:
    1. Refined text is non-trivial (min 80 chars, min 15 words)
    2. Q&A pair is properly formed (question ends with ?, answer has min 10 words)
    3. Textbook explanation contains required header structure
    """
    text_refined = str(record.get("text_refined", "")).strip()
    qa_question = str(record.get("qa_question", "")).strip()
    qa_answer = str(record.get("qa_answer", "")).strip()
    textbook = str(record.get("textbook_explanation", "")).strip()

    if len(text_refined) < 80 or len(text_refined.split()) < 15:
        return False, "refined_text_too_short"

    if not qa_question or not qa_question.endswith("?"):
        return False, "malformed_question"

    if len(qa_answer.split()) < 10:
        return False, "answer_too_short"

    if "# Chapter:" not in textbook or "## " not in textbook:
        return False, "missing_textbook_structure"

    return True, "valid"


def run_l4_export(config_path: str | Path) -> dict[str, Any]:
    """
    Execute Phase 3.5 (L4 Knowledge Export) pipeline.

    1. Load L3 refined records
    2. Validate structural constraints
    3. Format metadata fields (id, source, tier, generated_by, timestamp, validation_status)
    4. Save organized dataset to data/l4_organized/
    """
    config = load_config(config_path)
    input_path = Path(config["input"]["path"])
    if not input_path.is_absolute():
        input_path = _PROJECT_ROOT / input_path
    output_cfg = config["output"]

    if not input_path.exists():
        raise FileNotFoundError(
            f"L3 input file not found: {input_path.as_posix()}. "
            "Run Phase 3 first via 'python scripts/run_l3.py' to generate it."
        )

    if input_path.suffix == ".parquet":
        df_l3 = pd.read_parquet(input_path)
    else:
        records = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        df_l3 = pd.DataFrame(records)

    input_count = len(df_l3)
    if input_count == 0:
        raise ValueError("L3 input dataset is empty. Run Phase 3 first via 'python scripts/run_l3.py'.")

    now_iso = datetime.now(timezone.utc).isoformat()
    valid_records: list[dict[str, Any]] = []
    invalid_records: list[dict[str, Any]] = []

    for _, row in df_l3.iterrows():
        rec = row.to_dict()
        is_valid, reason = validate_l3_record(rec)

        structured_record = {
            "id": rec.get("id", ""),
            "source": rec.get("source", "l3_refined"),
            "url": rec.get("url", ""),
            "tier": "l4_organized",
            "generated_by": rec.get("generated_by", "mock_llm"),
            "timestamp": now_iso,
            "validation_status": reason,
            "primary_topic": rec.get("primary_topic", ""),
            "text_refined": rec.get("text_refined", ""),
            "qa_question": rec.get("qa_question", ""),
            "qa_answer": rec.get("qa_answer", ""),
            "textbook_explanation": rec.get("textbook_explanation", ""),
        }

        if is_valid:
            valid_records.append(structured_record)
        else:
            invalid_records.append(structured_record)

    output_dir = Path(output_cfg.get("dir", "data/l4_organized"))
    if not output_dir.is_absolute():
        output_dir = _PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = output_cfg.get("basename", "l4_organized")
    formats = output_cfg.get("formats", ["parquet", "jsonl"])

    parquet_path = (output_dir / f"{basename}.parquet").as_posix()
    jsonl_path = (output_dir / f"{basename}.jsonl").as_posix()

    df_valid = pd.DataFrame(valid_records)

    if "parquet" in formats and len(df_valid) > 0:
        df_valid.to_parquet(parquet_path, index=False)

    if "jsonl" in formats and len(df_valid) > 0:
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in valid_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    valid_count = len(valid_records)
    invalid_count = len(invalid_records)
    pass_rate = round(valid_count / max(input_count, 1), 4)

    return {
        "input_count": input_count,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "validation_pass_rate": pass_rate,
        "output_path_parquet": parquet_path if "parquet" in formats else None,
        "output_path_jsonl": jsonl_path if "jsonl" in formats else None,
    }
