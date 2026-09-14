"""
Tier 3 (L3: Refinement) Orchestration Module.

Part of the L0-L4 Tiered Data Management framework (arXiv:2602.09003).
Coordinates Phase 3: Transforms high-value selected web text (L2) into synthetic
educational artifacts (refined text, Q&A pairs, textbook chapters) via deterministic
offline MockLLMProvider. All outputs are explicitly tagged with 'mock_llm'.
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
from tqdm import tqdm

from src.utils.llm_provider import MockLLMProvider


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration."""
    path = Path(config_path)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_l2_data(input_path: str | Path) -> pd.DataFrame:
    """Load L2 selected records from Parquet or JSONL."""
    path = Path(input_path)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(
            f"L2 input file not found: {path.as_posix()}. "
            "Run Phase 2 first via 'python scripts/run_l2.py' to generate it."
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
        raise ValueError(f"Unsupported file format: {path.suffix}")


def run_l3(config_path: str | Path) -> dict[str, Any]:
    """
    Execute Phase 3 (L3 LLM Refinement & Synthesis) pipeline.
    
    1. Load L2 selected documents
    2. Pass through MockLLMProvider
    3. Output refined records with Q&A, textbook explanations, and metadata
    4. Save to data/l3_refined/
    
    Returns:
        Summary statistics dictionary.
    """
    config = load_config(config_path)
    raw_inp = Path(config["input"]["path"])
    input_path = raw_inp if raw_inp.is_absolute() else _PROJECT_ROOT / raw_inp
    output_cfg = config["output"]

    df_l2 = load_l2_data(input_path)
    input_count = len(df_l2)
    if input_count == 0:
        raise ValueError("L2 input dataset is empty. Run Phase 2 first via 'python scripts/run_l2.py'.")

    from src.utils.llm_provider import LLMProvider, MockLLMProvider

    provider_type = str(config.get("provider", "mock")).lower().strip()
    if provider_type == "gemini":
        from src.utils.llm_provider import GeminiLLMProvider
        gemini_model = config.get("gemini_model", "gemini-1.5-flash")
        provider: LLMProvider = GeminiLLMProvider(model_name=gemini_model)
    else:
        provider = MockLLMProvider()

    refined_records: list[dict[str, Any]] = []

    for _, row in tqdm(df_l2.iterrows(), total=input_count, desc="L3 Refinement", unit="doc"):
        doc_id = str(row.get("id", ""))
        text = str(row.get("text_clean", ""))
        source = str(row.get("source", "l2_selected"))
        url = str(row.get("url", ""))
        quality_score = float(row.get("quality_score", 1.0))

        # Synthesize via deterministic mock LLM
        synthesis = provider.refine_document(text, doc_id=doc_id)

        record = {
            "id": doc_id,
            "source": source,
            "url": url,
            "tier": "l3_refined",
            "quality_score": quality_score,
            "text_clean": text,
            "text_refined": synthesis["text_refined"],
            "qa_question": synthesis["qa_pair"]["question"],
            "qa_answer": synthesis["qa_pair"]["answer"],
            "textbook_explanation": synthesis["textbook_explanation"],
            "primary_topic": synthesis["primary_topic"],
            "generated_by": synthesis["generated_by"],
            "model": synthesis["model"],
        }
        refined_records.append(record)

    raw_out = Path(output_cfg.get("dir", "data/l3_refined"))
    output_dir = raw_out if raw_out.is_absolute() else _PROJECT_ROOT / raw_out
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = output_cfg.get("basename", "l3_refined")
    formats = output_cfg.get("formats", ["parquet", "jsonl"])

    df_refined = pd.DataFrame(refined_records)

    parquet_path = (output_dir / f"{basename}.parquet").as_posix()
    jsonl_path = (output_dir / f"{basename}.jsonl").as_posix()

    if "parquet" in formats:
        df_refined.to_parquet(parquet_path, index=False)

    if "jsonl" in formats:
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in refined_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {
        "input_count": input_count,
        "refined_count": len(refined_records),
        "generator": f"{provider.__class__.__name__} ({getattr(provider, 'model_name', 'default')})",
        "output_path_parquet": parquet_path if "parquet" in formats else None,
        "output_path_jsonl": jsonl_path if "jsonl" in formats else None,
    }
