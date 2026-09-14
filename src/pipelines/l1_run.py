"""
Tier 1 (L1: Clean) Orchestration Module.

Part of the L0-L4 Tiered Data Management framework (arXiv:2602.09003).
Coordinates Phase 1: Ingests raw data (L0) from real web streaming or local substitute,
applies pure text cleaning (text_clean.py), runs heuristic filtering and exact
deduplication (l1_filter.py), and persists clean records to Parquet and JSONL.
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
from datasets import load_dataset
from tqdm import tqdm

from src.utils.text_clean import clean_text
from src.pipelines.l1_filter import (
    FilterConfig,
    FilterResult,
    compute_stats,
    hash_doc,
    passes_filters,
)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration."""
    path = Path(config_path)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_fineweb_streaming(dataset: str, split: str, n: int) -> list[dict[str, Any]]:
    """
    Stream first n records from FineWeb (or similar) dataset.

    Uses datasets library streaming to avoid large downloads.
    """
    ds = load_dataset(dataset, split=split, streaming=True)
    records = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        records.append({
            "id": row.get("id", f"fineweb_{i}"),
            "text": row.get("text", ""),
            "url": row.get("url", ""),
            "source": "fineweb_sample10BT",
            "dump": row.get("dump", ""),
        })
    return records


def load_ultra_fineweb_streaming(dataset: str, split: str, n: int) -> list[dict[str, Any]]:
    """
    Stream first n records from Ultra-FineWeb dataset (paper-aligned).

    Falls back to FineWeb if this fails.
    """
    ds = load_dataset(dataset, split=split, streaming=True)
    records = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        records.append({
            "id": row.get("id", f"ultrafineweb_{i}"),
            "text": row.get("text", ""),
            "url": row.get("url", ""),
            "source": "ultra_fineweb",
            "dump": row.get("dump", ""),
        })
    return records


def load_local_substitute(path: str | Path) -> list[dict[str, Any]]:
    """Load local substitute JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                row = json.loads(line)
                # Ensure required fields
                row.setdefault("id", f"local_substitute_{i}")
                row.setdefault("source", "local_substitute")
                row.setdefault("url", "")
                row.setdefault("dump", "")
                records.append(row)
    return records


def load_raw_data(config: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """
    Load raw data with fallback chain:
    1. Check if use_real_data is enabled and real_data_path exists & non-empty
    2. Try ultra_fineweb_streaming (if mode set)
    3. Try fineweb_streaming
    4. Fall back to local substitute

    Returns:
        (records, source_used)
    """
    input_cfg = config["input"]
    use_real_data = input_cfg.get("use_real_data", False)
    real_data_path = input_cfg.get("real_data_path")
    n = input_cfg.get("n", 200)
    fallback_path = input_cfg.get("fallback_path")

    # 1. Check real_data_path if use_real_data is True
    if use_real_data and real_data_path:
        raw_p = Path(real_data_path)
        p = raw_p if raw_p.is_absolute() else _PROJECT_ROOT / raw_p
        if p.exists() and p.is_file() and p.stat().st_size > 0:
            try:
                records = load_local_substitute(p)
                if records:
                    print(f"[L1] Loaded {len(records[:n])} real data records from {p.as_posix()}")
                    return records[:n], "openbmb/Ultra-FineWeb (real)"
            except Exception as e:
                print(f"[L1] Warning: Failed to read real data file at {p.as_posix()}: {e}")
        else:
            print(f"[L1] Warning: Real data file '{real_data_path}' not found or empty.")
            print("[L1] Automatically falling back to local substitute data...")

    mode = input_cfg.get("mode", "local_substitute")

    # 2. Try Ultra-FineWeb first if requested
    if mode == "ultra_fineweb_streaming":
        try:
            dataset = input_cfg.get("dataset", "openbmb/Ultra-FineWeb")
            split = input_cfg.get("split", "train")
            records = load_ultra_fineweb_streaming(dataset, split, n)
            if records:
                return records, "ultra_fineweb_streaming"
        except Exception as e:
            print(f"[L1] Ultra-FineWeb streaming failed: {e}. Trying FineWeb...")

    # 3. Try FineWeb
    if mode in ("fineweb_streaming", "ultra_fineweb_streaming"):
        try:
            dataset = input_cfg.get("dataset", "HuggingFaceFW/fineweb")
            split = input_cfg.get("split", "sample-10BT")
            records = load_fineweb_streaming(dataset, split, n)
            if records:
                return records, "fineweb_streaming"
        except Exception as e:
            print(f"[L1] FineWeb streaming failed: {e}. Using local substitute...")

    # 4. Fallback to local substitute
    if fallback_path:
        raw_fb = Path(fallback_path)
        p_fb = raw_fb if raw_fb.is_absolute() else _PROJECT_ROOT / raw_fb
        if p_fb.exists():
            records = load_local_substitute(p_fb)
            print(f"[L1] Using local substitute data: {p_fb.as_posix()}")
            return records[:n], "local_substitute"

    raise RuntimeError("No data source available and no fallback file found.")


def run_l1(config_path: str | Path) -> dict[str, Any]:
    """
    Execute L1 pipeline: load → clean → filter → dedupe → save.

    Returns stats dict for CLI reporting.
    """
    config = load_config(config_path)
    filter_cfg = FilterConfig(**config["filters"])
    output_cfg = config["output"]

    # Load raw data
    raw_records, source_used = load_raw_data(config)
    input_count = len(raw_records)

    # Process each record
    seen_hashes: set[str] = set()
    output_records: list[dict[str, Any]] = []
    filter_reasons: dict[str, int] = {}
    duplicate_count = 0

    for row in tqdm(raw_records, desc="L1 processing", unit="doc"):
        raw_text = row.get("text", "")
        doc_id = row.get("id", "")
        url = row.get("url", "")
        source = row.get("source", source_used)

        # Clean
        cleaned = clean_text(raw_text)

        # Filter
        filter_result: FilterResult = passes_filters(cleaned, filter_cfg)
        filter_reasons[filter_result.reason] = filter_reasons.get(filter_result.reason, 0) + 1

        if not filter_result.passes:
            continue

        # Deduplicate (exact hash)
        text_hash = hash_doc(cleaned)
        if text_hash in seen_hashes:
            duplicate_count += 1
            filter_reasons["duplicate"] = filter_reasons.get("duplicate", 0) + 1
            continue

        seen_hashes.add(text_hash)

        # Compute stats
        stats = compute_stats(cleaned)

        # Build output record
        output_records.append({
            "id": doc_id,
            "text_clean": cleaned,
            "text_hash": text_hash,
            "char_len": stats["char_len"],
            "word_count": stats["word_count"],
            "alpha_ratio": stats["alpha_ratio"],
            "symbol_ratio": stats["symbol_ratio"],
            "filter_reason": filter_result.reason,
            "source": source,
            "url": url,
        })

    output_count = len(output_records)
    removed_count = input_count - output_count - duplicate_count

    # Save outputs
    raw_out = Path(output_cfg["dir"])
    output_dir = raw_out if raw_out.is_absolute() else _PROJECT_ROOT / raw_out
    output_dir.mkdir(parents=True, exist_ok=True)

    basename = output_cfg["basename"]
    formats = output_cfg.get("formats", ["parquet", "jsonl"])

    df = pd.DataFrame(output_records)

    if "parquet" in formats:
        parquet_path = output_dir / f"{basename}.parquet"
        df.to_parquet(parquet_path, index=False)

    if "jsonl" in formats:
        jsonl_path = output_dir / f"{basename}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for record in output_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "input_count": input_count,
        "output_count": output_count,
        "removed_count": removed_count,
        "duplicate_count": duplicate_count,
        "filter_reasons": filter_reasons,
        "output_path_parquet": (output_dir / f"{basename}.parquet").as_posix() if "parquet" in formats else None,
        "output_path_jsonl": (output_dir / f"{basename}.jsonl").as_posix() if "jsonl" in formats else None,
        "source_used": source_used,
    }