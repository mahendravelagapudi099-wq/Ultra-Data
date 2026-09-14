# Tiered Data Management: Results & Schema Template

This template records the data schema, expected artifacts, and evaluation baseline for reproducing tiered data management pipelines.

---

## 1. Data Schema by Tier

### Tier 0: L0 Raw
File: `data/l0_raw/*.jsonl`
```json
{
  "id": "local_substitute_l1_0",
  "text": "Raw unstructured text...",
  "url": "https://example.com/page",
  "source": "local_substitute_l1",
  "dump": "generated"
}
```

### Tier 1: L1 Filtered
Files: `data/l1_filtered/*.parquet`, `*.jsonl`
```json
{
  "id": "local_substitute_l1_0",
  "text_clean": "Normalized, boilerplate-free text...",
  "text_hash": "sha256_hex_digest_64_chars",
  "char_len": 312,
  "word_count": 45,
  "alpha_ratio": 0.845,
  "symbol_ratio": 0.021,
  "filter_reason": "pass",
  "source": "local_substitute_l1",
  "url": "https://example.com/page"
}
```

### Tier 2: L2 Selected & Scored
Files: `data/l2_scores/*.parquet`, `data/l2_selected/*.parquet`
```json
{
  "id": "local_substitute_l1_0",
  "text_clean": "...",
  "char_len": 312,
  "word_count": 45,
  "alpha_ratio": 0.845,
  "symbol_ratio": 0.021,
  "quality_score": 0.7825,
  "tier": "l2_selected"
}
```

### Tier 3: L3 Refined & Synthesized
Files: `data/l3_refined/*.parquet`, `*.jsonl`
```json
{
  "id": "local_substitute_l1_0",
  "source": "local_substitute_l1",
  "tier": "l3_refined",
  "quality_score": 0.7825,
  "text_clean": "...",
  "text_refined": "### Overview: Topic\n\nStructured text...",
  "qa_question": "What is the significance of Topic, and how does it operate?",
  "qa_answer": "Topic is...",
  "textbook_explanation": "# Chapter: Topic\n\n## 1. Introduction...",
  "primary_topic": "Topic",
  "generated_by": "mock_llm",
  "model": "mock-rule-synthesizer-v1"
}
```

### Tier 4: L4 Organized Knowledge Export
Files: `data/l4_organized/*.parquet`, `*.jsonl`
```json
{
  "id": "local_substitute_l1_0",
  "source": "local_substitute_l1",
  "url": "https://example.com/page",
  "tier": "l4_organized",
  "generated_by": "mock_llm",
  "timestamp": "2026-09-14T09:55:00.000000+00:00",
  "validation_status": "valid",
  "primary_topic": "Topic",
  "text_refined": "...",
  "qa_question": "...",
  "qa_answer": "...",
  "textbook_explanation": "..."
}
```

---

## 2. Evaluation Results Baseline Template

| Run Date | L0 Raw Count | L1 Filtered Count | L2 Selected Count | L3 Refined Count | L4 Valid Units | L1 Retention | L2 Selection Rate | L4 Pass Rate | End-to-End Retention |
|---|---|---|---|---|---|---|---|---|---|
| *YYYY-MM-DD* | *145* | *33* | *16* | *16* | *16* | *22.8%* | *48.5%* | *100.0%* | *11.0%* |

---

## 3. Notes & Disclaimer

> [!IMPORTANT]
> - This baseline was obtained using the local expanded substitute generator (`scripts/generate_l0_expanded.py`) and starter heuristics.
> - When scaling up to live web dumps (e.g. FineWeb 10BT or Ultra-FineWeb), update YAML configs and log the resulting counts in this table.
