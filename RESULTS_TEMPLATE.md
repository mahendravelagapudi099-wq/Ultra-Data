# Tiered Data Management: Results & Schema Specifications

This reference document defines the exact Parquet and JSONL schema specifications for each tier (L0 to L4) and records baseline demo metrics.

---

## 1. Production Data Schemas by Tier

### Tier 0: L0 Raw Web Data
- **File formats:** `.jsonl`
- **Location:** `data/l0_raw/l0_real_sample.jsonl` or `data/l0_raw/l0_expanded_SUBSTITUTE.jsonl`
- **Schema:**
| Column | Type | Description |
|---|---|---|
| `id` | `string` | Unique document identifier |
| `text` | `string` | Raw unnormalized web text |
| `url` | `string` | Source URL or origin metadata |
| `source` | `string` | Origin dataset name (`openbmb/Ultra-FineWeb` or `local_substitute`) |
| `dump` | `string` | Optional crawl batch or snapshot identifier |

```json
{
  "id": "real_0",
  "text": "Deep residual learning frameworks reformulate the layers...",
  "url": "https://arxiv.org/abs/1512.03385",
  "source": "openbmb/Ultra-FineWeb",
  "dump": "train"
}
```

---

### Tier 1: L1 Cleaned & Filtered Data
- **File formats:** `.parquet`, `.jsonl`
- **Location:** `data/l1_filtered/l1_expanded_SUBSTITUTE.parquet`
- **Schema:**
| Column | Type | Description |
|---|---|---|
| `id` | `string` | Document identifier |
| `text_clean` | `string` | NFKC-normalized, boilerplate-stripped text |
| `text_hash` | `string` | SHA-256 64-character hexadecimal hash for exact deduplication |
| `char_len` | `int64` | Total character count |
| `word_count` | `int64` | Word count using regex tokenization |
| `alpha_ratio` | `float64` | Ratio of alphabetic characters to total characters |
| `symbol_ratio` | `float64` | Ratio of non-alphanumeric, non-space characters |
| `filter_reason` | `string` | Validation status (`pass`) |
| `source` | `string` | Data source identifier |
| `url` | `string` | Source URL |

```json
{
  "id": "real_0",
  "text_clean": "Deep residual learning frameworks reformulate the layers...",
  "text_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "char_len": 312,
  "word_count": 45,
  "alpha_ratio": 0.8450,
  "symbol_ratio": 0.0210,
  "filter_reason": "pass",
  "source": "openbmb/Ultra-FineWeb",
  "url": "https://arxiv.org/abs/1512.03385"
}
```

---

### Tier 2: L2 Model-Selected Data
- **File formats:** `.parquet`, `.jsonl`
- **Location:** `data/l2_selected/l2_selected.parquet` (and `data/l2_scores/l2_selected_all_scored.parquet`)
- **Schema:**
| Column | Type | Description |
|---|---|---|
| `id` | `string` | Document identifier |
| `text_clean` | `string` | Cleaned text |
| `char_len` | `int64` | Character length |
| `word_count` | `int64` | Word count |
| `alpha_ratio` | `float64` | Alphabetic ratio |
| `symbol_ratio` | `float64` | Symbol ratio |
| `quality_score` | `float64` | Calibrated model probability score [0.0 - 1.0] |
| `tier` | `string` | Tier tag (`l2_selected`) |
| `source` | `string` | Data source identifier |
| `url` | `string` | Source URL |

```json
{
  "id": "real_0",
  "text_clean": "Deep residual learning frameworks reformulate the layers...",
  "char_len": 312,
  "word_count": 45,
  "alpha_ratio": 0.8450,
  "symbol_ratio": 0.0210,
  "quality_score": 0.7825,
  "tier": "l2_selected",
  "source": "openbmb/Ultra-FineWeb",
  "url": "https://arxiv.org/abs/1512.03385"
}
```

---

### Tier 3: L3 Refined & Synthesized Data
- **File formats:** `.parquet`, `.jsonl`
- **Location:** `data/l3_refined/l3_refined.parquet`
- **Schema:**
| Column | Type | Description |
|---|---|---|
| `id` | `string` | Document identifier |
| `source` | `string` | Origin source identifier |
| `url` | `string` | Source URL |
| `tier` | `string` | Tier tag (`l3_refined`) |
| `quality_score` | `float64` | Upstream L2 quality score |
| `text_clean` | `string` | Cleaned source text |
| `text_refined` | `string` | Refined article with structured overview header |
| `qa_question` | `string` | Synthesized conceptual question |
| `qa_answer` | `string` | Synthesized grounded answer |
| `textbook_explanation` | `string` | Formatted multi-section markdown chapter |
| `primary_topic` | `string` | Extracted core subject entity |
| `generated_by` | `string` | Synthesizer tag (`mock_llm`) |
| `model` | `string` | Generator model version |

```json
{
  "id": "real_0",
  "source": "openbmb/Ultra-FineWeb",
  "url": "https://arxiv.org/abs/1512.03385",
  "tier": "l3_refined",
  "quality_score": 0.7825,
  "text_clean": "Deep residual learning frameworks...",
  "text_refined": "### Overview: Deep residual learning\n\nDeep residual learning frameworks...",
  "qa_question": "What is the significance of Deep residual learning, and how does it operate?",
  "qa_answer": "Deep residual learning frameworks reformulate the layers...",
  "textbook_explanation": "# Chapter: Deep residual learning\n\n## 1. Introduction...",
  "primary_topic": "Deep residual learning",
  "generated_by": "mock_llm",
  "model": "mock-rule-synthesizer-v1"
}
```

---

### Tier 4: L4 Organized Structured Knowledge Units
- **File formats:** `.parquet`, `.jsonl`
- **Location:** `data/l4_organized/l4_organized.parquet`
- **Schema:**
| Column | Type | Description |
|---|---|---|
| `id` | `string` | Document identifier |
| `source` | `string` | Provenance source identifier |
| `url` | `string` | Source URL |
| `tier` | `string` | Tier tag (`l4_organized`) |
| `generated_by` | `string` | Synthesis origin (`mock_llm`) |
| `timestamp` | `string` | ISO-8601 UTC validation timestamp |
| `validation_status` | `string` | Structural quality gate outcome (`valid`) |
| `primary_topic` | `string` | Validated primary topic entity |
| `text_refined` | `string` | Validated refined text content |
| `qa_question` | `string` | Validated question string |
| `qa_answer` | `string` | Validated answer string |
| `textbook_explanation` | `string` | Validated textbook chapter content |

```json
{
  "id": "real_0",
  "source": "openbmb/Ultra-FineWeb",
  "url": "https://arxiv.org/abs/1512.03385",
  "tier": "l4_organized",
  "generated_by": "mock_llm",
  "timestamp": "2026-09-14T10:45:00.000000+00:00",
  "validation_status": "valid",
  "primary_topic": "Deep residual learning",
  "text_refined": "### Overview: Deep residual learning\n\n...",
  "qa_question": "What is the significance of Deep residual learning, and how does it operate?",
  "qa_answer": "Deep residual learning frameworks reformulate...",
  "textbook_explanation": "# Chapter: Deep residual learning\n\n..."
}
```

---

## 2. Baseline Evaluation Results Template

| Tier | Description | Doc Count | Avg Chars | Avg Words | Alpha Ratio | Symbol Ratio | Duplicates | Retention Rate |
|---|---|---|---|---|---|---|---|---|
| **L0_Raw** | Unfiltered raw web corpus | 145 | 181.9 | 24.0 | 0.7996 | 0.0762 | 10 | 100.0% |
| **L1_Filtered** | Cleaned & exact deduped | 33 | 282.8 | 38.2 | 0.8428 | 0.0248 | 0 | 22.8% |
| **L2_Selected** | Informative model tokens | 16 | 317.8 | 43.8 | 0.8377 | 0.0251 | 0 | 48.5% |
| **L3_Refined** | Mock LLM synthesized | 16 | 317.8 | 43.8 | 0.8377 | 0.0251 | 0 | 100.0% |
| **L4_Organized** | Validated knowledge units | 16 | 366.7 | 49.4 | 0.8251 | 0.0349 | 0 | 100.0% |

**Overall End-to-End Retention (L0 → L4):** `11.0%`
