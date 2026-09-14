# Tiered Data Management for LLM Pre-Training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mahendravelagapudi099-wq/Ultra-Data/blob/main/colab/run_all.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Paper](https://img.shields.io/badge/arXiv-2602.09003-b31b1b.svg)](https://arxiv.org/abs/2602.09003)

An end-to-end, modular data engineering pipeline reproducing tiered data management for large language model pre-training from raw web crawl to structured knowledge units.

Based on the research paper:
> **"Data Science and Technology Towards AGI Part I: Tiered Data Management"**  
> *arXiv:2602.09003* — [Read the Paper](https://arxiv.org/abs/2602.09003)

---

## Architecture: The L0 → L4 Tiered Progression

The pipeline organizes large-scale pre-training data into five progressive quality and density tiers:

```
┌───────────────────────────────────────────────────────────────────────────┐
│                           L0: Raw Web Data                                │
│   Unfiltered web crawl (openbmb/Ultra-FineWeb streaming or local mock)    │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼ [Phase 1: Clean & Filter]
┌───────────────────────────────────────────────────────────────────────────┐
│                           L1: Cleaned Data                                │
│   NFKC normalization, boilerplate stripping, heuristic filters, SHA-256   │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼ [Phase 2: Model-Driven Selection]
┌───────────────────────────────────────────────────────────────────────────┐
│                          L2: Selected Tokens                              │
│   Weak demo supervision, TF-IDF + text statistics, logistic scoring       │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼ [Phase 3: LLM Refinement & Synthesis]
┌───────────────────────────────────────────────────────────────────────────┐
│                          L3: Refined Knowledge                            │
│   Deterministic Mock LLM synthesis: refined text, Q&A, textbook chapters  │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼ [Phase 3.5: Structural Quality Gate]
┌───────────────────────────────────────────────────────────────────────────┐
│                         L4: Organized Units                               │
│   Structural validation, provenance tagging, ISO-8601 audit timestamps    │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼ [Phase 4: Evaluation & Reporting]
┌───────────────────────────────────────────────────────────────────────────┐
│                          Cross-Tier Reports                               │
│   Transition rates, retention statistics, Markdown / CSV / JSON reports   │
└───────────────────────────────────────────────────────────────────────────┘
```

### Tier Matrix

| Tier | Name | Target Format | Primary Operations | Key Output Columns |
|---|---|---|---|---|
| **L0** | Raw Data | JSONL | Web streaming, local substitute generation | `id`, `text`, `url`, `source` |
| **L1** | Cleaned | Parquet / JSONL | NFKC normalization, boilerplate removal, heuristic threshold filtering, exact deduplication | `id`, `text_clean`, `text_hash`, `char_len`, `word_count`, `alpha_ratio`, `symbol_ratio` |
| **L2** | Selected | Parquet / JSONL | Weak demo labeling, TF-IDF n-grams + numeric feature scaling, calibrated logistic regression scoring | `id`, `text_clean`, `quality_score`, `tier` |
| **L3** | Refined | Parquet / JSONL | Deterministic rule-based LLM refinement into synthetic educational formats | `id`, `text_refined`, `qa_question`, `qa_answer`, `textbook_explanation`, `generated_by` |
| **L4** | Organized | Parquet / JSONL | Structural quality gate validation, provenance formatting, ISO UTC timestamps | `id`, `source`, `url`, `tier`, `generated_by`, `timestamp`, `validation_status`, `text_refined`, `qa_question`, `qa_answer`, `textbook_explanation` |

---

## Tech Stack

- **Language:** Python 3.10+ (modern type hints, `from __future__ import annotations`, frozen dataclasses)
- **CLI Framework:** [Typer](https://typer.tiangolo.com/) & [Rich](https://rich.readthedocs.io/) for terminal formatting
- **Data Processing:** [Pandas](https://pandas.pydata.org/), [PyArrow](https://arrow.apache.org/docs/python/) (Parquet & JSONL)
- **Machine Learning:** [Scikit-learn](https://scikit-learn.org/) (TF-IDF vectorizer, numeric standard scaler, logistic regression)
- **Web Data Ingestion:** [Hugging Face Datasets](https://huggingface.co/docs/datasets/) (streaming mode)
- **Configuration:** [PyYAML](https://pyyaml.org/) (modular YAML configuration files)
- **Environment Support:** Google Colab, Linux, macOS, Windows (POSIX-compliant path handling)

---

## Quick Start: Google Colab

Run the complete pipeline end-to-end in Google Colab with one click:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mahendravelagapudi099-wq/Ultra-Data/blob/main/colab/run_all.ipynb)

Alternatively, execute the stages step-by-step in your environment:

```bash
# 1. Clone the repository
git clone https://github.com/mahendravelagapudi099-wq/Ultra-Data.git
cd Ultra-Data

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Stream real web records from Hugging Face
python scripts/load_real_data.py -n 150

# 4. Phase 1: Ingest & clean data (smart orchestrator: uses real data if present, else local fallback)
python scripts/run_phase1.py

# 5. Phase 2: Model-driven selection
python scripts/run_l2.py --config configs/l2_tiny.yaml

# 6. Phase 3 & 3.5: Synthesis and knowledge unit export
python scripts/run_l3.py --config configs/l3_tiny.yaml
python scripts/run_l4_export.py --config configs/l4_tiny.yaml

# 7. Phase 4: Cross-tier evaluation & report generation
python scripts/run_evaluation.py
```

---

## Project Structure

```
Ultra-Dataa/
├── configs/                     # Config-driven pipeline parameters
│   ├── l1_tiny.yaml             # L1 small test configuration (20 docs)
│   ├── l1_expanded.yaml         # L1 expanded configuration (real data + fallback)
│   ├── l2_tiny.yaml             # L2 selection model configuration
│   ├── l2_expanded.yaml         # L2 expanded selection configuration
│   ├── l3_tiny.yaml             # L3 Mock LLM refinement configuration
│   └── l4_tiny.yaml             # L4 structured knowledge export configuration
├── src/
│   ├── pipelines/               # Pipeline orchestration & business logic
│   │   ├── l1_filter.py         # Pure heuristic filter rules & SHA-256 deduplication
│   │   ├── l1_run.py            # Phase 1 I/O orchestration & streaming loading
│   │   ├── l2_select.py         # Phase 2 ML selector (preprocessing fit on train split only)
│   │   ├── l2_run.py            # Phase 2 scoring & selection orchestration
│   │   ├── l3_refine.py         # Phase 3 Mock LLM refinement orchestration
│   │   └── l4_export.py         # Phase 3.5 structural validation & export
│   ├── utils/                   # Pure utility functions
│   │   ├── text_clean.py        # NFKC normalization & regex boilerplate removal
│   │   └── llm_provider.py      # LLMProvider abstraction & MockLLMProvider
│   └── evaluation/
│       └── metrics.py           # Cross-tier quality metrics & Markdown report generator
├── scripts/                     # CLI entry points (Typer + Rich)
│   ├── generate_l0_expanded.py  # Deterministic local expanded substitute generator
│   ├── load_real_data.py        # Hugging Face streaming loader (openbmb/Ultra-FineWeb)
│   ├── run_phase1.py            # Phase 1 smart orchestrator (real data with fallback)
│   ├── run_l1.py                # Standalone Phase 1 runner
│   ├── run_l2.py                # Standalone Phase 2 runner
│   ├── run_l3.py                # Standalone Phase 3 runner
│   ├── run_l4_export.py         # Standalone Phase 3.5 runner
│   └── run_evaluation.py        # Cross-tier evaluation & report generator
├── colab/
│   ├── run_all.ipynb            # End-to-end Google Colab runner notebook
│   └── phase2_colab.ipynb       # Focused Phase 2 Colab exploration notebook
├── data/                        # Gitignored data directory (L0 → L4)
├── reports/                     # Generated evaluation summaries (Markdown, CSV, JSON)
├── AGENTS.md                    # System conventions and developer guidelines
├── PHASES.md                    # Deep-dive mapping of code modules to paper concepts
├── RESULTS_TEMPLATE.md          # Data schema specifications and baseline metrics
└── requirements.txt             # Project dependencies
```

---

## Disclaimer & Reproducibility Notice

> [!NOTE]
> **Demo Scale & Mock LLM Synthesizer:**  
> This project is an architectural reproduction of the tiered data management framework presented in [arXiv:2602.09003](https://arxiv.org/abs/2602.09003).
>
> - All filter thresholds, weak supervision heuristics, and selection ratios are starter values for demo purposes and do not claim paper-scale benchmark results.
> - The L3 synthesis phase uses an offline, deterministic `MockLLMProvider` to demonstrate educational Q&A and textbook formatting without requiring external API keys, GPUs, or heavy model weights.
> - The pipeline is designed to be fully modular: users can substitute production LLM endpoints (OpenAI, Anthropic, Gemini, local Ollama) by subclassing `LLMProvider` in `src/utils/llm_provider.py`.
