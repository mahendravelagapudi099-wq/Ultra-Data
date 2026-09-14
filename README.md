# Reproducing Tiered Data Management for LLM Pre-training

A modular, lightweight prototype reproducing the methodology from:
> **"Data Science and Technology Towards AGI Part I: Tiered Data Management"**  
> *(arXiv:2602.09003)*

This repository implements a multi-tier data curation pipeline designed for execution in **Google Colab** and **Linux** environments (managed via VS Code and opencode CLI).

---

## Pipeline Overview

```
L0 (Raw Web Corpus)
  │
  ▼ [Phase 1: Clean & Filter]
L1 (Cleaned Text + Exact Deduplication)
  │
  ▼ [Phase 2: Model Selection]
L2 (High-Value Selected Informative Tokens)
  │
  ▼ [Phase 3: LLM Synthesis & Refinement]
L3 (Refined Text, Q&A Pairs, Textbook Chapters)
  │
  ▼ [Phase 3.5: Quality Gate & Provenance Validation]
L4 (Structured Knowledge Units)
  │
  ▼ [Phase 4: Evaluation & Reporting]
Reports (Summary Markdown, CSV, JSON)
```

---

## Directory Structure

```
Ultra-Dataa/
├── configs/
│   ├── l1_tiny.yaml             # L1 small test config
│   ├── l1_expanded.yaml         # L1 expanded config
│   ├── l2_tiny.yaml             # L2 selection config
│   ├── l2_expanded.yaml         # L2 expanded selection config
│   ├── l3_tiny.yaml             # L3 LLM synthesis config
│   └── l4_tiny.yaml             # L4 structured export config
├── src/
│   ├── pipelines/
│   │   ├── l1_filter.py         # Pure heuristic filter rules & SHA-256 dedup
│   │   ├── l1_run.py            # L1 I/O orchestration & streaming
│   │   ├── l2_select.py         # TF-IDF + numeric feature selector
│   │   ├── l2_run.py            # L2 scoring & selection orchestration
│   │   ├── l3_refine.py         # L3 synthesis orchestration
│   │   └── l4_export.py         # L4 quality gate & export orchestration
│   ├── utils/
│   │   ├── text_clean.py        # Text normalization & boilerplate removal
│   │   └── llm_provider.py      # LLMProvider abstraction & MockLLMProvider
│   └── evaluation/
│       └── metrics.py           # Cross-tier statistics & report generation
├── scripts/
│   ├── generate_l0_expanded.py  # L0 synthetic substitute generator (~150 docs)
│   ├── run_l1.py                # CLI runner for Phase 1 (L1)
│   ├── run_l2.py                # CLI runner for Phase 2 (L2)
│   ├── run_l3.py                # CLI runner for Phase 3 (L3)
│   ├── run_l4_export.py         # CLI runner for Phase 3.5 (L4)
│   └── run_evaluation.py        # CLI runner for Phase 4 (Evaluation)
├── colab/
│   ├── phase2_colab.ipynb       # Colab execution notebook for Phase 2
│   └── run_all.ipynb            # Complete 9-cell end-to-end execution notebook
├── data/                        # Gitignored data directory
│   ├── l0_raw/
│   ├── l1_filtered/
│   ├── l2_scores/
│   ├── l2_selected/
│   ├── l3_refined/
│   └── l4_organized/
├── reports/                     # Evaluation summaries
│   ├── pipeline_summary.md
│   ├── tier_stats.csv
│   └── tier_stats.json
├── PHASES.md                    # Detailed architectural breakdown of all tiers
├── RESULTS_TEMPLATE.md          # Output schemas & baseline results table
└── requirements.txt             # Starter dependencies
```

---

## Running in Google Colab

Open [colab/run_all.ipynb](colab/run_all.ipynb) in Google Colab and run the cells sequentially:

```bash
# 1. Install dependencies
!pip install -q -r requirements.txt

# 2. Set PYTHONPATH
!PYTHONPATH=. python scripts/generate_l0_expanded.py

# 3. Phase 1: L1 Heuristic Filtering
!PYTHONPATH=. python scripts/run_l1.py --config configs/l1_expanded.yaml

# 4. Phase 2: L2 Model Selection
!PYTHONPATH=. python scripts/run_l2.py --config configs/l2_tiny.yaml

# 5. Phase 3: L3 Refinement & Phase 3.5: L4 Export
!PYTHONPATH=. python scripts/run_l3.py --config configs/l3_tiny.yaml
!PYTHONPATH=. python scripts/run_l4_export.py --config configs/l4_tiny.yaml

# 6. Phase 4: Evaluation
!PYTHONPATH=. python scripts/run_evaluation.py
```

---

## Disclaimer

This codebase is a small-scale, offline demo reproduction. Starter heuristic thresholds and offline mock synthesizers are provided for architectural demonstration and do not claim paper-scale benchmark results.
