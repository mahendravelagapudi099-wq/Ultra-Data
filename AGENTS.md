# AGENTS.md

## What this repo is

Reproduces tiered data management for LLM training data, based on the paper in `paper/paper.pdf` (*"Data Science and Technology Towards AGI Part I: Tiered Data Management"*, arXiv:2602.09003).

Full pipeline implemented:
- **L0:** Raw / local expanded substitute data
- **L1 (Clean):** Text normalization, heuristic filtering, exact deduplication
- **L2 (Selected):** Weak supervision demo labeling, TF-IDF + numeric feature selector
- **L3 (Refined):** Deterministic offline MockLLM synthesis (cleaned text, Q&A, textbook explanation)
- **L4 (Organized):** Knowledge unit validation & structured export with provenance metadata
- **Evaluation:** Cross-tier quality metrics and transition reports

## Run the pipeline (Colab / Linux)

```bash
# Set PYTHONPATH and run full pipeline
PYTHONPATH=. python scripts/generate_l0_expanded.py
PYTHONPATH=. python scripts/run_l1.py --config configs/l1_expanded.yaml
PYTHONPATH=. python scripts/run_l2.py --config configs/l2_tiny.yaml
PYTHONPATH=. python scripts/run_l3.py --config configs/l3_tiny.yaml
PYTHONPATH=. python scripts/run_l4_export.py --config configs/l4_tiny.yaml
PYTHONPATH=. python scripts/run_evaluation.py
```

Or open and run `colab/run_all.ipynb` in Google Colab.

## Architecture

```
scripts/                  # CLI entry points (Typer + Rich)
  generate_l0_expanded.py # Generates ~150-row local substitute dataset
  run_l1.py               # L1 Heuristic filtering runner
  run_l2.py               # L2 Model selection runner
  run_l3.py               # L3 Mock LLM refinement runner
  run_l4_export.py        # L4 Structured knowledge export runner
  run_evaluation.py       # Cross-tier evaluation & report runner

src/pipelines/            # Pipeline orchestration
  l1_run.py               # L1 I/O: streaming / local substitute loading, file output
  l1_filter.py            # L1 logic: heuristic thresholds, SHA-256 dedup
  l2_run.py               # L2 I/O: loads L1 data, saves scores & selected datasets
  l2_select.py            # L2 logic: weak labels, TF-IDF + numeric pipeline (fit on train split only)
  l3_refine.py            # L3 orchestration: applies MockLLMProvider to selected docs
  l4_export.py            # L4 orchestration: quality validation and metadata formatting

src/utils/                # Utilities
  text_clean.py           # Pure text cleaning: NFKC normalize, boilerplate removal
  llm_provider.py         # LLMProvider abstraction & deterministic MockLLMProvider

src/evaluation/           # Evaluation metrics
  metrics.py              # Pure metric computations & markdown/CSV/JSON report generation
```

## Configs

- `configs/l1_tiny.yaml` — 20 docs L1 test config
- `configs/l1_expanded.yaml` — 200 docs L1 config
- `configs/l2_tiny.yaml` — L2 selection config
- `configs/l2_expanded.yaml` — L2 expanded selection config
- `configs/l3_tiny.yaml` — L3 synthesis config
- `configs/l4_tiny.yaml` — L4 export config

Filter thresholds and weak labels are **starter heuristic values — NOT cited from paper.pdf**.

## Data

All data files are gitignored (`data/**/*.jsonl`, `data/**/*.parquet`, etc.):
- `data/l0_raw/` — Raw input
- `data/l1_filtered/` — L1 output
- `data/l2_scores/` — All scored L2 records
- `data/l2_selected/` — Selected L2 records
- `data/l3_refined/` — L3 synthesized text, Q&A, and textbook chapters
- `data/l4_organized/` — L4 validated knowledge units

## Conventions

- Modern Python type hints: `str | Path`, `list[dict[str, Any]]`
- `from __future__ import annotations`
- Relative POSIX forward-slash paths for Linux/Colab compatibility
- Frozen dataclasses with `slots=True`
- Preprocessing fit strictly on training splits to prevent data leakage
- Zero external API calls, offline deterministic mock LLM provider
