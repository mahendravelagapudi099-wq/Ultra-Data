# AGENTS.md

## What this repo is

Reproduces tiered data management for LLM training data, based on the research paper in `paper/paper.pdf`:
> **"Data Science and Technology Towards AGI Part I: Tiered Data Management"** *(arXiv:2602.09003)*

Full multi-tier pipeline implemented:
- **L0:** Raw web crawl (streamed from `openbmb/Ultra-FineWeb` or local expanded substitute)
- **L1 (Clean):** Text normalization, boilerplate removal, heuristic filtering, exact SHA-256 deduplication
- **L2 (Selected):** Weak supervision labeling, TF-IDF + numeric feature selector, calibrated logistic scoring
- **L3 (Refined):** Deterministic offline MockLLM synthesis (refined overview text, Q&A pairs, textbook chapters)
- **L4 (Organized):** Knowledge unit validation & structured export with provenance metadata and UTC timestamps
- **Evaluation:** Cross-tier quality progression metrics, duplicate reduction, and retention reports

## Run the pipeline (Colab / Linux)

```bash
# 1. (Optional) Stream real web sample from Hugging Face
python scripts/load_real_data.py -n 150

# 2. Phase 1: Ingest & clean data (uses real data if available, else local substitute)
python scripts/run_phase1.py

# 3. Phase 2: Model-driven selection
python scripts/run_l2.py --config configs/l2_tiny.yaml

# 4. Phase 3: Mock LLM refinement & synthesis
python scripts/run_l3.py --config configs/l3_tiny.yaml

# 5. Phase 3.5: Knowledge unit validation & organized export
python scripts/run_l4_export.py --config configs/l4_tiny.yaml

# 6. Phase 4: Cross-tier evaluation & report generation
python scripts/run_evaluation.py
```

Or open and run `colab/run_all.ipynb` in Google Colab.

## Architecture & Final File Tree

```
Ultra-Dataa/
├── configs/
│   ├── l1_tiny.yaml             # L1 small test config (20 docs)
│   ├── l1_expanded.yaml         # L1 expanded config (real data + fallback)
│   ├── l2_tiny.yaml             # L2 selection config
│   ├── l2_expanded.yaml         # L2 expanded selection config
│   ├── l3_tiny.yaml             # L3 synthesis config
│   ├── l3_tiny.yaml             # L3 synthesis config
│   ├── l4_tiny.yaml             # L4 structured export config
│   └── microtrain.yaml          # Extension C GPT-2 micro-training config
├── src/
│   ├── __init__.py
│   ├── pipelines/               # Pipeline orchestration & logic
│   │   ├── __init__.py
│   │   ├── l1_run.py            # L1 I/O: real data / substitute loading, file output
│   │   ├── l1_filter.py         # L1 logic: heuristic thresholds, SHA-256 dedup
│   │   ├── l2_run.py            # L2 I/O: loads L1 data, saves scores & selected datasets
│   │   ├── l2_select.py         # L2 logic: weak labels, TF-IDF + numeric pipeline (train split fit)
│   │   ├── l2_fasttext.py       # Extension B: supervised FastText classifier
│   │   ├── l3_refine.py         # L3 orchestration: applies MockLLMProvider or GeminiLLMProvider
│   │   └── l4_export.py         # L4 orchestration: quality validation and metadata formatting
│   ├── utils/                   # Pure utilities
│   │   ├── __init__.py
│   │   ├── text_clean.py        # NFKC normalize, whitespace collapse, boilerplate removal
│   │   └── llm_provider.py      # LLMProvider, MockLLMProvider & GeminiLLMProvider
│   └── evaluation/              # Evaluation metrics
│       ├── __init__.py
│       └── metrics.py           # Cross-tier metric computations & Markdown/CSV/JSON reporting
├── scripts/                     # CLI entry points (Typer + Rich)
│   ├── generate_l0_expanded.py  # Generates ~150-row local substitute dataset
│   ├── load_real_data.py        # Streams real records from openbmb/Ultra-FineWeb → L0
│   ├── run_phase1.py            # Smart Phase 1 launcher: detects real data or generates substitute
│   ├── run_l1.py                # Standalone L1 heuristic filtering runner
│   ├── run_l2.py                # Standalone L2 model selection runner
│   ├── run_l3.py                # Standalone L3 Mock/Gemini LLM refinement runner
│   ├── run_l4_export.py         # Standalone L4 structured knowledge export runner
│   ├── run_microtrain.py        # Extension C: GPT-2 micro-training runner (L1 vs L4 perplexity)
│   └── run_evaluation.py        # Cross-tier evaluation & report runner
├── colab/
│   ├── run_all.ipynb            # End-to-end execution notebook for Google Colab (with optional extensions)
│   └── phase2_colab.ipynb       # Focused Phase 2 exploration notebook
├── data/                        # Gitignored data tiers (L0 → L4)
│   ├── l0_raw/                  # Raw input (real sample or local substitute)
│   ├── l1_filtered/             # L1 cleaned & deduplicated data
│   ├── l2_scores/               # All scored L2 records
│   ├── l2_selected/             # Selected top-tier L2 records
│   ├── l3_refined/              # L3 synthesized text, Q&A, and textbook chapters
│   └── l4_organized/            # L4 validated knowledge units
├── reports/                     # Evaluation summaries (Markdown, CSV, JSON)
├── paper/                       # Research paper reference (paper.pdf)
├── AGENTS.md                    # Developer guidelines and system conventions
├── PHASES.md                    # Mapping of codebase to paper concepts
├── README.md                    # Project overview, Colab badge, architecture diagram
├── RESULTS_TEMPLATE.md          # Output schemas & baseline results table
├── requirements.txt             # Core pipeline dependencies
└── requirements-ml.txt          # Optional ML dependencies for Extensions A/B/C
```

## Conventions

- Modern Python type hints: `str | Path`, `list[dict[str, Any]]`, `pd.DataFrame`
- `from __future__ import annotations` in all Python modules
- All path handling uses `pathlib.Path` with `.as_posix()` for POSIX/Colab compatibility
- Frozen dataclasses with `slots=True` for immutable configuration structures
- Preprocessing fit strictly on training splits to prevent data leakage
- Zero required external API calls; offline deterministic mock LLM provider
- All synthetic outputs explicitly tagged with `generated_by: "mock_llm"`
