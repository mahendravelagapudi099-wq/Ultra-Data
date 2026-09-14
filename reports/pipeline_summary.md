# Pipeline Evaluation & Tier Comparison Report

> [!NOTE]
> **Notice:** This report reflects a small-scale, offline demo reproduction of the methodology from
> *"Data Science and Technology Towards AGI Part I: Tiered Data Management"* (arXiv:2602.09003).
> Heuristics, thresholds, and mock LLM synthesizers are lightweight starter implementations and do not claim paper-scale results.

**Generated:** 2026-09-14 09:55:46 UTC  
**Environment:** Colab / Linux / Local agnostic  

## 1. Tier-by-Tier Quality Progression

| Tier | Description | Doc Count | Avg Chars | Avg Words | Alpha Ratio | Symbol Ratio | Duplicates |
|---|---|---|---|---|---|---|---|
| `L0_Raw` | Unfiltered substitute web corpus | 145 | 181.9 | 24.0 | 0.7996 | 0.0762 | 10 |
| `L1_Filtered` | Heuristic filtered & exact deduped | 33 | 282.8 | 38.2 | 0.8428 | 0.0248 | 0 |
| `L2_Selected` | Model-selected informative tokens | 16 | 317.8 | 43.8 | 0.8377 | 0.0251 | 0 |
| `L3_Refined` | Mock LLM refined, Q&A & textbook | 16 | 317.8 | 43.8 | 0.8377 | 0.0251 | 0 |
| `L4_Organized` | Validated structured knowledge units | 16 | 366.7 | 49.4 | 0.8251 | 0.0349 | 0 |

## 2. Retention & Transition Rates

- **L0 → L1 Filter Retention Rate:** `22.8%`
- **L1 → L2 Model Selection Rate:** `48.5%`
- **L2 → L3 Refinement Rate:** `100.0%`
- **L3 → L4 Validation Pass Rate:** `100.0%`
- **End-to-End Retention (L0 → L4):** `11.0%`

## 3. Artifact Locations

- `data/l0_raw/l0_expanded_SUBSTITUTE.jsonl`
- `data/l1_filtered/l1_expanded_SUBSTITUTE.parquet`
- `data/l2_selected/l2_selected.parquet`
- `data/l3_refined/l3_refined.parquet`
- `data/l4_organized/l4_organized.parquet`
- `reports/tier_stats.csv`
- `reports/tier_stats.json`
