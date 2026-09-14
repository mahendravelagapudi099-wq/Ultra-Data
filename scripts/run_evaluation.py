#!/usr/bin/env python3
"""
Pipeline Evaluation and Comparison CLI — Tier-by-Tier Progression.

Usage:
    PYTHONPATH=. python scripts/run_evaluation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from src.evaluation.metrics import compute_tier_metrics, generate_markdown_report

app = typer.Typer(
    name="run_evaluation",
    help="Cross-Tier Evaluation: Compute stats across L0-L4 and output comparison reports",
    add_completion=False,
)
console = Console()


def _load_tier_df(paths: list[Path]) -> pd.DataFrame:
    """Load first existing file among candidate paths."""
    for p in paths:
        if p.exists():
            if p.suffix == ".parquet":
                return pd.read_parquet(p)
            elif p.suffix == ".jsonl":
                records = []
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            records.append(json.loads(line))
                return pd.DataFrame(records)
    return pd.DataFrame()


@app.command()
def main(
    output_dir: Path = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory to save evaluation reports (default: <project_root>/reports)",
    ),
) -> None:
    """Run pipeline evaluation and generate cross-tier comparison reports."""
    if output_dir is None:
        output_dir = _PROJECT_ROOT / "reports"
    console.rule("[bold cyan]Pipeline Evaluation — Cross-Tier Progression[/bold cyan]")

    # Tier candidate file paths (expanded or tiny)
    tier_files = {
        "L0_Raw": [
            _PROJECT_ROOT / "data/l0_raw/l0_real_sample.jsonl",
            _PROJECT_ROOT / "data/l0_raw/l0_expanded_SUBSTITUTE.jsonl",
            _PROJECT_ROOT / "data/l0_raw/sample_l0_tiny_SUBSTITUTE.jsonl",
        ],
        "L1_Filtered": [
            _PROJECT_ROOT / "data/l1_filtered/l1_expanded_SUBSTITUTE.parquet",
            _PROJECT_ROOT / "data/l1_filtered/l1_filtered_tiny.parquet",
            _PROJECT_ROOT / "data/l1_filtered/l1_expanded_SUBSTITUTE.jsonl",
        ],
        "L2_Selected": [
            _PROJECT_ROOT / "data/l2_selected/l2_selected.parquet",
            _PROJECT_ROOT / "data/l2_selected/l2_selected.jsonl",
        ],
        "L3_Refined": [
            _PROJECT_ROOT / "data/l3_refined/l3_refined.parquet",
            _PROJECT_ROOT / "data/l3_refined/l3_refined.jsonl",
        ],
        "L4_Organized": [
            _PROJECT_ROOT / "data/l4_organized/l4_organized.parquet",
            _PROJECT_ROOT / "data/l4_organized/l4_organized.jsonl",
        ],
    }

    tier_metrics = []
    doc_counts = {}

    for tier_name, candidates in tier_files.items():
        df = _load_tier_df(candidates)
        m = compute_tier_metrics(df, tier_name)
        tier_metrics.append(m)
        doc_counts[tier_name] = m["doc_count"]

    # Compute transition rates
    l0_count = doc_counts.get("L0_Raw", 0)
    l1_count = doc_counts.get("L1_Filtered", 0)
    l2_count = doc_counts.get("L2_Selected", 0)
    l3_count = doc_counts.get("L3_Refined", 0)
    l4_count = doc_counts.get("L4_Organized", 0)

    rates = {
        "l1_retention": round(l1_count / max(l0_count, 1), 4),
        "l2_selection_rate": round(l2_count / max(l1_count, 1), 4),
        "l3_refinement_rate": round(l3_count / max(l2_count, 1), 4),
        "l4_pass_rate": round(l4_count / max(l3_count, 1), 4),
        "end_to_end_rate": round(l4_count / max(l0_count, 1), 4),
    }

    # Rich summary table
    table = Table(title="Cross-Tier Progression Summary", show_header=True, header_style="bold magenta")
    table.add_column("Tier", style="cyan")
    table.add_column("Doc Count", style="green", justify="right")
    table.add_column("Avg Words", style="yellow", justify="right")
    table.add_column("Alpha Ratio", style="magenta", justify="right")
    table.add_column("Symbol Ratio", style="blue", justify="right")
    table.add_column("Duplicates", style="red", justify="right")

    for m in tier_metrics:
        table.add_row(
            m["tier"],
            str(m["doc_count"]),
            str(m["avg_word_count"]),
            f"{m['avg_alpha_ratio']:.4f}",
            f"{m['avg_symbol_ratio']:.4f}",
            str(m["duplicate_count"]),
        )

    console.print(table)

    # Save reports
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "tier_stats.csv"
    json_path = output_dir / "tier_stats.json"
    md_path = output_dir / "pipeline_summary.md"

    df_metrics = pd.DataFrame(tier_metrics)
    df_metrics.to_csv(csv_path, index=False)

    report_payload = {
        "tier_metrics": tier_metrics,
        "transition_rates": rates,
        "disclaimer": "Tiny demo reproduction — starter heuristics and mock models, not paper-scale results.",
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    md_content = generate_markdown_report(tier_metrics, rates)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    console.print()
    console.print("[bold]Generated Reports:[/bold]")
    console.print(f"  Markdown: [green]{md_path.as_posix()}[/green]")
    console.print(f"  CSV:      [green]{csv_path.as_posix()}[/green]")
    console.print(f"  JSON:     [green]{json_path.as_posix()}[/green]")
    console.rule("[bold cyan]Done[/bold cyan]")


if __name__ == "__main__":
    app()
