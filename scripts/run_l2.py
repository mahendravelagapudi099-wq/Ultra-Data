#!/usr/bin/env python3
"""
L2 Demo CLI — Tier 2 Model-Driven Selection.

Usage:
    PYTHONPATH=. python scripts/run_l2.py --config configs/l2_tiny.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import typer
from rich.console import Console
from rich.table import Table

from src.pipelines.l2_run import run_l2

app = typer.Typer(
    name="run_l2",
    help="L2 Selection Demo: TF-IDF + Numeric Selector → Score → Select → Save",
    add_completion=False,
)
console = Console()


@app.command()
def main(
    config: Path = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to L2 YAML config (default: configs/l2_tiny.yaml)",
    ),
) -> None:
    """Run L2 selection pipeline and display summary table."""
    if config is None:
        config = _PROJECT_ROOT / "configs/l2_tiny.yaml"
    elif not config.is_absolute() and not config.exists():
        if (_PROJECT_ROOT / config).exists():
            config = _PROJECT_ROOT / config

    if not config.exists():
        console.print(f"[bold red]Error:[/bold red] Config file not found at: {config.as_posix()}")
        console.print("[yellow]Hint:[/yellow] Verify the configuration path or choose 'configs/l2_tiny.yaml' or 'configs/l2_expanded.yaml'.")
        sys.exit(1)

    console.rule("[bold cyan]L2 Demo — Tier 2 Model-Driven Selection[/bold cyan]")
    console.print(f"Config: [green]{config.as_posix()}[/green]")

    try:
        stats = run_l2(config)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

    # Stats table
    table = Table(title="L2 Selection Statistics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Input Records (L1)", f"{stats['input_count']:,}")
    table.add_row("Selected Records (L2)", f"{stats['selected_count']:,}")
    table.add_row("Selection Rate", f"{stats['selection_rate'] * 100:.1f}%")
    table.add_row("Selector Train Accuracy", f"{stats['train_accuracy'] * 100:.1f}%")
    table.add_row("Selector Test Accuracy", f"{stats['test_accuracy'] * 100:.1f}%")
    table.add_row("Mean Quality Score (All)", f"{stats['mean_quality_score']:.4f}")
    table.add_row("Mean Quality Score (Selected)", f"{stats['mean_selected_score']:.4f}")

    console.print(table)

    # Output paths
    console.print()
    console.print("[bold]Output Files:[/bold]")
    if stats.get("output_scores_parquet"):
        console.print(f"  All Scored (Parquet): [green]{stats['output_scores_parquet']}[/green]")
    if stats.get("output_selected_parquet"):
        console.print(f"  Selected (Parquet):   [green]{stats['output_selected_parquet']}[/green]")
    if stats.get("output_selected_jsonl"):
        console.print(f"  Selected (JSONL):     [green]{stats['output_selected_jsonl']}[/green]")

    console.rule("[bold cyan]Done[/bold cyan]")


if __name__ == "__main__":
    app()
