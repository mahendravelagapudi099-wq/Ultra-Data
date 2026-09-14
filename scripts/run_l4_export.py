#!/usr/bin/env python3
"""
L4 Demo CLI — Tier 4 Organized Knowledge Export.

Usage:
    PYTHONPATH=. python scripts/run_l4_export.py --config configs/l4_tiny.yaml
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

from src.pipelines.l4_export import run_l4_export

app = typer.Typer(
    name="run_l4_export",
    help="L4 Export Demo: Validate L3 Refined Units → Add Strict Metadata → Save to L4 Organized",
    add_completion=False,
)
console = Console()


@app.command()
def main(
    config: Path = typer.Option(
        Path("configs/l4_tiny.yaml"),
        "--config",
        "-c",
        help="Path to L4 YAML config",
        exists=True,
        readable=True,
    ),
) -> None:
    """Run L4 validation & export pipeline and display summary table."""
    console.rule("[bold cyan]L4 Demo — Tier 4 Organized Knowledge Export[/bold cyan]")
    console.print(f"Config: [green]{config.as_posix()}[/green]")

    try:
        stats = run_l4_export(config)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

    table = Table(title="L4 Export Statistics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Input Units (L3)", str(stats["input_count"]))
    table.add_row("Valid Units (L4)", str(stats["valid_count"]))
    table.add_row("Invalid Units", str(stats["invalid_count"]))
    table.add_row("Validation Pass Rate", f"{stats['validation_pass_rate'] * 100:.1f}%")

    console.print(table)

    console.print()
    console.print("[bold]Output Files:[/bold]")
    if stats.get("output_path_parquet"):
        console.print(f"  Parquet: [green]{stats['output_path_parquet']}[/green]")
    if stats.get("output_path_jsonl"):
        console.print(f"  JSONL:   [green]{stats['output_path_jsonl']}[/green]")

    console.rule("[bold cyan]Done[/bold cyan]")


if __name__ == "__main__":
    app()
