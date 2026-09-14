#!/usr/bin/env python3
"""
L3 Demo CLI — Tier 3 LLM Refinement and Synthesis.

Usage:
    PYTHONPATH=. python scripts/run_l3.py --config configs/l3_tiny.yaml
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

from src.pipelines.l3_refine import run_l3

app = typer.Typer(
    name="run_l3",
    help="L3 Refinement Demo: L2 Selected → Mock LLM Synthesis (Text, Q&A, Textbook) → Save",
    add_completion=False,
)
console = Console()


@app.command()
def main(
    config: Path = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to L3 YAML config (default: configs/l3_tiny.yaml)",
    ),
) -> None:
    """Run L3 refinement pipeline and display summary table."""
    if config is None:
        config = _PROJECT_ROOT / "configs/l3_tiny.yaml"
    elif not config.is_absolute() and not config.exists():
        if (_PROJECT_ROOT / config).exists():
            config = _PROJECT_ROOT / config

    if not config.exists():
        console.print(f"[bold red]Error:[/bold red] Config file not found at: {config.as_posix()}")
        console.print("[yellow]Hint:[/yellow] Verify the configuration path or choose 'configs/l3_tiny.yaml'.")
        sys.exit(1)

    console.rule("[bold cyan]L3 Demo — Tier 3 LLM Refinement & Synthesis[/bold cyan]")
    console.print(f"Config: [green]{config.as_posix()}[/green]")

    try:
        stats = run_l3(config)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

    table = Table(title="L3 Refinement Statistics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Input Records (L2 Selected)", f"{stats['input_count']:,}")
    table.add_row("Refined Records (L3)", f"{stats['refined_count']:,}")
    table.add_row("Synthesis Generator", stats["generator"])

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
