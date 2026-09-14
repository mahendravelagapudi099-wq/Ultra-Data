#!/usr/bin/env python3
"""
L1 Tiny Demo CLI — Tier 1 Data Cleaning.

Usage:
    PYTHONPATH=. python scripts/run_l1.py --config configs/l1_tiny.yaml
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

from src.pipelines.l1_run import run_l1

app = typer.Typer(
    name="run_l1",
    help="L1 Tiny Demo: Load → Clean → Filter → Dedupe → Save",
    add_completion=False,
)
console = Console()


@app.command()
def main(
    config: Path = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to L1 YAML config (default: configs/l1_tiny.yaml)",
    ),
) -> None:
    """Run L1 pipeline and print stats table."""
    if config is None:
        config = _PROJECT_ROOT / "configs/l1_tiny.yaml"
    elif not config.is_absolute() and not config.exists():
        if (_PROJECT_ROOT / config).exists():
            config = _PROJECT_ROOT / config

    if not config.exists():
        console.print(f"[bold red]Error:[/bold red] Config file not found at: {config.as_posix()}")
        console.print("[yellow]Hint:[/yellow] Verify the configuration path or choose 'configs/l1_tiny.yaml' or 'configs/l1_expanded.yaml'.")
        sys.exit(1)

    console.rule("[bold cyan]L1 Tiny Demo — Tier 1 Data Cleaning[/bold cyan]")
    console.print(f"Config: [green]{config.as_posix()}[/green]")

    try:
        stats = run_l1(config)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

    # Stats table
    table = Table(title="L1 Pipeline Statistics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Input Count", f"{stats['input_count']:,}")
    table.add_row("Output Count", f"{stats['output_count']:,}")
    table.add_row("Removed (filtered)", f"{stats['removed_count']:,}")
    table.add_row("Duplicates Removed", f"{stats['duplicate_count']:,}")
    table.add_row("Source Used", str(stats["source_used"]))

    console.print(table)

    # Filter reasons breakdown
    if stats["filter_reasons"]:
        reason_table = Table(title="Filter Reasons", show_header=True, header_style="bold magenta")
        reason_table.add_column("Reason", style="cyan")
        reason_table.add_column("Count", style="yellow", justify="right")
        for reason, count in sorted(stats["filter_reasons"].items()):
            reason_table.add_row(reason, str(count))
        console.print(reason_table)

    # Output paths
    console.print()
    console.print("[bold]Output Files:[/bold]")
    if stats["output_path_parquet"]:
        console.print(f"  Parquet: [green]{stats['output_path_parquet']}[/green]")
    if stats["output_path_jsonl"]:
        console.print(f"  JSONL:   [green]{stats['output_path_jsonl']}[/green]")

    console.rule("[bold cyan]Done[/bold cyan]")


if __name__ == "__main__":
    app()