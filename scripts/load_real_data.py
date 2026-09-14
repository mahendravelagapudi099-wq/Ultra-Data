#!/usr/bin/env python3
"""
Stream a small real sample from openbmb/Ultra-FineWeb (Hugging Face) into L0 raw storage.

Usage:
    PYTHONPATH=. python scripts/load_real_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from datasets import load_dataset
from rich.console import Console
from rich.table import Table

console = Console()

DATASET_NAME = "openbmb/Ultra-FineWeb"
TARGET_SPLIT = "train"
NUM_RECORDS = 300
OUTPUT_PATH = Path("data/l0_raw/l0_real_sample.jsonl")


def extract_text_from_record(row: dict) -> tuple[str, str]:
    """
    Dynamically locate text content and optional URL in dataset row.
    """
    # Look for common text field names
    text_candidates = ["text", "content", "raw_content", "body", "document"]
    text = ""
    for field in text_candidates:
        if field in row and isinstance(row[field], str) and row[field].strip():
            text = row[field]
            break

    # If not found among candidates, pick first non-empty string value
    if not text:
        for k, v in row.items():
            if isinstance(v, str) and len(v.strip()) > 20:
                text = v
                break

    url = str(row.get("url", row.get("source_url", "")))
    return text, url


def load_real_sample(
    dataset_name: str = DATASET_NAME,
    split: str = TARGET_SPLIT,
    n: int = NUM_RECORDS,
    output_path: Path = OUTPUT_PATH,
) -> bool:
    """
    Stream records from Hugging Face dataset and write to JSONL.
    
    Returns True if successfully retrieved records, False otherwise.
    """
    console.rule("[bold cyan]Fetching Real Web Sample from Hugging Face[/bold cyan]")
    console.print(f"Dataset: [green]{dataset_name}[/green] (split: {split})")
    console.print(f"Target count: [yellow]{n} records (streaming mode)[/yellow]")
    console.print(f"Destination: [green]{output_path.as_posix()}[/green]\n")

    ds = None
    actual_split = split
    for s in ["en", "zh", split, "train"]:
        try:
            ds = load_dataset(dataset_name, split=s, streaming=True)
            actual_split = s
            console.print(f"Connected to [green]{dataset_name}[/green] (split: [cyan]{s}[/cyan])")
            break
        except Exception as err:
            console.print(f"[dim]Split '{s}' not found ({err}). Trying next...[/dim]")

    if ds is None:
        console.print("[yellow]Ultra-FineWeb unavailable. Trying HuggingFaceFW/fineweb (sample-10BT)...[/yellow]")
        try:
            ds = load_dataset("HuggingFaceFW/fineweb", split="sample-10BT", streaming=True)
            dataset_name = "HuggingFaceFW/fineweb"
            actual_split = "sample-10BT"
            console.print("Connected to [green]HuggingFaceFW/fineweb[/green] (split: [cyan]sample-10BT[/cyan])")
        except Exception as e:
            console.print(f"[bold yellow]Warning:[/bold yellow] Real streaming failed: {e}")
            console.print("[yellow]The pipeline will gracefully fall back to local substitute data.[/yellow]")
            return False

    records = []
    total_chars = 0

    try:
        console.print("Streaming records...")
        for i, row in enumerate(ds):
            if i >= n:
                break
            text, url = extract_text_from_record(row)
            if not text:
                continue

            record = {
                "id": f"real_{len(records)}",
                "text": text,
                "source": dataset_name,
                "url": url,
            }
            records.append(record)
            total_chars += len(text)

            if (len(records)) % 50 == 0:
                console.print(f"  Fetched {len(records)}/{n} records...")

    except Exception as e:
        console.print(f"[bold yellow]Warning:[/bold yellow] Error during streaming iteration: {e}")
        if not records:
            console.print("[yellow]No records fetched. Pipeline will use local fallback.[/yellow]")
            return False
        console.print(f"[yellow]Retaining {len(records)} partially fetched records.[/yellow]")

    if not records:
        console.print("[bold yellow]Warning: No records were extracted. Pipeline will use local fallback.[/bold yellow]")
        return False

    # Save to destination JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    avg_chars = round(total_chars / max(len(records), 1), 1)

    table = Table(title="Real Data Fetch Summary", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_row("Dataset Name", dataset_name)
    table.add_row("Records Fetched", str(len(records)))
    table.add_row("Average Text Length (chars)", f"{avg_chars:,}")
    table.add_row("Output File", output_path.as_posix())
    console.print(table)
    console.rule("[bold cyan]Fetch Complete[/bold cyan]")
    return True


def main() -> None:
    success = load_real_sample()
    if not success:
        # Exit with status 0 so calling automation / Colab continues to fallback
        console.print("[dim]Graceful exit — ready for fallback pipeline execution.[/dim]")
        sys.exit(0)


if __name__ == "__main__":
    main()
