"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit disclosure coverage in the GOV.UK ATRS repository."
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path.cwd(), help="Repository root"
    )
    parser.add_argument("--workers", type=int, default=8, help="Concurrent requests")
    args = parser.parse_args()
    summary = run(args.output_root.resolve(), max_workers=max(1, args.workers))
    print(
        f"Analysed {summary['record_count']} records from "
        f"{summary['organisation_count']} organisations."
    )


if __name__ == "__main__":
    main()
