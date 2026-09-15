import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from ingestion.config import DbSettings
from ingestion.pipeline import run_pipeline

DEFAULT_CSV = Path("data/raw/Transactions.csv")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ingestion", description="Load the DLD transactions CSV into Postgres."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="path to Transactions.csv")
    args = parser.parse_args(argv)
    load_dotenv()
    try:
        summary = run_pipeline(args.csv, DbSettings.from_env())
    except Exception as exc:  # noqa: BLE001 — CLI boundary: report and exit 1
        print(f"Ingestion failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        f"Run {summary.run_id}: read {summary.rows_read:,} rows, loaded {summary.rows_loaded:,}, "
        f"market sales {summary.rows_market_sale:,}"
    )
    for reason, count in summary.reason_counts.items():
        print(f"  {reason:<26} {count:>10,}")
    print("Peer tiers: " + ", ".join(f"{t}={n:,}" for t, n in summary.peer_tier_counts.items()))
    if summary.unresolved_curated_aliases:
        print("Unresolved curated aliases: " + ", ".join(summary.unresolved_curated_aliases))
    return 0


if __name__ == "__main__":
    sys.exit(main())
