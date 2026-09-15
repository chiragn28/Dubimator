from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ingestion.areas import build_area_aliases, build_areas
from ingestion.config import DbSettings
from ingestion.load import apply_schema, fail_run, file_sha256, finish_run, replace_data, start_run
from ingestion.normalize import to_typed
from ingestion.rules import REASONS, classify
from ingestion.schema import read_raw


@dataclass(frozen=True)
class RunSummary:
    run_id: int
    rows_read: int
    rows_loaded: int
    rows_market_sale: int
    reason_counts: dict[str, int]
    peer_tier_counts: dict[str, int]
    unresolved_curated_aliases: list[str]


def _reason_counts(classified: pl.DataFrame) -> dict[str, int]:
    counts = {reason: 0 for reason in REASONS}
    counts["market_sale"] = 0
    for reason, n in classified.group_by("exclusion_reason").len().iter_rows():
        counts["market_sale" if reason is None else reason] = n
    return counts


def _peer_tier_counts(classified: pl.DataFrame) -> dict[str, int]:
    tiers = classified.filter(pl.col("peer_tier").is_not_null()).group_by("peer_tier").len()
    return {str(tier): n for tier, n in sorted(tiers.iter_rows())}


def run_pipeline(csv_path: Path, settings: DbSettings) -> RunSummary:
    csv_path = Path(csv_path)
    classified = classify(to_typed(read_raw(csv_path)))
    areas = build_areas(classified)
    aliases, unresolved = build_area_aliases(classified, areas)
    reason_counts = _reason_counts(classified)
    peer_tier_counts = _peer_tier_counts(classified)
    details = {
        "reason_counts": reason_counts,
        "peer_tier_counts": peer_tier_counts,
        "unresolved_curated_aliases": unresolved,
    }
    rows_market_sale = reason_counts["market_sale"]

    conn = settings.connect()
    try:
        apply_schema(conn)
        run_id = start_run(conn, str(csv_path), file_sha256(csv_path), classified.height)
        try:
            with conn.cursor() as cur:
                rows_loaded = replace_data(cur, run_id, classified, areas, aliases)
                finish_run(cur, run_id, rows_loaded, rows_market_sale, details)
            conn.commit()
        except Exception as exc:
            try:
                fail_run(conn, run_id, f"{type(exc).__name__}: {exc}")
            except Exception as mark_exc:  # noqa: BLE001 — preserve root cause, see add_note below
                exc.add_note(f"also failed to mark run {run_id} as failed: {mark_exc!r}")
            raise
    finally:
        conn.close()

    return RunSummary(
        run_id=run_id,
        rows_read=classified.height,
        rows_loaded=rows_loaded,
        rows_market_sale=rows_market_sale,
        reason_counts=reason_counts,
        peer_tier_counts=peer_tier_counts,
        unresolved_curated_aliases=unresolved,
    )
