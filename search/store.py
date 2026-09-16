"""Postgres I/O for schema `search`. Callers own the transaction."""

from pathlib import Path

import polars as pl

from ingestion.load import LoadInvariantError, copy_frame
from listings.load import latest_corpus_run  # noqa: F401 — re-exported for search modules

SCHEMA_SQL = Path(__file__).parent / "sql" / "schema.sql"
QUERY_SCHEMA = {
    "query_id": pl.Int64,
    "text": pl.Utf8,
    "template_id": pl.Int64,
    "kind": pl.Utf8,
    "split": pl.Utf8,
    "true_slots": pl.Utf8,  # JSON text; COPY casts it to jsonb
    "seed_listing_id": pl.Int64,
    "n_grade3": pl.Int64,
    "corpus_run_id": pl.Int64,
}
QUERY_COLUMNS = tuple(name for name in QUERY_SCHEMA if name != "true_slots")
JUDGMENT_SCHEMA = {
    "query_id": pl.Int64,
    "listing_id": pl.Int64,
    "grade": pl.Int64,
    "semantic_cos": pl.Float64,
    "semantic_pos": pl.Int64,
    "fulltext_rank": pl.Float64,
    "fulltext_pos": pl.Int64,
    "rrf_score": pl.Float64,
    "fused_pos": pl.Int64,
    "cluster_size": pl.Int64,
}
ESTIMATE_SCHEMA = {
    "listing_id": pl.Int64,
    "estimate": pl.Float64,
    "low": pl.Float64,
    "high": pl.Float64,
}
LOCK_TIMEOUT = "60s"


def apply_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL.read_text(encoding="utf-8"))


def _count(cur, table: str) -> int:
    cur.execute(f"SELECT count(*) FROM {table}")
    return cur.fetchone()[0]


def replace_query_set(conn, queries: pl.DataFrame, judgments: pl.DataFrame) -> None:
    """Truncate search.queries and search.judgments and COPY these in, verified."""
    with conn.cursor() as cur:
        cur.execute(f"SET LOCAL lock_timeout = '{LOCK_TIMEOUT}'")
        cur.execute("TRUNCATE search.judgments, search.queries")
        copy_frame(cur, "search.queries", queries.select(list(QUERY_SCHEMA)))
        copy_frame(cur, "search.judgments", judgments.select(list(JUDGMENT_SCHEMA)))
        for table, expected in (
            ("search.queries", queries.height),
            ("search.judgments", judgments.height),
        ):
            loaded = _count(cur, table)
            if loaded != expected:
                raise LoadInvariantError(f"{table}: loaded {loaded} rows but expected {expected}")


def read_queries(conn, splits: tuple[str, ...] | None = None) -> pl.DataFrame:
    columns = ", ".join(QUERY_COLUMNS)
    sql = f"SELECT {columns} FROM search.queries"
    params: tuple = ()
    if splits is not None:
        sql += " WHERE split = ANY(%s)"
        params = (list(splits),)
    with conn.cursor() as cur:
        cur.execute(sql + " ORDER BY query_id", params)
        rows = cur.fetchall()
    schema = {name: QUERY_SCHEMA[name] for name in QUERY_COLUMNS}
    return pl.DataFrame(rows, schema=schema, orient="row")


def read_judgments(conn, query_ids: list[int] | None = None) -> pl.DataFrame:
    sql = f"SELECT {', '.join(JUDGMENT_SCHEMA)} FROM search.judgments"
    params: tuple = ()
    if query_ids is not None:
        sql += " WHERE query_id = ANY(%s)"
        params = (list(query_ids),)
    with conn.cursor() as cur:
        cur.execute(sql + " ORDER BY query_id, fused_pos", params)
        rows = cur.fetchall()
    return pl.DataFrame(rows, schema=JUDGMENT_SCHEMA, orient="row")


def queries_corpus_run(conn) -> int | None:
    with conn.cursor() as cur:
        cur.execute("SELECT max(corpus_run_id) FROM search.queries")
        return cur.fetchone()[0]


def replace_estimates(
    conn, estimates: pl.DataFrame, corpus_run_id: int, version: str | None
) -> int:
    frame = estimates.select(list(ESTIMATE_SCHEMA)).with_columns(
        pl.lit(version, dtype=pl.Utf8).alias("price_model_version"),
        pl.lit(corpus_run_id, dtype=pl.Int64).alias("corpus_run_id"),
    )
    with conn.cursor() as cur:
        cur.execute(f"SET LOCAL lock_timeout = '{LOCK_TIMEOUT}'")
        cur.execute("TRUNCATE search.listing_estimates")
        copy_frame(cur, "search.listing_estimates", frame)
        return _count(cur, "search.listing_estimates")


def read_estimates(conn) -> pl.DataFrame:
    with conn.cursor() as cur:
        cur.execute(f"SELECT {', '.join(ESTIMATE_SCHEMA)} FROM search.listing_estimates")
        return pl.DataFrame(cur.fetchall(), schema=ESTIMATE_SCHEMA, orient="row")


def estimates_corpus_run(conn) -> int | None:
    with conn.cursor() as cur:
        cur.execute("SELECT max(corpus_run_id) FROM search.listing_estimates")
        return cur.fetchone()[0]
