import hashlib
import io
from pathlib import Path

import polars as pl
from psycopg2.extras import Json

SCHEMA_SQL = Path(__file__).parent / "sql" / "schema.sql"
COPY_CHUNK_ROWS = 100_000


class LoadInvariantError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL.read_text(encoding="utf-8"))
    conn.commit()


def start_run(conn, source_path: str, source_sha256: str, rows_read: int) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO dld.ingestion_runs (source_path, source_sha256, rows_read, status) "
            "VALUES (%s, %s, %s, 'running') RETURNING run_id",
            (source_path, source_sha256, rows_read),
        )
        run_id = cur.fetchone()[0]
    conn.commit()
    return run_id


def copy_frame(cur, table: str, frame: pl.DataFrame) -> None:
    columns = ", ".join(frame.columns)
    for offset in range(0, frame.height, COPY_CHUNK_ROWS):
        buffer = io.BytesIO()
        frame.slice(offset, COPY_CHUNK_ROWS).write_csv(
            buffer, include_header=False, null_value="\\N"
        )
        buffer.seek(0)
        cur.copy_expert(
            f"COPY {table} ({columns}) FROM STDIN WITH (FORMAT csv, NULL '\\N')", buffer
        )


def replace_data(
    cur, run_id: int, transactions: pl.DataFrame, areas: pl.DataFrame, aliases: pl.DataFrame
) -> int:
    cur.execute("TRUNCATE dld.transactions, dld.areas, dld.area_aliases")
    copy_frame(
        cur, "dld.transactions", transactions.with_columns(pl.lit(run_id).alias("ingest_run_id"))
    )
    copy_frame(cur, "dld.areas", areas)
    copy_frame(cur, "dld.area_aliases", aliases)
    cur.execute("SELECT count(*) FROM dld.transactions")
    loaded = cur.fetchone()[0]
    if loaded != transactions.height:
        raise LoadInvariantError(f"loaded {loaded} rows but read {transactions.height}")
    return loaded


def finish_run(cur, run_id: int, rows_loaded: int, rows_market_sale: int, details: dict) -> None:
    cur.execute(
        "UPDATE dld.ingestion_runs SET status = 'succeeded', finished_at = now(), "
        "rows_loaded = %s, rows_market_sale = %s, details = %s WHERE run_id = %s",
        (rows_loaded, rows_market_sale, Json(details), run_id),
    )


def fail_run(conn, run_id: int, error: str) -> None:
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE dld.ingestion_runs SET status = 'failed', finished_at = now(), error = %s "
            "WHERE run_id = %s",
            (error, run_id),
        )
    conn.commit()
