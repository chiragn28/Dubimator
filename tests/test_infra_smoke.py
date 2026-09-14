"""Infra smoke tests for the Phase 1 Docker Compose stack.

Run manually after `docker compose up -d`:
    uv run pytest tests/test_infra_smoke.py -v

Each test uses a short connection timeout and SKIPS (not fails) with a
clear message if the corresponding service isn't reachable, so running
this suite without Docker running gives a clear signal instead of a
mysterious hang or a red failure.
"""
import os

import psycopg2
import pytest
from dotenv import load_dotenv

load_dotenv()

CONNECT_TIMEOUT = 5


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def test_postgres_pgvector():
    port = _env("POSTGRES_PORT", "5432")
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=port,
            user=_env("POSTGRES_USER", "zestimator"),
            password=_env("POSTGRES_PASSWORD", "changeme"),
            dbname=_env("POSTGRES_DB", "zestimator"),
            connect_timeout=CONNECT_TIMEOUT,
        )
    except psycopg2.OperationalError as exc:
        pytest.skip(f"Postgres not reachable on localhost:{port} — start it with `docker compose up -d postgres`. ({exc})")

    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()
    finally:
        conn.close()
