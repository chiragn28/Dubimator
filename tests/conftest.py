import models.price  # noqa: F401 — Windows DLL preload, see models/price/__init__.py

# isort: split
import dataclasses
import os
import socket

import psycopg2
import pytest
from dotenv import load_dotenv

load_dotenv()  # no override: shell env wins, matching docker compose's own precedence

from ingestion.config import DbSettings

TEST_DB = "zestimator_test"


def _infra_unavailable(message: str) -> None:
    """Report a missing piece of local infra.

    Normally this SKIPs the test — infra isn't expected to be up everywhere. When
    REQUIRE_INFRA=1 (set by CI, which does bring up Postgres), the same gap is a
    real problem, so it FAILs instead.
    """
    if os.environ.get("REQUIRE_INFRA") == "1":
        pytest.fail(message)
    pytest.skip(message)


@pytest.fixture
def pg_test_db():
    if os.environ.get("POSTGRES_PORT") is None:
        _infra_unavailable(
            "POSTGRES_PORT is not set — start the stack with `docker compose up -d --wait` "
            "(or export POSTGRES_PORT) before running database tests."
        )
    base = DbSettings.from_env()
    try:
        socket.create_connection((base.host, base.port), timeout=5).close()
    except OSError as exc:
        _infra_unavailable(
            f"Postgres not reachable at {base.host}:{base.port} — start the stack with "
            f"`docker compose up -d --wait`. ({exc})"
        )
    admin = psycopg2.connect(
        host=base.host, port=base.port, user=base.user, password=base.password, dbname=base.dbname
    )
    admin.autocommit = True
    try:
        with admin.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
            cur.execute(f"CREATE DATABASE {TEST_DB}")
        yield dataclasses.replace(base, dbname=TEST_DB)
    finally:
        with admin.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
        admin.close()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-mark any test that uses `pg_test_db` as `db`, so `-m "not db"` (scripts/ci.py
    --fast) can exclude the database suite without every such test remembering the marker."""
    for item in items:
        if "pg_test_db" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.db)
