import dataclasses
import socket

import psycopg2
import pytest
from dotenv import load_dotenv

load_dotenv()  # no override: shell env wins, matching docker compose's own precedence

from ingestion.config import DbSettings

TEST_DB = "zestimator_test"


@pytest.fixture
def pg_test_db():
    base = DbSettings.from_env()
    try:
        socket.create_connection((base.host, base.port), timeout=5).close()
    except OSError as exc:
        pytest.skip(
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
