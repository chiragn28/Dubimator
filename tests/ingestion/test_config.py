import pytest

from ingestion.config import DbSettings

ENV_VARS = ("POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB")


def test_defaults_when_env_is_empty(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    assert DbSettings.from_env() == DbSettings(
        host="127.0.0.1", port=5432, user="zestimator", password="changeme", dbname="zestimator"
    )


def test_reads_environment(monkeypatch):
    monkeypatch.setenv("POSTGRES_HOST", "postgres")
    monkeypatch.setenv("POSTGRES_PORT", "5433")
    monkeypatch.setenv("POSTGRES_USER", "u")
    monkeypatch.setenv("POSTGRES_PASSWORD", "p")
    monkeypatch.setenv("POSTGRES_DB", "d")
    assert DbSettings.from_env() == DbSettings(
        host="postgres", port=5433, user="u", password="p", dbname="d"
    )


def test_non_numeric_port_raises(monkeypatch):
    monkeypatch.setenv("POSTGRES_PORT", "abc")
    with pytest.raises(ValueError):
        DbSettings.from_env()


def test_pg_test_db_is_a_separate_database(pg_test_db):
    conn = pg_test_db.connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT current_database()")
            assert cur.fetchone()[0] == "zestimator_test"
    finally:
        conn.close()
