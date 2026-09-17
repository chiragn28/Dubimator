import pytest

from ingestion.config import DbSettings

ENV_VARS = ("POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB")


def test_missing_port_raises_instead_of_defaulting_to_5432(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match=r"POSTGRES_PORT.*load_dotenv\(\).*\.env"):
        DbSettings.from_env()


def test_defaults_for_the_other_fields_when_only_the_port_is_set(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("POSTGRES_PORT", "5433")
    assert DbSettings.from_env() == DbSettings(
        host="127.0.0.1", port=5433, user="dubimator", password="changeme", dbname="dubimator"
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
            assert cur.fetchone()[0] == "dubimator_test"
    finally:
        conn.close()
