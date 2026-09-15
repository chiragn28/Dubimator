from pathlib import Path

import polars as pl
import psycopg2
import pytest

from ingestion.__main__ import main
from ingestion.pipeline import run_pipeline
from ingestion.rules import MARKET_SALE_PROCEDURES, REASONS
from ingestion.schema import SchemaDriftError

FIXTURE = Path(__file__).parent.parent / "fixtures" / "dld_sample.csv"
MARKET = sorted(MARKET_SALE_PROCEDURES)
NOT_DUP = "exclusion_reason IS DISTINCT FROM 'duplicate_transaction_id'"


def query(settings, sql, params=()):
    conn = settings.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    finally:
        conn.close()


def scalar(settings, sql, params=()):
    return query(settings, sql, params)[0][0]


def test_loads_every_fixture_row(pg_test_db):
    summary = run_pipeline(FIXTURE, pg_test_db)
    fixture_rows = pl.read_csv(FIXTURE, infer_schema=False).height
    assert summary.rows_read == summary.rows_loaded == fixture_rows
    assert set(summary.reason_counts) == {*REASONS, "market_sale"}
    assert sum(summary.reason_counts.values()) == summary.rows_read
    assert summary.rows_market_sale == summary.reason_counts["market_sale"]
    assert scalar(pg_test_db, "SELECT count(*) FROM dld.market_sales") == summary.rows_market_sale
    assert scalar(pg_test_db, "SELECT count(*) FROM dld.transactions") == fixture_rows
    run = query(
        pg_test_db,
        "SELECT status, rows_read, rows_loaded, details->'reason_counts' FROM dld.ingestion_runs",
    )
    assert run == [("succeeded", fixture_rows, fixture_rows, summary.reason_counts)]


def test_deterministic_rules_hold_on_loaded_rows(pg_test_db):
    run_pipeline(FIXTURE, pg_test_db)
    violations = {
        "mortgage": f"trans_group = 'mortgages' AND {NOT_DUP} AND exclusion_reason <> 'mortgage'",
        "gift": f"trans_group = 'gifts' AND {NOT_DUP} AND exclusion_reason <> 'gift'",
        "non_market_procedure": (
            f"trans_group = 'sales' AND NOT (procedure_name = ANY(%(market)s)) AND {NOT_DUP} "
            "AND exclusion_reason <> 'non_market_procedure'"
        ),
        "missing_price": (
            f"trans_group = 'sales' AND procedure_name = ANY(%(market)s) AND instance_date IS NOT NULL "
            f"AND price_aed IS NULL AND {NOT_DUP} AND exclusion_reason <> 'missing_price'"
        ),
        "price_below_floor": (
            f"trans_group = 'sales' AND procedure_name = ANY(%(market)s) AND instance_date IS NOT NULL "
            f"AND price_aed < 10000 AND area_sqm > 0 AND {NOT_DUP} "
            "AND exclusion_reason IS DISTINCT FROM 'price_below_floor'"
        ),
    }
    for reason, condition in violations.items():
        bad = scalar(
            pg_test_db,
            f"SELECT count(*) FROM dld.transactions WHERE {condition}",
            {"market": MARKET},
        )
        assert bad == 0, reason
        seen = scalar(
            pg_test_db,
            "SELECT count(*) FROM dld.transactions WHERE exclusion_reason = %s",
            (reason,),
        )
        assert seen >= 1, reason
    assert (
        scalar(pg_test_db, "SELECT count(*) FROM dld.transactions WHERE instance_date IS NULL") == 5
    )


def test_areas_and_aliases_are_loaded(pg_test_db):
    summary = run_pipeline(FIXTURE, pg_test_db)
    areas = scalar(pg_test_db, "SELECT count(*) FROM dld.areas")
    assert areas == scalar(pg_test_db, "SELECT count(DISTINCT area_id) FROM dld.transactions")
    official = scalar(pg_test_db, "SELECT count(*) FROM dld.area_aliases WHERE source = 'official'")
    assert official == areas
    assert isinstance(summary.unresolved_curated_aliases, list)


def test_rerun_is_a_full_refresh(pg_test_db):
    first = run_pipeline(FIXTURE, pg_test_db)
    second = run_pipeline(FIXTURE, pg_test_db)
    assert second.reason_counts == first.reason_counts
    assert scalar(pg_test_db, "SELECT count(*) FROM dld.transactions") == first.rows_read
    assert query(pg_test_db, "SELECT status FROM dld.ingestion_runs ORDER BY run_id") == [
        ("succeeded",),
        ("succeeded",),
    ]
    assert query(pg_test_db, "SELECT DISTINCT ingest_run_id FROM dld.transactions") == [
        (second.run_id,)
    ]


def test_failure_inside_transaction_rolls_back_and_marks_run_failed(pg_test_db, monkeypatch):
    first = run_pipeline(FIXTURE, pg_test_db)

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("ingestion.pipeline.finish_run", boom)

    with pytest.raises(RuntimeError, match="boom"):
        run_pipeline(FIXTURE, pg_test_db)

    assert scalar(pg_test_db, "SELECT count(*) FROM dld.transactions") == first.rows_read
    assert query(pg_test_db, "SELECT DISTINCT ingest_run_id FROM dld.transactions") == [
        (first.run_id,)
    ]
    assert query(pg_test_db, "SELECT status FROM dld.ingestion_runs ORDER BY run_id") == [
        ("succeeded",),
        ("failed",),
    ]
    error, finished_at = query(
        pg_test_db, "SELECT error, finished_at FROM dld.ingestion_runs WHERE status = 'failed'"
    )[0]
    assert error.startswith("RuntimeError: boom")
    assert finished_at is not None


def test_fail_run_failure_preserves_root_cause_as_note(pg_test_db, monkeypatch):
    def boom_finish(*args, **kwargs):
        raise RuntimeError("boom")

    def boom_fail(*args, **kwargs):
        raise psycopg2.InterfaceError("connection already closed")

    monkeypatch.setattr("ingestion.pipeline.finish_run", boom_finish)
    monkeypatch.setattr("ingestion.pipeline.fail_run", boom_fail)

    with pytest.raises(RuntimeError, match="boom") as excinfo:
        run_pipeline(FIXTURE, pg_test_db)

    notes = getattr(excinfo.value, "__notes__", [])
    assert any("also failed to mark run" in note for note in notes)


def test_schema_drift_writes_nothing(pg_test_db, tmp_path):
    drifted = tmp_path / "drift.csv"
    pl.read_csv(FIXTURE, infer_schema=False).rename({"actual_worth": "amount"}).write_csv(drifted)
    with pytest.raises(SchemaDriftError):
        run_pipeline(drifted, pg_test_db)
    assert scalar(pg_test_db, "SELECT to_regclass('dld.ingestion_runs')") is None


def test_cli_success(pg_test_db, monkeypatch, capsys):
    for name, value in {
        "POSTGRES_HOST": pg_test_db.host,
        "POSTGRES_PORT": str(pg_test_db.port),
        "POSTGRES_USER": pg_test_db.user,
        "POSTGRES_PASSWORD": pg_test_db.password,
        "POSTGRES_DB": pg_test_db.dbname,
    }.items():
        monkeypatch.setenv(name, value)
    assert main(["--csv", str(FIXTURE)]) == 0
    out = capsys.readouterr().out
    assert "market_sale" in out and "mortgage" in out


def test_cli_failure_returns_1(tmp_path, capsys):
    assert main(["--csv", str(tmp_path / "missing.csv")]) == 1
    assert "missing.csv" in capsys.readouterr().err
