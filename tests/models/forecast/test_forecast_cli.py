import json
from datetime import date

import polars as pl
from forecast_fixtures import fake_load_rows, prepared_history, project_table

from models.forecast import __main__ as cli
from models.forecast.rows import EXCLUDED_SCHEMA, DataQuality, excluded_summary

HISTORY = prepared_history(areas=2, buildings_per_area=2, start=date(2019, 1, 7))


def quiet(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cli, "_settings", lambda: None)
    monkeypatch.setattr(cli, "load_projects", lambda: project_table(tmp_path))


def test_build_prints_the_report_and_writes_quality_json(monkeypatch, tmp_path, capsys):
    quiet(monkeypatch, tmp_path)
    seen = {}

    def load(settings, config):
        seen.update(sample=config.sample_rows, seed=config.seed)
        return fake_load_rows(HISTORY)(settings, config)

    monkeypatch.setattr(cli, "load_rows", load)
    assert cli.main(["build", "--sample", "500", "--seed", "3"]) == 0
    assert seen == {"sample": 500, "seed": 3}
    out = capsys.readouterr().out
    assert "Dropped 0 of" in out
    assert "Target rows:" in out
    assert "Feature coverage" in out
    payload = json.loads((tmp_path / "quality.json").read_text(encoding="utf-8"))
    assert payload["data_end"] == HISTORY[1].isoformat()
    assert payload["sample_rows"] == 500
    assert payload["targets"]["rows"] == HISTORY[0].height
    assert set(payload) >= {"quality", "excluded", "targets", "feature_coverage"}
    assert "infra_active_metro_rail" in payload["feature_coverage"]
    timings = json.loads((tmp_path / "stage_timings.json").read_text(encoding="utf-8"))
    assert set(timings["build"]) == {"load_rows", "dataset"}


def test_build_stops_when_too_many_rows_are_dropped(monkeypatch, tmp_path, capsys):
    quiet(monkeypatch, tmp_path)
    rows, _ = HISTORY
    quality = DataQuality(
        rows.height * 2,
        {"bad_price": rows.height},
        rows.height,
        pl.DataFrame(schema=EXCLUDED_SCHEMA),
    )
    monkeypatch.setattr(cli, "load_rows", fake_load_rows(HISTORY, quality))
    assert cli.main(["build"]) == 1
    err = capsys.readouterr().err
    assert "Build stopped: 50.0% of rows were dropped, above the 30% limit" in err
    assert not (tmp_path / "quality.json").exists()


def test_build_reports_errors(monkeypatch, tmp_path, capsys):
    quiet(monkeypatch, tmp_path)

    def boom(settings, config):
        raise RuntimeError("database unreachable")

    monkeypatch.setattr(cli, "load_rows", boom)
    assert cli.main(["build"]) == 1
    assert "Build failed: RuntimeError: database unreachable" in capsys.readouterr().err


def test_build_stops_on_an_invalid_infrastructure_table(monkeypatch, tmp_path, capsys):
    quiet(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "load_rows", fake_load_rows(HISTORY))
    monkeypatch.setattr(
        cli, "load_projects", lambda: project_table(tmp_path, affected_area_ids="9")
    )
    assert cli.main(["build"]) == 1
    err = capsys.readouterr().err
    assert "Build stopped: the infrastructure table has problems" in err
    assert "P00: area id 9 is not in dld.areas" in err


def test_excluded_summary_counts_by_segment():
    excluded = pl.DataFrame(
        {
            "area_id": [1, 1, 2],
            "sub_kind": ["flat", "flat", "villa"],
            "reg_type": ["off_plan", "off_plan", "ready"],
            "month": [date(2020, 1, 1)] * 3,
            "reason": ["outlier"] * 3,
        },
        schema=EXCLUDED_SCHEMA,
    )
    assert excluded_summary(excluded) == [
        {"area_id": 1, "sub_kind": "flat", "reg_type": "off_plan", "reason": "outlier", "count": 2},
        {"area_id": 2, "sub_kind": "villa", "reg_type": "ready", "reason": "outlier", "count": 1},
    ]


def test_build_runs_on_the_test_database(pg_test_db, monkeypatch, tmp_path, capsys):
    from pathlib import Path

    from forecast_fixtures import project_record, write_projects

    from ingestion.pipeline import run_pipeline
    from models.forecast.infra import load_projects

    run_pipeline(Path("tests/fixtures/price_sample.csv"), pg_test_db)
    conn = pg_test_db.connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT area_id FROM dld.areas ORDER BY area_id LIMIT 2")
            ids = ";".join(str(row[0]) for row in cur.fetchall())
    finally:
        conn.close()
    path = write_projects(tmp_path, [project_record(i, affected_area_ids=ids) for i in range(15)])
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cli, "_settings", lambda: pg_test_db)
    monkeypatch.setattr(cli, "load_projects", lambda: load_projects(path))
    assert cli.main(["build", "--sample", "1000"]) == 0
    assert "Dropped" in capsys.readouterr().out
    payload = json.loads((tmp_path / "quality.json").read_text(encoding="utf-8"))
    assert payload["quality"]["loaded"] <= 1000


def test_infra_check_lists_a_valid_table(monkeypatch, tmp_path, capsys):
    quiet(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "load_projects", lambda: project_table(tmp_path))
    monkeypatch.setattr(
        cli,
        "fetch_reference",
        lambda settings: ({1: "Dubai Marina", 2: "Al Barsha"}, date(2017, 1, 1)),
    )
    assert cli.main(["infra-check"]) == 0
    out = capsys.readouterr().out
    assert "P00" in out and "Dubai Marina (1)" in out
    assert "15 projects; 0 announced after the data end (2017-01-01)" in out
    assert "Infrastructure table OK" in out


def test_infra_check_fails_on_problems(monkeypatch, tmp_path, capsys):
    quiet(monkeypatch, tmp_path)
    monkeypatch.setattr(
        cli, "load_projects", lambda: project_table(tmp_path, affected_area_ids="5")
    )
    monkeypatch.setattr(cli, "fetch_reference", lambda settings: ({1: "A"}, date(2015, 1, 1)))
    assert cli.main(["infra-check"]) == 1
    captured = capsys.readouterr()
    assert "15 announced after the data end" in captured.out
    assert "P00: area id 5 is not in dld.areas" in captured.err


def test_infra_check_runs_on_the_test_database(pg_test_db, monkeypatch, tmp_path, capsys):
    from pathlib import Path

    from forecast_fixtures import project_record, write_projects

    from ingestion.pipeline import run_pipeline
    from models.forecast.infra import load_projects

    run_pipeline(Path("tests/fixtures/price_sample.csv"), pg_test_db)
    conn = pg_test_db.connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT area_id FROM dld.areas ORDER BY area_id LIMIT 2")
            ids = ";".join(str(row[0]) for row in cur.fetchall())
    finally:
        conn.close()
    path = write_projects(tmp_path, [project_record(i, affected_area_ids=ids) for i in range(15)])
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "_settings", lambda: pg_test_db)
    monkeypatch.setattr(cli, "load_projects", lambda: load_projects(path))
    assert cli.main(["infra-check"]) == 0
    assert "Infrastructure table OK" in capsys.readouterr().out
