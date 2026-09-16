import json
from datetime import date

import polars as pl
from forecast_fixtures import fake_load_rows, prepared_history

from models.forecast import __main__ as cli
from models.forecast.rows import EXCLUDED_SCHEMA, DataQuality, excluded_summary

HISTORY = prepared_history(areas=2, buildings_per_area=2, start=date(2019, 1, 7))


def quiet(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cli, "_settings", lambda: None)


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
    timings = json.loads((tmp_path / "stage_timings.json").read_text(encoding="utf-8"))
    assert set(timings["build"]) == {"load_rows", "targets", "features"}


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

    from ingestion.pipeline import run_pipeline

    run_pipeline(Path("tests/fixtures/price_sample.csv"), pg_test_db)
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cli, "_settings", lambda: pg_test_db)
    assert cli.main(["build", "--sample", "1000"]) == 0
    assert "Dropped" in capsys.readouterr().out
    payload = json.loads((tmp_path / "quality.json").read_text(encoding="utf-8"))
    assert payload["quality"]["loaded"] <= 1000
