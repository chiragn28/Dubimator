import json
import re
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
    rows, _ = HISTORY
    excluded = pl.DataFrame(
        {"area_id": [1, 2], "sub_kind": ["flat", "villa"], "market_kind": ["flat", "villa"],
         "reg_type": ["ready", "ready"], "month": [date(2020, 1, 1), date(2020, 2, 1)],
         "reason": ["repeat_sale", "outlier"]},
        schema=EXCLUDED_SCHEMA,
    )  # fmt: skip
    quality = DataQuality(rows.height + 2, {"repeat_sale": 1, "outlier": 1}, rows.height, excluded)

    def load(settings, config):
        seen.update(sample=config.sample_rows, seed=config.seed)
        return fake_load_rows(HISTORY, quality)(settings, config)

    monkeypatch.setattr(cli, "load_rows", load)
    assert cli.main(["build", "--sample", "500", "--seed", "3"]) == 0
    assert seen == {"sample": 500, "seed": 3}
    out = capsys.readouterr().out
    assert f"Dropped 2 of {rows.height + 2:,}" in out
    assert "Target rows:" in out
    assert "Feature coverage" in out
    payload = json.loads((tmp_path / "quality.json").read_text(encoding="utf-8"))
    assert payload["data_end"] == HISTORY[1].isoformat()
    assert payload["sample_rows"] == 500
    assert payload["targets"]["rows"] == HISTORY[0].height
    assert set(payload) >= {"quality", "excluded", "targets", "feature_coverage"}
    assert "infra_active_metro_rail" in payload["feature_coverage"]
    written = pl.read_csv(tmp_path / "excluded.csv", try_parse_dates=True)
    assert written.columns == list(EXCLUDED_SCHEMA)
    assert written.cast(EXCLUDED_SCHEMA).equals(excluded)
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
            "market_kind": ["flat", "flat", "villa_plot"],
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


def _fast_config(**overrides):
    from models.forecast.config import ForecastConfig

    values = {
        "n_trials": 1, "n_estimators": 200, "learning_rate": 0.1, "tune_folds": 2,
        "min_test_rows": 50, "n_bootstrap": 200, **overrides,
    }  # fmt: skip
    return lambda: ForecastConfig(**values)


def test_train_end_to_end_then_evaluate_and_predict(monkeypatch, tmp_path, capsys, temp_mlflow):
    import mlflow
    from forecast_fixtures import FakePrice

    from models.forecast import predict as predict_module

    rich = prepared_history(per_building_per_week=4)
    quiet(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "load_rows", fake_load_rows(rich))
    monkeypatch.setattr(cli, "ForecastConfig", _fast_config())
    monkeypatch.setattr(predict_module, "load_rows", fake_load_rows(rich))
    monkeypatch.setattr(predict_module, "load_projects", lambda: project_table(tmp_path))
    monkeypatch.setattr("listings.fraud.load_price_predictor", lambda uri: FakePrice())
    mlflow.create_experiment("price-forecast", artifact_location=temp_mlflow["artifact_location"])

    assert cli.main(["train", "--trials", "1", "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert re.search(r"3m: passed \(22 folds: 17 score, 2 tune, 2 gap, 1 test; \d+s\)", out)
    assert "3y: insufficient_data" in out
    assert "segment" in out and "area_trend" in out
    assert "Registered zestimator-forecast-3m version 1 as @champion" in out

    assert cli.main(["evaluate"]) == 0
    out = capsys.readouterr().out
    assert "3m champion v1" in out
    assert "3y: not deployed (3y: only 0 walk-forward folds (needs 2))" in out

    code = cli.main(
        ["predict", "--area", "Dubai Marina", "--kind", "apartment", "--status", "ready",
         "--size", "80", "--bedrooms", "1", "--building", "Tower 1-0", "--property-id", "p-9"]
    )  # fmt: skip
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["property_id"] == "p-9"
    assert set(result["forecast_3m"]) >= {"point", "ci_low", "ci_high", "confidence"}
    assert result["forecast_3y"]["status"] == "not_deployed"
    assert result["model_versions"]["forecast_3m"] == "1"


def test_evaluate_reports_no_registered_model_when_gate_passed(monkeypatch, tmp_path, capsys):
    quiet(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "load_champion", lambda name: (None, None))
    monkeypatch.setattr(
        cli, "latest_gates",
        lambda experiment: {h: {"status": "passed", "reason": "passed"} for h in cli.HORIZONS},
    )  # fmt: skip
    assert cli.main(["evaluate"]) == 0
    out = capsys.readouterr().out
    assert "3m: not deployed (no registered 3m model)" in out
    assert "3m: WARNING" not in out


def test_evaluate_warns_before_rescoring_a_blocked_champion(monkeypatch, tmp_path, capsys):
    from types import SimpleNamespace

    quiet(monkeypatch, tmp_path)
    model = SimpleNamespace(
        metadata={"test_cutoff": "2023-01-01", "test_end": "2023-02-01", "top_areas": [1]},
        predict_growth=lambda test: [0.0] * test.height,
    )
    champions = {"3m": (model, "5"), "1y": (None, None), "3y": (None, None)}
    monkeypatch.setattr(cli, "load_champion", lambda name: champions[name.rsplit("-", 1)[-1]])
    gates = {
        "3m": {"status": "failed", "reason": "3m resale MAPE 16.0% exceeds"},
        "1y": {"status": "unknown", "reason": ""},
        "3y": {"status": "unknown", "reason": ""},
    }
    monkeypatch.setattr(cli, "latest_gates", lambda experiment: gates)
    frame = pl.DataFrame({"instance_date": [date(2023, 1, 15)], "growth_3m": [0.01]})
    monkeypatch.setattr(
        cli, "_dataset", lambda config, stages: (frame, None, None, None, date(2023, 3, 1))
    )
    monkeypatch.setattr(cli, "usable_rows", lambda frame, horizon: frame)
    monkeypatch.setattr(
        cli, "segment_table",
        lambda *a, **k: pl.DataFrame(
            {"segment": ["all"], "model": ["model"], "mape": [0.1], "median_ape": [0.1], "rows": [1]}
        ),
    )  # fmt: skip
    monkeypatch.setattr(cli, "format_table", lambda name, table: ["  fake table line"])
    monkeypatch.setattr(cli, "baseline_growth", lambda test, horizon: {})
    assert cli.main(["evaluate"]) == 0
    out = capsys.readouterr().out
    assert "3m: WARNING champion would not be served (3m resale MAPE 16.0% exceeds)" in out
    assert "3m champion v5" in out
    assert "fake table line" in out
    assert "1y: not deployed (no registered 1y model)" in out


def test_train_exits_2_when_nothing_passes(monkeypatch, tmp_path, capsys):
    from types import SimpleNamespace

    quiet(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "load_rows", fake_load_rows(HISTORY))
    result = SimpleNamespace(
        horizon="3m", status="failed", reasons=("3m resale MAPE 16.0% exceeds the 15% gate",),
        folds=[], table=None, fold_scores=None, coverage={}, upper=None, seconds=1.0,
    )  # fmt: skip
    summary = SimpleNamespace(
        device="cpu", run_id="r1", results={"3m": result}, versions={}, seconds=2.0
    )
    seen = {}

    def fake_run(frame, report, quality, projects, data_end, config, **kwargs):
        seen.update(trials=config.n_trials, kwargs=kwargs)
        return summary

    monkeypatch.setattr(cli, "run_training", fake_run)
    assert cli.main(["train", "--trials", "3", "--device", "cpu", "--no-register"]) == 2
    assert seen == {"trials": 3, "kwargs": {"device": "cpu", "register": False}}
    captured = capsys.readouterr()
    assert "3m: failed" in captured.out
    assert "3m resale MAPE 16.0% exceeds the 15% gate" in captured.out
    assert "No horizon passed its gate; nothing was registered." in captured.err


def test_train_reports_errors(monkeypatch, tmp_path, capsys):
    quiet(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "load_rows", fake_load_rows(HISTORY))

    def boom(*args, **kwargs):
        raise RuntimeError("GPU lost")

    monkeypatch.setattr(cli, "run_training", boom)
    assert cli.main(["train"]) == 1
    assert "Training failed: RuntimeError: GPU lost" in capsys.readouterr().err


def test_predict_reports_invalid_input(monkeypatch, tmp_path, capsys):
    from models.price.predictor import PriceInputError

    quiet(monkeypatch, tmp_path)

    class Refuses:
        @classmethod
        def from_registry(cls, settings, config):
            return cls()

        def forecast(self, request):
            raise PriceInputError("area", "unknown area 'Atlantis'")

    monkeypatch.setattr(cli, "Forecaster", Refuses)
    code = cli.main(["predict", "--area", "Atlantis", "--kind", "apartment", "--status", "ready",
                     "--size", "100"])  # fmt: skip
    assert code == 1
    assert "Invalid input: area: unknown area 'Atlantis'" in capsys.readouterr().err
