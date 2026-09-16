import dataclasses
from datetime import timedelta

import numpy as np
import polars as pl
import pytest
from forecast_fixtures import fake_load_rows, prepared_history, project_table

from models.forecast.config import FEATURES, HORIZON_SPECS, ForecastConfig
from models.forecast.features import build_dataset
from models.forecast.train import run_horizon, run_training

FAST = ForecastConfig(
    n_trials=1,
    n_estimators=200,
    learning_rate=0.1,
    tune_folds=2,
    min_test_rows=50,
    n_bootstrap=200,
    device="cpu",
)
THREE_M = HORIZON_SPECS["3m"]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    rows, data_end = prepared_history(per_building_per_week=4)
    projects = project_table(tmp_path_factory.mktemp("projects"))
    frame, report = build_dataset(rows, data_end, projects)
    return frame, report, projects, data_end


@pytest.fixture(scope="module")
def passed(dataset):
    frame, _, _, data_end = dataset
    return run_horizon(frame, THREE_M, "cpu", FAST, data_end)


def test_a_learnable_horizon_passes_its_gate(passed):
    assert passed.status == "passed", passed.reasons
    assert passed.reasons == ()
    assert [fold.role for fold in passed.folds][-1] == "test"
    assert passed.table is not None and passed.table.height > 0
    assert 0.5 <= passed.coverage["all"] <= 1.0
    assert passed.model is not None and passed.model.rounds >= 1
    metrics = passed.metrics()
    assert metrics["gate.3m.passed"] == 1.0
    assert "3m.test.all.model.mape" in metrics
    assert "3m.folds.area_trend.mape_mean" in metrics
    assert set(passed.fold_scores["role"].unique()) <= {"score", "tune", "gap"}
    assert "gap" in set(passed.fold_scores["role"])
    assert passed.model.metadata["test_cutoff"] == passed.folds[-1].cutoff.isoformat()


def test_gap_folds_are_scored_but_never_tune_or_calibrate(dataset, monkeypatch):
    from models.forecast import train as train_module
    from models.forecast.folds import fold_split

    frame, _, _, data_end = dataset
    calls, calibration = [], {}
    run_fold = train_module._run_fold
    fit_intervals = train_module.fit_intervals

    def spy_fold(frame, horizon, fold, *args):
        calls.append(fold.role)
        return run_fold(frame, horizon, fold, *args)

    def spy_intervals(segments, errors, config):
        calibration["rows"] = len(segments)
        return fit_intervals(segments, errors, config)

    monkeypatch.setattr(train_module, "_run_fold", spy_fold)
    monkeypatch.setattr(train_module, "fit_intervals", spy_intervals)
    config = dataclasses.replace(FAST, n_trials=2)
    result = run_horizon(frame, THREE_M, "cpu", config, data_end)
    folds = result.folds
    test = folds[-1]
    tuning = [fold for fold in folds if fold.role == "tune"]
    gaps = [fold for fold in folds if fold.role == "gap"]
    assert gaps and tuning
    assert all(f.end + timedelta(days=THREE_M.end_days) <= test.cutoff for f in tuning)
    assert calls.count("gap") == len(gaps)  # scored once, never inside the Optuna objective
    assert calls.count("tune") == len(tuning) * (config.n_trials + 1)
    expected = sum(fold_split(frame, THREE_M, fold)[1].height for fold in tuning)
    assert calibration["rows"] == expected
    scored = result.fold_scores.filter(pl.col("model") == "model")
    assert sorted(scored.filter(pl.col("role") == "gap")["fold"].to_list()) == [
        fold.index for fold in gaps
    ]


def test_a_strict_gate_fails_but_still_reports(dataset):
    frame, _, _, data_end = dataset
    strict = dataclasses.replace(THREE_M, mape_gate=0.0)
    result = run_horizon(frame, strict, "cpu", FAST, data_end)
    assert result.status == "failed"
    assert any("exceeds the 0% gate" in reason for reason in result.reasons)
    assert result.table is not None
    assert result.metrics()["gate.3m.passed"] == 0.0


def test_a_short_history_is_insufficient(dataset):
    frame, _, _, data_end = dataset
    result = run_horizon(frame, HORIZON_SPECS["3y"], "cpu", FAST, data_end)
    assert result.status == "insufficient_data"
    assert result.reasons[0].startswith("3y: only ")
    assert result.model is None and result.table is None
    assert result.metrics()["gate.3y.passed"] == 0.0


def test_the_model_round_trips_and_explains_itself(passed, dataset, tmp_path):
    from models.forecast.model import ForecastModel

    frame = dataset[0].filter(dataset[0]["growth_3m"].is_not_null()).head(50)
    model = passed.model
    loaded = ForecastModel.load(model.save(tmp_path / "model"))
    assert np.allclose(loaded.predict_growth(frame), model.predict_growth(frame), atol=1e-6)
    contributions = loaded.contributions(frame)
    assert contributions.shape == (50, len(FEATURES) + 1)
    assert np.allclose(contributions.sum(axis=1), loaded.predict_growth(frame), atol=1e-4)
    assert loaded.half_width("ready_unit") > 0
    assert loaded.area_rows == model.area_rows
    assert loaded.importance().columns == ["feature", "gain"]


def test_run_training_registers_only_passing_horizons(dataset, temp_mlflow):
    from models.forecast.model import latest_gates, load_champion

    frame, report, projects, data_end = dataset
    quality = fake_load_rows(prepared_history(per_building_per_week=4))(None, None)[1]
    strict_year = dataclasses.replace(HORIZON_SPECS["1y"], mape_gate=0.0)
    summary = run_training(
        frame, report, quality, projects, data_end, FAST,
        device="cpu", horizons=(THREE_M, strict_year),
        tracking_uri=temp_mlflow["tracking_uri"],
        artifact_location=temp_mlflow["artifact_location"],
    )  # fmt: skip
    assert summary.versions == {"3m": "1"}
    assert summary.results["1y"].status == "failed"
    model, version = load_champion("zestimator-forecast-3m")
    assert version == "1" and model is not None
    assert model.horizon == "3m"
    assert load_champion("zestimator-forecast-1y") == (None, None)
    gates = latest_gates(FAST.experiment)
    assert gates["3m"] == {"status": "passed", "reason": "passed"}
    assert gates["1y"]["status"] == "failed"
    assert "exceeds the 0% gate" in gates["1y"]["reason"]
    assert gates["3y"]["status"] == "unknown"
    tags = mlflow_tags(summary.run_id)
    assert tags["gate_run"] == "true"
    assert "gate.3m.champion_removed" not in tags  # only horizons that did not pass
    assert tags["gate.1y.champion_removed"] == "false"  # there was no 1y model to remove

    strict_three = dataclasses.replace(THREE_M, mape_gate=0.0)
    again = run_training(
        frame, report, quality, projects, data_end, FAST,
        device="cpu", horizons=(strict_three,),
        tracking_uri=temp_mlflow["tracking_uri"],
        artifact_location=temp_mlflow["artifact_location"],
    )  # fmt: skip
    assert again.results["3m"].status == "failed"
    assert again.versions == {}
    assert load_champion("zestimator-forecast-3m") == (None, None)
    assert mlflow_tags(again.run_id)["gate.3m.champion_removed"] == "true"
    gates = latest_gates(FAST.experiment)
    assert gates["3m"]["status"] == "failed"
    assert gates["1y"]["status"] == "unknown"  # this run trained only 3m


def mlflow_tags(run_id):
    import mlflow

    return mlflow.get_run(run_id).data.tags


def test_latest_gates_reads_only_finished_training_runs(temp_mlflow):
    import mlflow
    from mlflow.tracking import MlflowClient

    from models.forecast.model import latest_gates

    experiment = mlflow.create_experiment(
        "gates-only", artifact_location=temp_mlflow["artifact_location"]
    )
    client = MlflowClient()
    finished = client.create_run(experiment, tags={"gate_run": "true", "gate.3m.status": "passed"})
    client.set_terminated(finished.info.run_id, "FINISHED")
    other = client.create_run(experiment, tags={"gate.3m.status": "failed"})
    client.set_terminated(other.info.run_id, "FINISHED")  # not a training run
    crashed = client.create_run(experiment, tags={"gate_run": "true", "gate.3m.status": "failed"})
    client.set_terminated(crashed.info.run_id, "FAILED")
    client.create_run(experiment, tags={"gate_run": "true", "gate.3m.status": "failed"})  # RUNNING
    assert latest_gates("gates-only")["3m"]["status"] == "passed"
