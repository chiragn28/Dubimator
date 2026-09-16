import dataclasses

import numpy as np
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
    assert set(passed.fold_scores["role"].unique()) <= {"score", "tune"}
    assert passed.model.metadata["test_cutoff"] == passed.folds[-1].cutoff.isoformat()


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
