import dataclasses
from pathlib import Path

import mlflow
import polars as pl

from ingestion.pipeline import run_pipeline
from models.price.config import TrainConfig
from models.price.train import run_training

FIXTURE = Path("tests/fixtures/price_sample.csv")
SMALL = dataclasses.replace(
    TrainConfig(),
    min_segment_rows=10,
    min_index_sales=3,
    oof_folds=3,
    comps_min_n=2.0,
    bounds_min_area_n=5.0,
    min_conformal_rows=20,
    n_trials=2,
    max_rounds=200,
    early_stopping_rounds=20,
    gate_ratio=None,
    experiment="price-integration",
    model_name="zestimator-price-it",
)


def train(settings, temp_mlflow, config):
    run_pipeline(FIXTURE, settings)
    return run_training(
        settings,
        config,
        device="cpu",
        register=True,
        tracking_uri=temp_mlflow["tracking_uri"],
        artifact_location=temp_mlflow["artifact_location"],
        log=lambda _message: None,
    )


def test_end_to_end_training_registers_a_working_champion(pg_test_db, temp_mlflow):
    summary = train(pg_test_db, temp_mlflow, SMALL)

    assert summary.gate_passed and summary.registered_version == "1"
    assert set(summary.metrics) == {"b0-comps", "b1-lightgbm", "xgb-champion-eval"}
    for metrics in summary.metrics.values():
        assert metrics["test_clean.all.n"] > 0
        assert 0.0 < metrics["test_clean.all.mdape"] < 1.0
        assert metrics["test_honest.all.n"] >= metrics["test_clean.all.n"]
    champion = summary.metrics["xgb-champion-eval"]
    assert 0.0 <= champion["test_clean.all.coverage_80"] <= 1.0
    assert any(key.startswith("test_clean.segment.") for key in champion)

    run_names = set(
        mlflow.search_runs(experiment_names=["price-integration"])["tags.mlflow.runName"]
    )
    assert {"b0-comps", "b1-lightgbm", "xgb-tune", "trial-000", "trial-001",
            "xgb-champion-eval", "xgb-production"} <= run_names  # fmt: skip

    model = mlflow.pyfunc.load_model("models:/zestimator-price-it@champion")
    predictor = model.unwrap_python_model().predictor
    areas = predictor.bundle.priors.stats["area"].filter(pl.col("property_type") == "unit")
    busiest_area = int(areas.sort("sum_w", descending=True)["area_id"][0])
    estimate = predictor.predict_one(
        {"area_id": busiest_area, "property_kind": "apartment", "status": "ready",
         "size_sqm": 90.0, "bedrooms": 2}
    )  # fmt: skip
    assert estimate.estimate_aed > 0
    assert estimate.range_80[0] < estimate.estimate_aed < estimate.range_80[1]
    assert estimate.as_of == predictor.bundle.data_end
    assert estimate.model_version == summary.model_uri.split("/")[1]


def test_failed_gate_registers_nothing(pg_test_db, temp_mlflow):
    summary = train(pg_test_db, temp_mlflow, dataclasses.replace(SMALL, gate_ratio=0.0))
    assert not summary.gate_passed
    assert summary.registered_version is None and summary.model_uri is None
    run_names = set(
        mlflow.search_runs(experiment_names=["price-integration"])["tags.mlflow.runName"]
    )
    assert "xgb-champion-eval" in run_names and "xgb-production" not in run_names
