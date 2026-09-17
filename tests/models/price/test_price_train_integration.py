import dataclasses
from datetime import date
from pathlib import Path

import mlflow
import polars as pl

from ingestion.pipeline import run_pipeline
from models.price.config import TrainConfig
from models.price.train import eval_filters, run_training

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
    model_name="dubimator-price-it",
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

    runs = mlflow.search_runs(experiment_names=["price-integration"])
    run_names = set(runs["tags.mlflow.runName"])
    assert {"b0-comps", "b1-lightgbm", "xgb-tune", "trial-000", "trial-001",
            "xgb-champion-eval", "xgb-production"} <= run_names  # fmt: skip

    client = mlflow.MlflowClient()
    run_ids = dict(zip(runs["tags.mlflow.runName"], runs["run_id"], strict=True))
    eval_run = client.get_run(run_ids["xgb-champion-eval"])
    prod_run = client.get_run(run_ids["xgb-production"])
    eval_artifacts = {artifact.path for artifact in client.list_artifacts(eval_run.info.run_id)}
    assert {"metrics.json", "conformal_val.json", "conformal_production.json",
            "feature_importance.png", "residuals_by_segment.png", "error_by_loc_level.png",
            "pred_vs_actual.png"} <= eval_artifacts  # fmt: skip
    assert "conformal.json" not in eval_artifacts
    # The production run carries the eval run's headline metrics, prefixed and tagged as such.
    assert prod_run.data.tags["metrics_source"] == "xgb-champion-eval"
    assert "test_clean.all.mdape" not in prod_run.data.metrics
    assert (
        prod_run.data.metrics["eval.test_clean.all.mdape"]
        == (eval_run.data.metrics["test_clean.all.mdape"])
    )
    for run in (eval_run, prod_run):
        assert {"lineage.ingest_run_id", "lineage.source_sha256"} <= set(run.data.params)
    num_rounds = int(eval_run.data.metrics["best_iteration"]) + 1
    assert int(prod_run.data.params["num_boost_round"]) == num_rounds

    model = mlflow.pyfunc.load_model("models:/dubimator-price-it@champion")
    predictor = model.unwrap_python_model().predictor
    assert predictor.bundle.metadata["num_boost_round"] == num_rounds
    run_uri = f"runs:/{eval_run.info.run_id}"
    production_conformal = mlflow.artifacts.load_dict(f"{run_uri}/conformal_production.json")
    val_conformal = mlflow.artifacts.load_dict(f"{run_uri}/conformal_val.json")
    assert predictor.bundle.conformal == production_conformal  # shipped: test-period errors
    assert "_pooled" in val_conformal  # reported coverage: val-calibrated quantiles
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


def test_eval_sets_stop_at_data_end():
    frame = pl.DataFrame(
        {
            "split": ["val", "test", "test", "test"],
            "is_clean": [True, True, False, False],
            "instance_date": [date(2022, 8, 1), date(2023, 3, 17), date(2023, 3, 1),
                              date(2023, 4, 2)],  # last row: stat-excluded, after DATA_END
        }
    )  # fmt: skip
    selected = {
        name: frame.filter(expr)["instance_date"].to_list()
        for name, expr in eval_filters(date(2023, 3, 17)).items()
    }
    assert selected == {
        "val": [date(2022, 8, 1)],
        "test_clean": [date(2023, 3, 17)],
        "test_honest": [date(2023, 3, 17), date(2023, 3, 1)],
    }


def test_failed_gate_registers_nothing(pg_test_db, temp_mlflow):
    summary = train(pg_test_db, temp_mlflow, dataclasses.replace(SMALL, gate_ratio=0.0))
    assert not summary.gate_passed
    assert summary.registered_version is None and summary.model_uri is None
    run_names = set(
        mlflow.search_runs(experiment_names=["price-integration"])["tags.mlflow.runName"]
    )
    assert "xgb-champion-eval" in run_names and "xgb-production" not in run_names
    registered = mlflow.MlflowClient().search_registered_models("name='dubimator-price-it'")
    assert registered == []
