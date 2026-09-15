import mlflow
import pandas as pd

from models.price.predictor import save_bundle
from models.price.pyfunc import pip_requirements
from models.price.registry import configure, log_metrics, log_price_model, register_champion

REQUESTS = pd.DataFrame(
    [
        {"area": "Dubai Marina", "property_kind": "apartment", "status": "ready",
         "size_sqm": 100.0, "bedrooms": 2},
        {"area": "JBR", "property_kind": "apartment", "status": "ready",
         "size_sqm": 80.0, "bedrooms": None},
    ]
)  # fmt: skip


def test_configure_points_mlflow_at_the_temporary_store(temp_mlflow):
    configure("price-test", temp_mlflow["tracking_uri"], temp_mlflow["artifact_location"])
    assert mlflow.get_tracking_uri() == temp_mlflow["tracking_uri"]
    experiment = mlflow.get_experiment_by_name("price-test")
    assert experiment.artifact_location == temp_mlflow["artifact_location"]


def test_log_register_load_and_predict(temp_mlflow, tiny_bundle, tmp_path):
    configure("price-test", temp_mlflow["tracking_uri"], temp_mlflow["artifact_location"])
    save_bundle(tiny_bundle(), tmp_path / "model_dir")
    with mlflow.start_run(run_name="xgb-production"):
        log_metrics({"test_clean.all.mdape": 0.12, "not_finite": float("nan")})
        model_uri = log_price_model(tmp_path / "model_dir")
    assert register_champion(model_uri, "zestimator-price-test") == "1"

    model = mlflow.pyfunc.load_model("models:/zestimator-price-test@champion")
    out = model.predict(REQUESTS)
    assert len(out) == 2
    assert (out["estimate_aed"] > 0).all()
    assert model.unwrap_python_model().predictor.model_version == "run-abc"

    runs = mlflow.search_runs(experiment_names=["price-test"])
    assert runs["metrics.test_clean.all.mdape"].iloc[0] == 0.12
    assert "metrics.not_finite" not in runs.columns


def test_registering_again_moves_the_champion_alias(temp_mlflow, tiny_bundle, tmp_path):
    configure("price-test", temp_mlflow["tracking_uri"], temp_mlflow["artifact_location"])
    save_bundle(tiny_bundle(), tmp_path / "model_dir")
    for expected in ("1", "2"):
        with mlflow.start_run():
            uri = log_price_model(tmp_path / "model_dir")
        assert register_champion(uri, "zestimator-price-test") == expected
    alias = mlflow.MlflowClient().get_model_version_by_alias("zestimator-price-test", "champion")
    assert str(alias.version) == "2"


def test_pip_requirements_are_exact_pins():
    requirements = pip_requirements()
    assert any(line.startswith("xgboost==") for line in requirements)
    assert all("==" in line for line in requirements)
