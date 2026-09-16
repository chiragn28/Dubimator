import pytest
from forecast_fixtures import build_history


@pytest.fixture(scope="session")
def history():
    return build_history()


@pytest.fixture
def temp_mlflow(tmp_path, monkeypatch):
    """Throwaway MLflow store; touched during setup so alembic's logging reset happens here."""
    import mlflow

    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr("mlflow.tracking._tracking_service.utils._tracking_uri", uri)
    monkeypatch.setattr("mlflow.tracking.fluent._active_experiment_id", None)
    mlflow.search_experiments()
    yield {"tracking_uri": uri, "artifact_location": (tmp_path / "artifacts").as_uri()}
    while mlflow.active_run():
        mlflow.end_run()
