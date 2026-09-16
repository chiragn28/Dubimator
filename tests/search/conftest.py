import pytest
from search_fixtures import build_search_listings, seed_search_db


@pytest.fixture
def search_listings():
    return build_search_listings()


@pytest.fixture
def search_db(pg_test_db, search_listings):
    corpus_run_id = seed_search_db(pg_test_db, search_listings)
    return pg_test_db, search_listings, corpus_run_id


@pytest.fixture
def fake_embedder():
    from listings.embed import FakeEmbedder

    return FakeEmbedder()


@pytest.fixture
def temp_mlflow(tmp_path, monkeypatch):
    """Throwaway MLflow tracking + registry store. Never the real server, never ./mlruns."""
    import mlflow

    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr("mlflow.tracking._tracking_service.utils._tracking_uri", uri)
    monkeypatch.setattr("mlflow.tracking.fluent._active_experiment_id", None)
    yield {"tracking_uri": uri, "artifact_location": (tmp_path / "artifacts").as_uri()}
    while mlflow.active_run():
        mlflow.end_run()
