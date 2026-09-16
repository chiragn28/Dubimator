import json

import mlflow
import pytest

from ingestion.config import DbSettings
from search.__main__ import main


@pytest.fixture
def cli_env(search_db, temp_mlflow, monkeypatch, tmp_path):
    """Point DbSettings.from_env() at the seeded test database, and MLflow at a temp store."""
    settings, _, _ = search_db
    for name, value in {
        "POSTGRES_HOST": settings.host,
        "POSTGRES_PORT": str(settings.port),
        "POSTGRES_USER": settings.user,
        "POSTGRES_PASSWORD": settings.password,
        "POSTGRES_DB": settings.dbname,
    }.items():
        monkeypatch.setenv(name, value)
    assert DbSettings.from_env().dbname == "zestimator_test"
    # create the experiment up front so the CLI never falls back to ./mlruns for artifacts
    mlflow.create_experiment("search-ranking", artifact_location=temp_mlflow["artifact_location"])
    return tmp_path / "data"


def test_the_cli_drives_the_whole_pipeline(cli_env, capsys):
    data = str(cli_env)
    assert main(["queries", "--n", "150", "--fake", "--data-dir", data]) == 0
    out = capsys.readouterr()
    assert "150 queries" in out.out
    assert "value features were SKIPPED" in out.err  # no price model in the temp registry
    stats = json.loads((cli_env / "queries_stats.json").read_text(encoding="utf-8"))
    assert stats["queries"] == 150 and stats["value_features_skipped"] == 1.0

    assert main(["train", "--trials", "1", "--device", "cpu", "--data-dir", data]) == 0
    out = capsys.readouterr().out
    assert "baseline_fused" in out and "gate:" in out
    runs = mlflow.search_runs(experiment_names=["search-ranking"], output_format="list")
    assert len(runs) == 1
    train_run = runs[0]
    assert train_run.info.run_name == "search-train"
    assert {"xgboost.ndcg_at_10", "baseline_fused.ndcg_at_10", "gate.passed"} <= set(
        train_run.data.metrics
    )
    assert train_run.data.params["winner"] in {"xgboost", "lightgbm"}
    artifacts = {a.path for a in mlflow.MlflowClient().list_artifacts(train_run.info.run_id)}
    assert {"ndcg_comparison.png", "per_query_report.csv", "timings.json"} <= artifacts
    passed = train_run.data.metrics["gate.passed"] == 1.0
    assert (train_run.data.params["registered_version"] != "none") is passed

    assert main(["query", "2 bed apartment in Dubai Marina", "--fake"]) == 0
    out = capsys.readouterr().out
    assert "Dubai Marina" in out and "area ✓" in out

    status = main(["evaluate", "--data-dir", data])
    captured = capsys.readouterr()
    if passed:
        assert status == 0
        runs = mlflow.search_runs(experiment_names=["search-ranking"], output_format="list")
        assert {run.info.run_name for run in runs} == {"search-train", "search-evaluate"}
    else:
        assert status == 1 and "no champion registered" in captured.err

    timings = json.loads((cli_env / "stage_timings.json").read_text(encoding="utf-8"))
    assert {"queries", "train"} <= set(timings)


def test_train_refuses_a_stale_query_set(cli_env, capsys):
    data = str(cli_env)
    assert main(["queries", "--n", "60", "--fake", "--data-dir", data]) == 0
    settings = DbSettings.from_env()
    conn = settings.connect()
    try:
        with conn.cursor() as cur:  # a newer corpus run makes the stored queries stale
            cur.execute(
                "INSERT INTO listings.corpus_runs (seed, photo_dataset_sha, counts) "
                "VALUES (1, 'x', '{}'::jsonb)"
            )
        conn.commit()
    finally:
        conn.close()
    capsys.readouterr()
    assert main(["train", "--trials", "1", "--device", "cpu", "--data-dir", data]) == 1
    assert "python -m search queries" in capsys.readouterr().err


def test_a_bad_command_line_is_an_argparse_error():
    with pytest.raises(SystemExit):
        main(["train", "--trials", "many"])
