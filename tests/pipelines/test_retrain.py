"""TDD for pipelines/retrain.py (Phase 9 scheduled retraining pipeline).

Uses a fake subprocess runner throughout — no real ingestion/training ever runs in tests.
Covers: stage order, stop-on-first-failure, gate_failed (exit code 2 from a training step)
continues, --only/--skip, the unchanged-source ingest skip, and the JSON report file.
The latest-ingested-hash reader is always faked, so no test touches the database.
"""

import hashlib
import json

import pytest

from pipelines import retrain


class FakeResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def make_runner(outcomes=None):
    """Fake `runner(argv) -> FakeResult`. `outcomes` maps a substring of the joined argv to the
    FakeResult to return for any call whose argv contains it; unmatched calls succeed (rc 0).
    Records every call for order/content assertions.
    """
    outcomes = outcomes or {}
    calls: list[list[str]] = []

    def runner(argv):
        calls.append(argv)
        joined = " ".join(argv)
        for key, result in outcomes.items():
            if key in joined:
                return result
        return FakeResult(returncode=0)

    runner.calls = calls
    return runner


@pytest.fixture(autouse=True)
def _no_database(monkeypatch):
    """The default reader queries Postgres; tests must never reach it."""

    def forbidden():
        raise AssertionError("the database reader must not be called in tests")

    monkeypatch.setattr(retrain, "latest_ingested_sha256", lambda: None)
    return forbidden


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "Transactions.csv"
    path.write_text("id\n1\n")
    return path


def test_stage_names_in_order():
    assert list(retrain.STAGE_NAMES) == ["ingest", "price", "listings", "search", "forecast"]


def test_build_stages_matches_stage_names(csv_path):
    stages = retrain.build_stages(csv_path)
    assert [s.name for s in stages] == list(retrain.STAGE_NAMES)


def test_run_pipeline_runs_every_command_in_order(csv_path):
    runner = make_runner()
    result = retrain.run_pipeline(retrain.build_stages(csv_path), runner)

    assert [r.name for r in result.reports] == list(retrain.STAGE_NAMES)
    assert all(r.status == "ok" for r in result.reports)
    assert result.failed is False

    # ingest, price train, listings detect, search queries, search train, forecast build, forecast train
    assert len(runner.calls) == 7
    joined = [" ".join(c) for c in runner.calls]
    assert "-m ingestion --csv" in joined[0] and str(csv_path) in joined[0]
    assert "-m models.price train" in joined[1]
    assert "-m listings detect" in joined[2]
    assert "-m search queries" in joined[3]
    assert "-m search train" in joined[4]
    assert "-m models.forecast build" in joined[5]
    assert "-m models.forecast train" in joined[6]


def test_ingest_skips_when_source_missing(tmp_path):
    missing = tmp_path / "does-not-exist.csv"
    runner = make_runner()
    result = retrain.run_pipeline(retrain.build_stages(missing), runner)

    ingest_report = result.reports[0]
    assert ingest_report.status == "skipped"
    assert not any("ingestion" in " ".join(c) for c in runner.calls)
    # The rest of the pipeline still runs.
    assert [r.name for r in result.reports] == list(retrain.STAGE_NAMES)
    assert result.failed is False


def test_stops_at_first_failure(csv_path):
    runner = make_runner({"models.price train": FakeResult(returncode=1, stderr="boom")})
    result = retrain.run_pipeline(retrain.build_stages(csv_path), runner)

    assert [r.name for r in result.reports] == ["ingest", "price"]
    assert result.reports[-1].status == "failed"
    assert result.reports[-1].exit_code == 1
    assert result.failed is True
    # listings/search/forecast never ran.
    assert not any("listings" in " ".join(c) for c in runner.calls)


def test_gate_failed_records_gate_failed_and_continues(csv_path):
    runner = make_runner({"models.price train": FakeResult(returncode=2)})
    result = retrain.run_pipeline(retrain.build_stages(csv_path), runner)

    assert [r.name for r in result.reports] == list(retrain.STAGE_NAMES)
    price_report = result.reports[1]
    assert price_report.status == "gate_failed"
    assert price_report.exit_code == 2
    # A gate failure is a legitimate outcome, not a pipeline failure.
    assert result.failed is False


def test_gate_failed_in_multi_command_stage(csv_path):
    # forecast has two commands (build, train); a gate failure on the second must not re-run
    # or skip the first, and must not stop later stages (there are none after forecast here,
    # so assert directly on this stage's report).
    runner = make_runner({"models.forecast train": FakeResult(returncode=2)})
    result = retrain.run_pipeline(retrain.build_stages(csv_path), runner)
    forecast_report = result.reports[-1]
    assert forecast_report.status == "gate_failed"
    assert forecast_report.exit_code == 2
    assert result.failed is False


def test_select_stages_only(csv_path):
    stages = retrain.build_stages(csv_path)
    selected = retrain.select_stages(stages, only=["price", "search"], skip=None)
    assert [s.name for s in selected] == ["price", "search"]


def test_select_stages_skip(csv_path):
    stages = retrain.build_stages(csv_path)
    selected = retrain.select_stages(stages, only=None, skip=["listings"])
    assert [s.name for s in selected] == ["ingest", "price", "search", "forecast"]


def test_write_report(tmp_path):
    reports = [
        retrain.StageReport("ingest", "ok", 1.5, 0, ["line1", "line2"]),
        retrain.StageReport("price", "gate_failed", 2.25, 2, ["gate failed"]),
    ]
    path = tmp_path / "data" / "pipelines" / "retrain_20260101T000000Z.json"
    retrain.write_report(
        path,
        reports,
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:04+00:00",
    )

    payload = json.loads(path.read_text())
    assert payload["started_at"] == "2026-01-01T00:00:00+00:00"
    assert payload["finished_at"] == "2026-01-01T00:00:04+00:00"
    assert payload["stages"][0] == {
        "name": "ingest",
        "status": "ok",
        "seconds": 1.5,
        "exit_code": 0,
        "output_tail": ["line1", "line2"],
    }
    assert payload["stages"][1]["status"] == "gate_failed"


def test_output_tail_keeps_last_20_lines(csv_path):
    stdout = "\n".join(f"line{i}" for i in range(50))
    runner = make_runner({"models.price train": FakeResult(returncode=0, stdout=stdout)})
    result = retrain.run_pipeline(retrain.build_stages(csv_path), runner)
    price_report = result.reports[1]
    assert len(price_report.output_tail) == 20
    assert price_report.output_tail[0] == "line30"
    assert price_report.output_tail[-1] == "line49"


def test_main_dry_run_prints_plan_and_does_not_run_anything(csv_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(retrain, "default_runner", lambda argv: calls.append(argv))
    code = retrain.main(["retrain", "--dry-run", "--csv", str(csv_path)])
    out = capsys.readouterr().out

    assert code == 0
    assert calls == []
    for name in retrain.STAGE_NAMES:
        assert name in out
    assert "models.forecast train" in out


def test_main_writes_report_and_returns_1_on_failure(tmp_path, csv_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Transactions.csv").write_text("id\n1\n")
    runner = make_runner({"models.price train": FakeResult(returncode=1, stderr="boom")})
    code = retrain.main(["retrain", "--csv", str(csv_path)], runner=runner)

    assert code == 1
    report_files = list((tmp_path / "data" / "pipelines").glob("retrain_*.json"))
    assert len(report_files) == 1
    payload = json.loads(report_files[0].read_text())
    assert payload["stages"][-1]["status"] == "failed"


def test_main_returns_0_when_only_a_gate_fails(tmp_path, csv_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = make_runner({"models.price train": FakeResult(returncode=2)})
    code = retrain.main(["retrain", "--csv", str(csv_path)], runner=runner)
    assert code == 0


def test_main_only_and_skip_are_mutually_composable(tmp_path, csv_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = make_runner()
    code = retrain.main(
        ["retrain", "--csv", str(csv_path), "--only", "price", "search", "--skip", "search"],
        runner=runner,
    )
    assert code == 0
    joined = [" ".join(c) for c in runner.calls]
    assert any("models.price train" in j for j in joined)
    assert not any("search" in j for j in joined)


USAGE_ERROR = (
    "usage: python -m models.price [-h] {train}\n"
    "python -m models.price: error: argument command: invalid choice: 'trian'\n"
)


@pytest.mark.parametrize(
    "failing",
    ["-m ingestion", "models.forecast build", "listings detect", "search queries", "search train"],
)
def test_exit_2_outside_training_steps_is_a_failure(csv_path, failing):
    runner = make_runner({failing: FakeResult(returncode=2)})
    result = retrain.run_pipeline(retrain.build_stages(csv_path), runner)

    assert result.reports[-1].status == "failed"
    assert result.reports[-1].exit_code == 2
    assert result.failed is True


def test_exit_2_with_an_argparse_usage_error_is_a_failure(csv_path):
    runner = make_runner({"models.price train": FakeResult(returncode=2, stderr=USAGE_ERROR)})
    result = retrain.run_pipeline(retrain.build_stages(csv_path), runner)

    assert [r.name for r in result.reports] == ["ingest", "price"]
    assert result.reports[-1].status == "failed"
    assert result.failed is True


def test_exit_2_from_forecast_train_after_build_is_gate_failed(csv_path):
    runner = make_runner({"models.forecast train": FakeResult(returncode=2, stdout="no gate")})
    result = retrain.run_pipeline(retrain.build_stages(csv_path), runner)

    assert result.reports[-1].status == "gate_failed"
    joined = [" ".join(c) for c in runner.calls]
    assert any("models.forecast build" in j for j in joined)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ingest_skip_reason_when_source_is_unchanged(csv_path):
    reason = retrain.ingest_skip_reason(csv_path, lambda: _sha(csv_path))
    assert reason is not None
    assert "unchanged" in reason


def test_ingest_skip_reason_none_when_source_changed_or_never_ingested(csv_path):
    assert retrain.ingest_skip_reason(csv_path, lambda: "0" * 64) is None
    assert retrain.ingest_skip_reason(csv_path, lambda: None) is None


def test_ingest_skip_reason_runs_ingest_when_the_reader_fails(csv_path):
    def broken():
        raise OSError("connection refused")

    assert retrain.ingest_skip_reason(csv_path, broken) is None


def test_ingest_skip_reason_missing_file_does_not_read_the_database(tmp_path, _no_database):
    reason = retrain.ingest_skip_reason(tmp_path / "missing.csv", _no_database)
    assert reason is not None and "not found" in reason


def test_ingest_skip_reason_uses_the_ingestion_hash_helper(csv_path, monkeypatch):
    monkeypatch.setattr(retrain, "file_sha256", lambda path: "abc")
    assert retrain.ingest_skip_reason(csv_path, lambda: "abc") is not None


def test_unchanged_source_skips_ingest_and_continues(csv_path):
    runner = make_runner()
    stages = retrain.build_stages(csv_path, read_latest_sha=lambda: _sha(csv_path))
    result = retrain.run_pipeline(stages, runner)

    assert result.reports[0].status == "skipped"
    assert "unchanged" in result.reports[0].output_tail[0]
    assert not any("ingestion" in " ".join(c) for c in runner.calls)
    assert [r.name for r in result.reports] == list(retrain.STAGE_NAMES)
    assert result.failed is False


def test_main_uses_the_default_reader(tmp_path, csv_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(retrain, "latest_ingested_sha256", lambda: _sha(csv_path))
    runner = make_runner()
    assert retrain.main(["retrain", "--csv", str(csv_path)], runner=runner) == 0
    assert not any("ingestion" in " ".join(c) for c in runner.calls)


def test_dry_run_checks_the_skip_once(csv_path, monkeypatch, capsys):
    reads = []
    monkeypatch.setattr(
        retrain, "latest_ingested_sha256", lambda: reads.append(1) or _sha(csv_path)
    )
    assert retrain.main(["retrain", "--dry-run", "--csv", str(csv_path)]) == 0
    assert "SKIP" in capsys.readouterr().out
    assert len(reads) == 1
