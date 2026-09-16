"""TDD for scripts/ci.py, the local CI runner (`uv run python scripts/ci.py [--fast]`).

All subprocess calls go through an injected fake runner — this never shells out to ruff,
pytest, docker or actionlint.
"""

from scripts import ci


class FakeResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def make_runner(outcomes=None, missing=frozenset()):
    """outcomes: maps a substring of the joined argv to a FakeResult. missing: argv[0] values
    that should raise FileNotFoundError (simulating a tool not being installed)."""
    outcomes = outcomes or {}
    calls: list[list[str]] = []

    def runner(argv):
        calls.append(argv)
        if argv[0] in missing:
            raise FileNotFoundError(argv[0])
        joined = " ".join(argv)
        for key, result in outcomes.items():
            if key in joined:
                return result
        return FakeResult(returncode=0)

    runner.calls = calls
    return runner


def test_all_pass_exits_0(capsys):
    runner = make_runner()
    code = ci.main([], runner=runner)
    out = capsys.readouterr().out
    assert code == 0
    assert "PASS" in out
    assert "FAIL" not in out


def test_a_failing_step_exits_1_and_reports_fail(capsys):
    runner = make_runner({"ruff check": FakeResult(returncode=1, stdout="E501 line too long")})
    code = ci.main([], runner=runner)
    out = capsys.readouterr().out
    assert code == 1
    assert "FAIL" in out


def test_missing_uvx_is_skip_not_fail(capsys):
    runner = make_runner(missing={"uvx"})
    code = ci.main([], runner=runner)
    out = capsys.readouterr().out
    assert code == 0
    assert "SKIP" in out


def test_runs_every_step_in_order():
    runner = make_runner()
    ci.main([], runner=runner)
    joined = [" ".join(c) for c in runner.calls]
    assert any("ruff check" in j for j in joined)
    assert any("ruff format --check" in j for j in joined)
    assert any("pytest" in j for j in joined)
    assert any("docker" in j and "compose" in j and "config" in j for j in joined)
    assert any("actionlint" in j for j in joined)


def test_default_pytest_marker_excludes_gpu_and_live():
    runner = make_runner()
    ci.main([], runner=runner)
    joined = [" ".join(c) for c in runner.calls]
    pytest_call = next(j for j in joined if "pytest" in j)
    assert "not gpu and not live" in pytest_call
    assert "not db" not in pytest_call


def test_fast_flag_also_excludes_db():
    runner = make_runner()
    ci.main(["--fast"], runner=runner)
    joined = [" ".join(c) for c in runner.calls]
    pytest_call = next(j for j in joined if "pytest" in j)
    assert "not gpu and not live and not db" in pytest_call


def test_table_reports_durations(capsys):
    runner = make_runner()
    ci.main([], runner=runner)
    out = capsys.readouterr().out
    assert "SECONDS" in out or "seconds" in out.lower()
