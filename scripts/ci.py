"""Local CI runner: `uv run python scripts/ci.py [--fast]`.

Runs the same checks as `.github/workflows/ci.yml`'s `lint` and `test` jobs, in order, plus a
Compose config sanity check and actionlint on the workflow files. Prints a PASS/FAIL/SKIP table
with durations and exits 1 if anything failed (SKIP never fails the run).

`--fast` also excludes the `db` marker (tests using the `pg_test_db` fixture — a live Postgres
test database), for a quick check when the Docker stack isn't up.
"""

import argparse
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field

Runner = Callable[[list[str]], subprocess.CompletedProcess]

# Heuristics for "actionlint couldn't be fetched" (offline) vs. "actionlint ran and found
# problems" (a real failure) — both come back as a non-zero exit from `uvx`.
_OFFLINE_MARKERS = (
    "failed to fetch",
    "failed to download",
    "could not resolve host",
    "temporary failure in name resolution",
    "network is unreachable",
    "no address associated",
    "connection refused",
    "unable to fetch",
    "name or service not known",
)


def default_runner(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, check=False)


@dataclass
class StepResult:
    name: str
    status: str  # "PASS" | "FAIL" | "SKIP"
    seconds: float
    detail: str = field(default="")


def _run_step(name: str, argv: list[str], runner: Runner) -> StepResult:
    start = time.monotonic()
    try:
        result = runner(argv)
    except FileNotFoundError as exc:
        return StepResult(name, "SKIP", time.monotonic() - start, f"{argv[0]} not found: {exc}")
    seconds = time.monotonic() - start
    status = "PASS" if result.returncode == 0 else "FAIL"
    detail = "" if status == "PASS" else ((result.stdout or "") + (result.stderr or ""))[-800:]
    return StepResult(name, status, seconds, detail)


def _run_actionlint_step(runner: Runner) -> StepResult:
    name = "actionlint"
    start = time.monotonic()
    try:
        result = runner(["uvx", "--from", "actionlint-py", "actionlint"])
    except FileNotFoundError as exc:
        return StepResult(name, "SKIP", time.monotonic() - start, f"uvx not found: {exc}")
    seconds = time.monotonic() - start
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0 and any(marker in output.lower() for marker in _OFFLINE_MARKERS):
        return StepResult(name, "SKIP", seconds, "actionlint unavailable offline: " + output[-400:])
    status = "PASS" if result.returncode == 0 else "FAIL"
    return StepResult(name, status, seconds, "" if status == "PASS" else output[-800:])


def _print_table(results: list[StepResult]) -> None:
    name_width = max((len(r.name) for r in results), default=4)
    print(f"{'STEP':<{name_width}}  STATUS  SECONDS")
    for r in results:
        print(f"{r.name:<{name_width}}  {r.status:<6}  {r.seconds:7.2f}")
        if r.status == "FAIL" and r.detail:
            for line in r.detail.splitlines()[-15:]:
                print(f"    {line}")


def main(argv: list[str] | None = None, runner: Runner | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scripts/ci.py", description="Run the local CI checks and print a pass/fail table."
    )
    parser.add_argument(
        "--fast", action="store_true", help="also exclude the `db` marker (skips Postgres tests)"
    )
    args = parser.parse_args(argv)
    active_runner = runner if runner is not None else default_runner

    pytest_marker = "not gpu and not live"
    if args.fast:
        pytest_marker += " and not db"

    results: list[StepResult] = [
        _run_step("ruff check", ["uv", "run", "ruff", "check", "."], active_runner),
        _run_step(
            "ruff format --check", ["uv", "run", "ruff", "format", "--check", "."], active_runner
        ),
        _run_step(
            "pytest",
            ["uv", "run", "pytest", "-q", "-W", "error", "-m", pytest_marker],
            active_runner,
        ),
        _run_step("docker compose config", ["docker", "compose", "config", "-q"], active_runner),
        _run_actionlint_step(active_runner),
    ]

    _print_table(results)
    return 1 if any(r.status == "FAIL" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
