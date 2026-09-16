"""`python -m pipelines retrain`: the scheduled retraining pipeline.

Runs ingest -> price -> listings -> search -> forecast, each stage as one or more `python -m
...` subprocesses (via `sys.executable`, so it works inside `uv run` too). Stops at the first
real failure. A stage that exits 2 (the gate-failure convention used by `models.price train`
and `models.forecast train`) is recorded as `gate_failed` and the pipeline continues — a failed
acceptance gate is a legitimate, expected outcome, not a bug.

Every run writes `data/pipelines/retrain_<UTC timestamp>.json` with each stage's status,
duration and the last 20 lines of its output. See the README's "## CI/CD" section for the
Windows Task Scheduler and cron entries that call this on a schedule.
"""

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_CSV = Path("data/raw/Transactions.csv")
REPORT_DIR = Path("data/pipelines")

GATE_EXIT_CODE = 2
OUTPUT_TAIL_LINES = 20

STAGE_NAMES: tuple[str, ...] = ("ingest", "price", "listings", "search", "forecast")


class RunResult:
    """Duck-compatible with `subprocess.CompletedProcess` — the pieces tests need to fake."""

    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


Runner = Callable[[list[str]], subprocess.CompletedProcess | RunResult]
SkipCheck = Callable[[], str | None]


def default_runner(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, check=False)


def _python_cmd(*args: str) -> list[str]:
    return [sys.executable, *args]


@dataclass(frozen=True)
class Stage:
    name: str
    commands: list[list[str]]
    skip_check: SkipCheck | None = None


@dataclass
class StageReport:
    name: str
    status: str  # "ok" | "failed" | "gate_failed" | "skipped" | "dry_run"
    seconds: float
    exit_code: int | None
    output_tail: list[str] = field(default_factory=list)


@dataclass
class RetrainResult:
    reports: list[StageReport]
    failed: bool


def _ingest_skip_check(csv_path: Path) -> SkipCheck:
    def check() -> str | None:
        if not csv_path.exists():
            return f"source CSV not found at {csv_path} — skipping ingest"
        return None

    return check


def build_stages(csv_path: Path = DEFAULT_CSV) -> list[Stage]:
    """The five retraining stages, in order. `csv_path` is the ingestion source file — the
    ingestion CLI takes it as `--csv PATH` (default `data/raw/Transactions.csv`)."""
    return [
        Stage(
            "ingest",
            [_python_cmd("-m", "ingestion", "--csv", str(csv_path))],
            skip_check=_ingest_skip_check(csv_path),
        ),
        Stage("price", [_python_cmd("-m", "models.price", "train")]),
        Stage("listings", [_python_cmd("-m", "listings", "detect")]),
        Stage(
            "search",
            [
                _python_cmd("-m", "search", "queries"),
                _python_cmd("-m", "search", "train"),
            ],
        ),
        Stage(
            "forecast",
            [
                _python_cmd("-m", "models.forecast", "build"),
                _python_cmd("-m", "models.forecast", "train"),
            ],
        ),
    ]


def select_stages(
    stages: Sequence[Stage], only: Iterable[str] | None, skip: Iterable[str] | None
) -> list[Stage]:
    result = list(stages)
    if only is not None:
        only_set = set(only)
        result = [s for s in result if s.name in only_set]
    if skip is not None:
        skip_set = set(skip)
        result = [s for s in result if s.name not in skip_set]
    return result


def _tail(text: str, n: int = OUTPUT_TAIL_LINES) -> list[str]:
    lines = text.splitlines()
    return lines[-n:] if lines else []


def run_stage(stage: Stage, runner: Runner) -> StageReport:
    if stage.skip_check is not None:
        reason = stage.skip_check()
        if reason is not None:
            return StageReport(stage.name, "skipped", 0.0, None, [reason])

    start = time.monotonic()
    exit_code = 0
    tail: list[str] = []
    for argv in stage.commands:
        result = runner(argv)
        exit_code = result.returncode
        combined = (result.stdout or "") + (result.stderr or "")
        tail = _tail(combined)
        if exit_code != 0:
            break
    seconds = time.monotonic() - start

    if exit_code == GATE_EXIT_CODE:
        status = "gate_failed"
    elif exit_code != 0:
        status = "failed"
    else:
        status = "ok"
    return StageReport(stage.name, status, seconds, exit_code, tail)


def run_pipeline(stages: Sequence[Stage], runner: Runner) -> RetrainResult:
    reports: list[StageReport] = []
    failed = False
    for stage in stages:
        report = run_stage(stage, runner)
        reports.append(report)
        if report.status == "failed":
            failed = True
            break
        # "gate_failed" and "skipped" are legitimate outcomes: the pipeline continues.
    return RetrainResult(reports=reports, failed=failed)


def write_report(
    path: Path, reports: Iterable[StageReport], *, started_at: str, finished_at: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "started_at": started_at,
        "finished_at": finished_at,
        "stages": [
            {
                "name": r.name,
                "status": r.status,
                "seconds": r.seconds,
                "exit_code": r.exit_code,
                "output_tail": r.output_tail,
            }
            for r in reports
        ],
    }
    path.write_text(json.dumps(payload, indent=2))


def _print_plan(stages: Sequence[Stage]) -> None:
    print(f"Dry run — plan ({len(stages)} stage(s), no commands executed):")
    for stage in stages:
        if stage.skip_check is not None and stage.skip_check() is not None:
            print(f"  [{stage.name}] SKIP — {stage.skip_check()}")
            continue
        for argv in stage.commands:
            print(f"  [{stage.name}] {' '.join(argv)}")


def _print_summary(reports: Sequence[StageReport], report_path: Path) -> None:
    print(f"{'STAGE':<10} STATUS       SECONDS  EXIT")
    for r in reports:
        exit_display = "-" if r.exit_code is None else str(r.exit_code)
        print(f"{r.name:<10} {r.status:<12} {r.seconds:7.2f}  {exit_display}")
    print(f"Report written to {report_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pipelines", description="Scheduled retraining pipeline."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    retrain_cmd = commands.add_parser(
        "retrain",
        help="run ingest -> price -> listings -> search -> forecast, stage by stage",
    )
    retrain_cmd.add_argument(
        "--dry-run", action="store_true", help="print the plan and exit without running anything"
    )
    retrain_cmd.add_argument(
        "--only", nargs="+", metavar="STAGE", choices=STAGE_NAMES, help="run only these stages"
    )
    retrain_cmd.add_argument(
        "--skip", nargs="+", metavar="STAGE", choices=STAGE_NAMES, help="skip these stages"
    )
    retrain_cmd.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="ingestion source CSV, passed to `python -m ingestion --csv`",
    )
    return parser


def main(argv: list[str] | None = None, runner: Runner | None = None) -> int:
    args = _build_parser().parse_args(argv)
    active_runner = runner if runner is not None else default_runner

    stages = select_stages(build_stages(args.csv), only=args.only, skip=args.skip)

    if args.dry_run:
        _print_plan(stages)
        return 0

    started_at = datetime.now(UTC)
    result = run_pipeline(stages, active_runner)
    finished_at = datetime.now(UTC)

    report_path = REPORT_DIR / f"retrain_{started_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    write_report(
        report_path,
        result.reports,
        started_at=started_at.isoformat(),
        finished_at=finished_at.isoformat(),
    )
    _print_summary(result.reports, report_path)
    return 1 if result.failed else 0


if __name__ == "__main__":
    sys.exit(main())
