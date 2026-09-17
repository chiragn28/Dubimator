"""`python -m pipelines retrain`: the scheduled retraining pipeline.

Runs ingest -> price -> listings -> search -> forecast, each stage as one or more `python -m
...` subprocesses (via `sys.executable`, so it works inside `uv run` too). Stops at the first
real failure.

Exit codes: 0 is `ok`. Exit 2 is `gate_failed` ONLY for the two training steps that use it as
their gate-failure convention (`models.price train` and `models.forecast train`), and only
when the output is not an argparse usage error (argparse also exits 2). A gate failure is a
legitimate, expected outcome, so the pipeline continues. Any other non-zero exit, including 2
from any other step (`ingestion`, `models.forecast build`, `listings detect`, `search ...`) or
an argparse usage error, is `failed` and stops the pipeline.

Ingest is skipped when the source CSV is missing, or when its SHA-256 (the same
`ingestion.load.file_sha256` an ingestion run records) equals `source_sha256` of the latest
succeeded `dld.ingestion_runs` row: re-loading an unchanged file only costs time. If that
lookup fails (e.g. Postgres is down), ingest runs anyway and reports the real error itself.

Every run writes `data/pipelines/retrain_<UTC timestamp>.json` with each stage's status,
duration and the last 20 lines of its output. See the README's "## CI/CD" section for the
Windows Task Scheduler and cron entries that call this on a schedule.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ingestion.load import file_sha256

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
ShaReader = Callable[[], str | None]

# argparse's usage error: "usage: ..." then "<prog>: error: ..." on stderr, exit 2.
_USAGE_ERROR = re.compile(r"^usage: .*^\S.*: error: ", re.MULTILINE | re.DOTALL)


def default_runner(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, check=False)


def _python_cmd(*args: str) -> list[str]:
    return [sys.executable, *args]


@dataclass(frozen=True)
class Stage:
    name: str
    commands: list[list[str]]
    skip_check: SkipCheck | None = None
    # Indexes into `commands` whose exit code 2 means "acceptance gate failed".
    gate_steps: frozenset[int] = frozenset()


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


LATEST_SHA_SQL = (
    "SELECT source_sha256 FROM dld.ingestion_runs "
    "WHERE status = 'succeeded' ORDER BY run_id DESC LIMIT 1"
)


def latest_ingested_sha256() -> str | None:
    """`source_sha256` of the latest succeeded ingestion run, or None if there is none."""
    from dotenv import load_dotenv

    from ingestion.config import DbSettings

    load_dotenv()
    with closing(DbSettings.from_env().connect()) as conn, conn.cursor() as cur:
        cur.execute(LATEST_SHA_SQL)
        row = cur.fetchone()
    return row[0] if row else None


def ingest_skip_reason(csv_path: Path, read_latest_sha: ShaReader) -> str | None:
    """Why ingest should be skipped, or None to run it. `read_latest_sha` is injectable."""
    if not csv_path.exists():
        return f"source CSV not found at {csv_path} — skipping ingest"
    try:
        latest = read_latest_sha()
    except Exception:  # noqa: BLE001 — ingest runs anyway and reports the real problem
        return None
    if latest is None:
        return None
    digest = file_sha256(csv_path)
    if digest != latest:
        return None
    return (
        "source CSV unchanged since the last successful ingestion "
        f"(sha256 {digest[:12]}) — skipping ingest"
    )


def _ingest_skip_check(csv_path: Path, read_latest_sha: ShaReader | None) -> SkipCheck:
    def check() -> str | None:
        # Resolved at call time so tests can replace the module-level reader.
        reader = read_latest_sha if read_latest_sha is not None else latest_ingested_sha256
        return ingest_skip_reason(csv_path, reader)

    return check


def build_stages(
    csv_path: Path = DEFAULT_CSV, read_latest_sha: ShaReader | None = None
) -> list[Stage]:
    """The five retraining stages, in order. `csv_path` is the ingestion source file — the
    ingestion CLI takes it as `--csv PATH` (default `data/raw/Transactions.csv`).
    `read_latest_sha` defaults to `latest_ingested_sha256` (a database read)."""
    return [
        Stage(
            "ingest",
            [_python_cmd("-m", "ingestion", "--csv", str(csv_path))],
            skip_check=_ingest_skip_check(csv_path, read_latest_sha),
        ),
        Stage("price", [_python_cmd("-m", "models.price", "train")], gate_steps=frozenset({0})),
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
            gate_steps=frozenset({1}),
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


def _is_gate_failure(stage: Stage, step: int, exit_code: int, output: str) -> bool:
    return (
        exit_code == GATE_EXIT_CODE
        and step in stage.gate_steps
        and _USAGE_ERROR.search(output) is None
    )


def run_stage(stage: Stage, runner: Runner) -> StageReport:
    if stage.skip_check is not None:
        reason = stage.skip_check()
        if reason is not None:
            return StageReport(stage.name, "skipped", 0.0, None, [reason])

    start = time.monotonic()
    exit_code = 0
    gate_failed = False
    tail: list[str] = []
    for step, argv in enumerate(stage.commands):
        result = runner(argv)
        exit_code = result.returncode
        combined = (result.stdout or "") + (result.stderr or "")
        tail = _tail(combined)
        if exit_code != 0:
            gate_failed = _is_gate_failure(stage, step, exit_code, combined)
            break
    seconds = time.monotonic() - start

    if gate_failed:
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
        reason = stage.skip_check() if stage.skip_check is not None else None
        if reason is not None:
            print(f"  [{stage.name}] SKIP — {reason}")
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
