"""The hand-curated, sourced infrastructure table: loading, validation and listing.

Spec: "Infrastructure table" in
docs/superpowers/specs/2026-09-16-phase6-price-forecasting-design.md. A project counts only
from its announced_date; a completion only from its actual_completion_date.
planned_completion_date is the date stated when the project was announced.
"""

from datetime import date
from pathlib import Path

import polars as pl

from ingestion.config import DbSettings
from models.forecast.config import INFRA_TYPES

PROJECTS_PATH = Path(__file__).resolve().parent / "reference" / "infrastructure_projects.csv"
INFRA_COLUMNS = (
    "project_id", "name", "type", "announced_date", "planned_completion_date",
    "actual_completion_date", "affected_area_ids", "source_url", "source_accessed", "notes",
)  # fmt: skip
DATE_COLUMNS = (
    "announced_date", "planned_completion_date", "actual_completion_date", "source_accessed",
)  # fmt: skip
REQUIRED_DATES = ("announced_date", "planned_completion_date", "source_accessed")
MIN_PROJECTS = 15


class InfraTableError(ValueError):
    """The infrastructure CSV cannot be read as a table of projects."""


def load_projects(path: Path = PROJECTS_PATH) -> pl.DataFrame:
    raw = pl.read_csv(path, infer_schema=False)
    if tuple(raw.columns) != INFRA_COLUMNS:
        raise InfraTableError(
            f"{Path(path).name} columns must be {', '.join(INFRA_COLUMNS)}; "
            f"found {', '.join(raw.columns)}"
        )
    stripped = raw.with_columns(
        pl.when(pl.col(name).str.strip_chars() != "")
        .then(pl.col(name).str.strip_chars())
        .alias(name)
        for name in INFRA_COLUMNS
    )
    return stripped.with_columns(
        *(pl.col(name).alias(f"{name}_text") for name in DATE_COLUMNS),
        *(pl.col(name).str.to_date("%Y-%m-%d", strict=False).alias(name) for name in DATE_COLUMNS),
        pl.col("affected_area_ids")
        .str.split(";")
        .list.eval(pl.element().str.strip_chars().cast(pl.Int64, strict=False))
        .alias("area_ids"),
    )


def _date_problem(row: dict, name: str) -> str | None:
    text, value = row[f"{name}_text"], row[name]
    if text is None:
        return f"{name} is missing" if name in REQUIRED_DATES else None
    if value is None:
        return f"{name} {text!r} is not a YYYY-MM-DD date"
    return None


def _row_problems(row: dict, known_area_ids) -> list[str]:
    found = []
    if row["project_id"] is None:
        found.append("project_id is missing")
    if row["name"] is None:
        found.append("name is missing")
    if row["type"] not in INFRA_TYPES:
        found.append(f"type {row['type']!r} is not one of {', '.join(INFRA_TYPES)}")
    found += [problem for name in DATE_COLUMNS if (problem := _date_problem(row, name))]
    announced = row["announced_date"]
    planned = row["planned_completion_date"]
    actual = row["actual_completion_date"]
    if announced and planned and announced > planned:
        found.append(f"announced_date {announced} is after planned_completion_date {planned}")
    if announced and actual and actual < announced:
        found.append(f"actual_completion_date {actual} is before announced_date {announced}")
    if not (row["source_url"] or "").startswith("https://"):
        found.append("source_url must be an https:// link")
    if row["notes"] is None:
        found.append("notes must explain the area mapping")
    ids = row["area_ids"]
    if not ids:
        found.append("affected_area_ids is empty")
    elif any(value is None for value in ids):
        found.append("affected_area_ids must be integers separated by ';'")
    else:
        found += [
            f"area id {value} is not in dld.areas" for value in ids if value not in known_area_ids
        ]
    return found


def validate_projects(projects: pl.DataFrame, known_area_ids) -> list[str]:
    problems = []
    if projects.height < MIN_PROJECTS:
        problems.append(
            f"the table has {projects.height} projects; at least {MIN_PROJECTS} are required"
        )
    duplicated = projects.filter(
        pl.col("project_id").is_not_null() & pl.col("project_id").is_duplicated()
    )
    problems += [
        f"{project_id}: project_id is used more than once"
        for project_id in sorted(set(duplicated["project_id"].to_list()))
    ]
    for row in projects.iter_rows(named=True):
        label = row["project_id"] or "(missing id)"
        problems += [f"{label}: {problem}" for problem in _row_problems(row, known_area_ids)]
    return problems


def fetch_reference(settings: DbSettings) -> tuple[dict[int, str], date]:
    """dld.areas names by id and the last clean sale date (read-only)."""
    conn = settings.connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT area_id, name_en FROM dld.areas ORDER BY area_id")
            names = dict(cur.fetchall())
            cur.execute(
                "SELECT max(instance_date) FROM dld.transactions WHERE exclusion_reason IS NULL"
            )
            (data_end,) = cur.fetchone()
    finally:
        conn.close()
    return names, data_end


def project_lines(projects: pl.DataFrame, area_names: dict[int, str]) -> list[str]:
    lines = []
    for row in projects.iter_rows(named=True):
        done = f"done {row['actual_completion_date']}" if row["actual_completion_date"] else "open"
        areas = ", ".join(
            f"{area_names.get(value, '?')} ({value})" for value in (row["area_ids"] or [])
        )
        lines.append(
            f"{row['project_id']}  {row['type']:<10} {row['announced_date']} -> "
            f"{row['planned_completion_date']} ({done})  {row['name']}  areas: {areas}"
        )
    return lines
