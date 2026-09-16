from datetime import date

import pytest
from forecast_fixtures import project_record, project_table, write_projects

from models.forecast.config import INFRA_TYPES
from models.forecast.infra import (
    MIN_PROJECTS,
    InfraTableError,
    load_projects,
    project_lines,
    validate_projects,
)

KNOWN = {1, 2}


def problems_for(tmp_path, **overrides):
    records = [project_record(index) for index in range(MIN_PROJECTS)]
    records[0] = project_record(0, **overrides)
    return validate_projects(load_projects(write_projects(tmp_path, records)), KNOWN)


def test_a_valid_table_has_no_problems(tmp_path):
    projects = project_table(tmp_path)
    assert validate_projects(projects, KNOWN) == []
    row = projects.row(1, named=True)
    assert row["announced_date"] == date(2016, 1, 1)
    assert row["actual_completion_date"] == date(2019, 6, 1)
    assert projects.row(0, named=True)["actual_completion_date"] is None
    assert row["area_ids"] == [1, 2]
    assert row["announced_date_text"] == "2016-01-01"


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"source_url": ""}, "P00: source_url must be an https:// link"),
        ({"source_url": "http://example.org"}, "P00: source_url must be an https:// link"),
        (
            {"announced_date": "2020-01-01"},
            "P00: announced_date 2020-01-01 is after planned_completion_date 2019-01-01",
        ),
        (
            {"announced_date": "2019-01-01", "planned_completion_date": "2019-03-01",
             "actual_completion_date": "2018-12-31"},
            "P00: actual_completion_date 2018-12-31 is before announced_date 2019-01-01",
        ),
        ({"affected_area_ids": "1;999"}, "P00: area id 999 is not in dld.areas"),
        ({"affected_area_ids": "1;x"}, "P00: affected_area_ids must be integers separated by ';'"),
        ({"affected_area_ids": ""}, "P00: affected_area_ids is empty"),
        ({"type": "stadium"}, f"P00: type 'stadium' is not one of {', '.join(INFRA_TYPES)}"),
        ({"announced_date": "2016-13-01"}, "P00: announced_date '2016-13-01' is not a YYYY-MM-DD date"),
        ({"planned_completion_date": ""}, "P00: planned_completion_date is missing"),
        ({"source_accessed": ""}, "P00: source_accessed is missing"),
        ({"name": " "}, "P00: name is missing"),
        ({"notes": ""}, "P00: notes must explain the area mapping"),
        ({"project_id": "P01"}, "P01: project_id is used more than once"),
    ],
)  # fmt: skip
def test_each_problem_is_reported(tmp_path, overrides, expected):
    assert expected in problems_for(tmp_path, **overrides)


def test_a_short_table_is_rejected(tmp_path):
    problems = validate_projects(project_table(tmp_path, count=3), KNOWN)
    assert problems == [f"the table has 3 projects; at least {MIN_PROJECTS} are required"]


def test_wrong_columns_raise(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("project_id,name\nP1,x\n", encoding="utf-8")
    with pytest.raises(InfraTableError, match="columns must be"):
        load_projects(path)


def test_the_committed_table_is_well_formed():
    projects = load_projects()
    assert projects.height >= MIN_PROJECTS
    every_id = {value for ids in projects["area_ids"].to_list() for value in (ids or []) if value}
    assert validate_projects(projects, every_id) == []
    assert set(projects["type"].unique()) <= set(INFRA_TYPES)
    assert projects["notes"].null_count() == 0


def test_project_lines_name_the_areas(tmp_path):
    lines = project_lines(project_table(tmp_path, count=2), {1: "Dubai Marina", 2: "Al Barsha"})
    assert len(lines) == 2
    assert lines[0].startswith("P00")
    assert "Dubai Marina (1), Al Barsha (2)" in lines[0]
    assert "done 2019-06-01" in lines[1]
    assert project_lines(project_table(tmp_path, count=1, affected_area_ids="7"), {})[0].endswith(
        "areas: ? (7)"
    )
