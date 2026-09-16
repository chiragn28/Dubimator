from datetime import date, timedelta

import polars as pl
import pytest
from forecast_fixtures import project_record, project_table, write_projects

from models.forecast.config import INFRA_TYPES
from models.forecast.infra import (
    MIN_PROJECTS,
    InfraTableError,
    add_infra_features,
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


ANNOUNCED = date(2018, 1, 1)
PLANNED = date(2020, 1, 1)
ACTUAL = date(2020, 6, 1)


def rows_on(days_and_areas):
    return pl.DataFrame(
        {
            "row_id": list(range(len(days_and_areas))),
            "instance_date": [day for day, _ in days_and_areas],
            "area_id": [area for _, area in days_and_areas],
        },
        schema={"row_id": pl.Int64, "instance_date": pl.Date, "area_id": pl.Int64},
    )


def one_metro(tmp_path, **overrides):
    record = project_record(
        0,
        type="metro_rail",
        announced_date=str(ANNOUNCED),
        planned_completion_date=str(PLANNED),
        actual_completion_date=str(ACTUAL),
        affected_area_ids="1",
        **overrides,
    )
    return load_projects(write_projects(tmp_path, [record]))


def test_a_project_counts_only_from_its_announcement(tmp_path):
    frame = rows_on([(ANNOUNCED - timedelta(days=1), 1), (ANNOUNCED, 1), (date(2020, 3, 1), 1)])
    out = add_infra_features(frame, one_metro(tmp_path))
    assert out["infra_active_metro_rail"].to_list() == [0.0, 1.0, 1.0]
    assert out["infra_mix_metro_rail"].to_list() == [0.0, 1.0, 1.0]
    months = out["infra_months_to_next"].to_list()
    assert months[0] is None
    assert months[1] == pytest.approx((PLANNED - ANNOUNCED).days / 30.4375)
    assert months[2] == 0.0  # planned date passed but not yet open: delayed, not negative


def test_a_completion_counts_only_from_its_opening(tmp_path):
    days = [
        ACTUAL - timedelta(days=1),
        ACTUAL,
        ACTUAL + timedelta(days=729),
        ACTUAL + timedelta(days=730),
    ]
    out = add_infra_features(rows_on([(day, 1) for day in days]), one_metro(tmp_path))
    assert out["infra_active_metro_rail"].to_list() == [1.0, 0.0, 0.0, 0.0]
    assert out["infra_completed_24m"].to_list() == [0.0, 1.0, 1.0, 0.0]
    assert out["infra_months_to_next"].to_list()[1:] == [None, None, None]


def test_unaffected_areas_and_order(tmp_path):
    frame = rows_on([(date(2019, 1, 1), 2), (date(2019, 1, 1), 1), (date(2014, 1, 1), 1)])
    out = add_infra_features(frame, one_metro(tmp_path))
    assert out["row_id"].to_list() == [0, 1, 2]
    assert out["infra_active_metro_rail"].to_list() == [0.0, 1.0, 0.0]
    for kind in INFRA_TYPES:
        assert out[f"infra_mix_{kind}"].null_count() == 0
    assert out["infra_completed_24m"].to_list() == [0.0, 0.0, 0.0]


def test_the_type_mix_splits_active_projects(tmp_path):
    records = [
        project_record(0, type="metro_rail", affected_area_ids="1", actual_completion_date=""),
        project_record(1, type="mall", affected_area_ids="1", actual_completion_date=""),
        project_record(2, type="mall", affected_area_ids="1", actual_completion_date=""),
    ]
    projects = load_projects(write_projects(tmp_path, records))
    out = add_infra_features(rows_on([(date(2017, 1, 1), 1)]), projects)
    assert out["infra_active_mall"][0] == 2.0
    assert out["infra_mix_mall"][0] == pytest.approx(2 / 3)
    assert out["infra_mix_metro_rail"][0] == pytest.approx(1 / 3)
    assert out["infra_mix_park"][0] == 0.0
