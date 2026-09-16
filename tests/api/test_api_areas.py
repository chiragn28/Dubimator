"""api.areas.AreaStats: pure summary/history statistics, and the /v1/areas routes.

Spec: docs/superpowers/specs/2026-09-17-phase8-demo-design.md ("Area statistics").
"""

from datetime import date
from pathlib import Path

import polars as pl
import pytest
from api_fixtures import KEY, client, fake_loaders, settings

from api.areas import HISTORY_MIN_SALES, HISTORY_MONTHS, SUMMARY_MIN_SALES, AreaStats
from models.price.features import month_from_number, month_number

SCHEMA = {
    "area_id": pl.Int64,
    "instance_date": pl.Date,
    "sub_kind": pl.Utf8,
    "size_basis": pl.Utf8,
    "area_sqm": pl.Float64,
    "price_aed": pl.Float64,
    "is_clean": pl.Boolean,
}

AREAS = pl.DataFrame(
    {"area_id": [1, 2], "name_en": ["Area One", "Area Two"]},
    schema={"area_id": pl.Int64, "name_en": pl.Utf8},
)

DATA_END = date(2023, 3, 17)


def _rows(*records) -> pl.DataFrame:
    return pl.DataFrame(list(records), schema=SCHEMA)


def _bucket(
    area_id: int,
    month: date,
    ppsms: list[float],
    sub_kind: str = "flat",
    size_basis: str = "built_up",
    is_clean: bool = True,
) -> list[dict]:
    """One row per ppsm value, dated within `month` (area_sqm fixed at 100 m2)."""
    return [
        {
            "area_id": area_id,
            "instance_date": date(month.year, month.month, 1 + i % 28),
            "sub_kind": sub_kind,
            "size_basis": size_basis,
            "area_sqm": 100.0,
            "price_aed": ppsm * 100.0,
            "is_clean": is_clean,
        }
        for i, ppsm in enumerate(ppsms)
    ]


def _months_before(data_end: date, back: int) -> date:
    return month_from_number(month_number(data_end) - back)


# --- from_homes: summary ----------------------------------------------------------------


def test_summary_computes_the_12_month_median():
    ppsms = [8000.0, 9000.0, 10000.0, 11000.0, 12000.0] * 5  # 25 sales, median 10000
    rows = _rows(*_bucket(1, _months_before(DATA_END, 1), ppsms))
    stats = AreaStats.from_homes(rows, AREAS, DATA_END)
    records = stats.summary_records()
    assert len(records) == 1
    assert records[0]["area_id"] == 1
    assert records[0]["name"] == "Area One"
    assert records[0]["median_ppsm_12m"] == pytest.approx(10000.0)
    assert records[0]["sales_12m"] == 25


def test_summary_computes_the_change_ratio_against_the_prior_12_months():
    last12 = [10000.0] * 20
    prev12 = [8000.0] * 20
    rows = _rows(
        *_bucket(1, _months_before(DATA_END, 1), last12),
        *_bucket(1, _months_before(DATA_END, 13), prev12),
    )
    stats = AreaStats.from_homes(rows, AREAS, DATA_END)
    record = stats.summary_records()[0]
    assert record["change_12m"] == pytest.approx(10000.0 / 8000.0 - 1.0)


def test_summary_omits_areas_with_fewer_than_20_sales_in_the_last_12_months():
    rows = _rows(
        *_bucket(1, _months_before(DATA_END, 1), [10000.0] * SUMMARY_MIN_SALES),
        *_bucket(2, _months_before(DATA_END, 1), [10000.0] * (SUMMARY_MIN_SALES - 1)),
    )
    stats = AreaStats.from_homes(rows, AREAS, DATA_END)
    records = stats.summary_records()
    assert [r["area_id"] for r in records] == [1]


def test_summary_records_are_sorted_by_sales_12m_descending():
    rows = _rows(
        *_bucket(1, _months_before(DATA_END, 1), [10000.0] * 20),
        *_bucket(2, _months_before(DATA_END, 1), [10000.0] * 35),
    )
    stats = AreaStats.from_homes(rows, AREAS, DATA_END)
    records = stats.summary_records()
    assert [r["area_id"] for r in records] == [2, 1]
    assert [r["sales_12m"] for r in records] == [35, 20]


# --- from_homes: history -----------------------------------------------------------------


def test_history_nulls_the_median_for_months_with_fewer_than_5_sales():
    month = _months_before(DATA_END, 1)
    rows = _rows(
        *_bucket(1, month, [10000.0] * HISTORY_MIN_SALES),
        *_bucket(1, _months_before(DATA_END, 2), [10000.0] * (HISTORY_MIN_SALES - 1)),
    )
    stats = AreaStats.from_homes(rows, AREAS, DATA_END)
    points = {p["month"]: p for s in stats.history_records(1)["series"] for p in s["points"]}
    assert points[f"{month:%Y-%m}"]["median_ppsm"] == pytest.approx(10000.0)
    assert points[f"{month:%Y-%m}"]["sales"] == HISTORY_MIN_SALES
    thin_month = _months_before(DATA_END, 2)
    assert points[f"{thin_month:%Y-%m}"]["median_ppsm"] is None
    assert points[f"{thin_month:%Y-%m}"]["sales"] == HISTORY_MIN_SALES - 1


def test_history_keeps_villa_plot_as_its_own_market_kind():
    month = _months_before(DATA_END, 1)
    rows = _rows(
        *_bucket(1, month, [10000.0] * 6, sub_kind="flat", size_basis="built_up"),
        *_bucket(1, month, [15000.0] * 6, sub_kind="villa", size_basis="built_up"),
        *_bucket(1, month, [20000.0] * 6, sub_kind="villa", size_basis="plot"),
    )
    stats = AreaStats.from_homes(rows, AREAS, DATA_END)
    series_by_kind = {s["market_kind"]: s for s in stats.history_records(1)["series"]}
    assert set(series_by_kind) == {"flat", "villa", "villa_plot"}
    villa_point = next(
        p for p in series_by_kind["villa"]["points"] if p["month"] == f"{month:%Y-%m}"
    )
    plot_point = next(
        p for p in series_by_kind["villa_plot"]["points"] if p["month"] == f"{month:%Y-%m}"
    )
    assert villa_point["median_ppsm"] == pytest.approx(15000.0)
    assert plot_point["median_ppsm"] == pytest.approx(20000.0)


def test_history_caps_at_96_months():
    within = _months_before(DATA_END, HISTORY_MONTHS - 1)  # the oldest included month
    outside = _months_before(DATA_END, HISTORY_MONTHS)  # one month too old
    rows = _rows(
        *_bucket(1, within, [10000.0] * 6),
        *_bucket(1, outside, [10000.0] * 6),
    )
    stats = AreaStats.from_homes(rows, AREAS, DATA_END)
    months = {p["month"] for s in stats.history_records(1)["series"] for p in s["points"]}
    assert f"{within:%Y-%m}" in months
    assert f"{outside:%Y-%m}" not in months


def test_history_records_none_for_an_unknown_area():
    rows = _rows(*_bucket(1, _months_before(DATA_END, 1), [10000.0] * 6))
    stats = AreaStats.from_homes(rows, AREAS, DATA_END)
    assert stats.history_records(999) is None


def test_dirty_rows_are_excluded():
    rows = _rows(
        *_bucket(1, _months_before(DATA_END, 1), [10000.0] * SUMMARY_MIN_SALES, is_clean=False),
    )
    stats = AreaStats.from_homes(rows, AREAS, DATA_END)
    assert stats.summary_records() == []


# --- routes (FakeAreas) -------------------------------------------------------------------


def _headers():
    return {"X-API-Key": KEY}


def test_areas_endpoint_returns_200():
    with client(settings(), fake_loaders()) as c:
        response = c.get("/v1/areas", headers=_headers())
    assert response.status_code == 200
    body = response.json()
    assert "data_end" in body
    assert isinstance(body["areas"], list)
    assert body["areas"][0]["area_id"]


def test_area_history_unknown_area_is_404():
    with client(settings(), fake_loaders()) as c:
        response = c.get("/v1/areas/404/history", headers=_headers())
    assert response.status_code == 404
    assert response.json()["error"]["message"] == "area 404 not found"


def test_area_history_known_area_returns_200():
    with client(settings(), fake_loaders()) as c:
        response = c.get("/v1/areas/1/history", headers=_headers())
    assert response.status_code == 200
    body = response.json()
    assert body["area_id"] == 1
    assert body["series"]


def test_areas_missing_component_returns_503():
    def broken(settings, context):
        raise RuntimeError("no data")

    with client(settings(), fake_loaders(areas=broken)) as c:
        response = c.get("/v1/areas", headers=_headers())
    assert response.status_code == 503


# --- database ------------------------------------------------------------------------------


def test_area_stats_load_reads_postgres(pg_test_db):
    from ingestion.pipeline import run_pipeline

    run_pipeline(Path("tests/fixtures/price_sample.csv"), pg_test_db)
    stats = AreaStats.load(pg_test_db)
    assert stats.summary.height > 0 or stats.history.height > 0
