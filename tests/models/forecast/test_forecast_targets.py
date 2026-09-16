import math
from datetime import date, timedelta

import polars as pl
import pytest

from models.forecast.config import BASE_DAYS
from models.forecast.rows import market_kind
from models.forecast.targets import (
    STATUSES,
    add_base,
    add_keys,
    build_targets,
    rolling_stats,
)

T = date(2020, 6, 1)
SCHEMA = {
    "row_id": pl.Int64,
    "transaction_id": pl.Utf8,
    "instance_date": pl.Date,
    "area_id": pl.Int64,
    "sub_kind": pl.Utf8,
    "market_kind": pl.Utf8,
    "building_name": pl.Utf8,
    "ppsm": pl.Float64,
}


def sale(days, ppsm, building="Tower A", area_id=1, sub_kind="flat", name=None, market=None):
    return {
        "transaction_id": name,
        "instance_date": T + timedelta(days=days),
        "area_id": area_id,
        "sub_kind": sub_kind,
        "market_kind": market or sub_kind,
        "building_name": building,
        "ppsm": ppsm,
    }


def frame_of(sales):
    rows = [
        {**record, "row_id": index, "transaction_id": record["transaction_id"] or f"t{index}"}
        for index, record in enumerate(sales)
    ]
    return pl.DataFrame(rows, schema=SCHEMA)


def anchor(frame, name="anchor"):
    return frame.filter(pl.col("transaction_id") == name).row(0, named=True)


# Area 1: the anchor at T, five building sales inside [T-92, T), one just outside it,
# five inside the 3m window [T+76, T+107] and one just outside each end.
AREA_ONE = [
    sale(0, 100.0, name="anchor"),
    sale(-93, 999.0),
    sale(-92, 10.0),
    sale(-50, 20.0),
    sale(-30, 30.0),
    sale(-10, 40.0),
    sale(-1, 50.0),
    sale(75, 999.0),
    sale(76, 60.0),
    sale(80, 60.0),
    sale(90, 60.0),
    sale(100, 66.0),
    sale(107, 66.0),
    sale(108, 999.0),
]


def test_add_keys_builds_the_three_levels():
    frame = add_keys(
        frame_of([sale(0, 1.0, building="Al Noor Tower"), sale(0, 1.0, building=None)])
    )
    assert frame["building_key"].to_list() == ["1|noor tower", None]
    assert frame["area_key"].to_list() == ["1|flat", "1|flat"]
    assert frame["city_key"].to_list() == ["flat", "flat"]


def test_plot_and_built_up_villas_have_separate_area_keys():
    frame = add_keys(
        frame_of(
            [
                sale(0, 1.0, building=None, sub_kind="villa"),
                sale(0, 1.0, building=None, sub_kind="villa", market="villa_plot"),
            ]
        )
    )
    assert frame["area_key"].to_list() == ["1|villa", "1|villa_plot"]
    assert frame["city_key"].to_list() == ["villa", "villa_plot"]


def test_rolling_stats_window_edges():
    frame = add_keys(frame_of(AREA_ONE))
    base = rolling_stats(
        frame, "building_key", start_days=-92, length_days=92, closed="left", prefix="b"
    )
    row = anchor(base)
    assert row["b_n"] == 5  # -92 is in, -93 is out, the anchor's own day is out
    assert row["b_ppsm"] == 30.0
    ahead = rolling_stats(
        frame, "building_key", start_days=76, length_days=31, closed="both", prefix="f"
    )
    row = anchor(ahead)
    assert row["f_n"] == 5  # 76 and 107 are in, 75 and 108 are out
    assert row["f_ppsm"] == 60.0
    assert base["row_id"].to_list() == sorted(base["row_id"].to_list())


def test_rolling_stats_ignores_null_ppsm_and_null_keys():
    sales = [*AREA_ONE, sale(-5, None, name="query"), sale(-5, 5.0, building=None, name="villa")]
    frame = add_keys(frame_of(sales))
    out = rolling_stats(
        frame, "building_key", start_days=-92, length_days=92, closed="left", prefix="b"
    )
    assert anchor(out)["b_n"] == 5  # the null-ppsm row at -5 does not count
    assert anchor(out, "villa")["b_n"] == 0
    assert anchor(out, "villa")["b_ppsm"] is None


def test_base_prefers_the_building_then_the_area():
    sales = [
        sale(0, 100.0, building="B", area_id=2, name="anchor"),
        *[sale(-d, 10.0 * d, building="B", area_id=2) for d in (5, 10, 15, 20)],
        sale(-25, 250.0, building="C", area_id=2),
    ]
    row = anchor(add_base(add_keys(frame_of(sales))))
    assert row["building_base_n"] == 4
    assert row["base_level"] == "area"
    assert row["base_n"] == 5
    assert row["base_ppsm"] == 150.0  # median of 50, 100, 150, 200, 250
    one = anchor(add_base(add_keys(frame_of(AREA_ONE))))
    assert (one["base_level"], one["base_n"], one["base_ppsm"]) == ("building", 5, 30.0)


def test_no_base_when_the_area_has_fewer_than_five_sales():
    sales = [
        sale(0, 100.0, area_id=4, name="anchor"),
        *[sale(-d, 50.0, area_id=4) for d in (1, 2, 3, 4)],
    ]
    frame, report = build_targets(frame_of(sales), T + timedelta(days=2000))
    assert anchor(frame)["base_level"] is None
    assert anchor(frame)["status_3m"] == "no_base"
    assert report.horizons["3m"]["no_base"] == 5  # no row has five earlier sales
    assert report.rows == 5


def test_growth_is_the_log_ratio_at_the_base_level():
    frame, _ = build_targets(frame_of(AREA_ONE), T + timedelta(days=2000))
    row = anchor(frame)
    assert row["status_3m"] == "usable"
    assert row["target_n_3m"] == 5
    assert row["growth_3m"] == pytest.approx(math.log(60.0 / 30.0))
    assert row["status_1y"] == "no_target"  # nothing in [T+335, T+396]
    assert row["growth_1y"] is None


def test_target_stays_at_the_building_level():
    sales = [
        sale(0, 100.0, building="D", area_id=3, name="anchor"),
        *[sale(-d, 20.0, building="D", area_id=3) for d in (5, 10, 15, 20, 25)],
        *[sale(80 + d, 40.0, building="D", area_id=3) for d in (0, 1, 2)],
        *[sale(80 + d, 40.0, building="E", area_id=3) for d in (0, 1, 2)],
    ]
    frame, _ = build_targets(frame_of(sales), T + timedelta(days=2000))
    row = anchor(frame)
    assert row["base_level"] == "building"
    assert row["target_n_3m"] == 3  # the area has 6 sales in the window; the building has 3
    assert row["status_3m"] == "no_target"
    assert row["growth_3m"] is None


def test_open_windows_are_excluded_and_counted():
    frame, report = build_targets(frame_of(AREA_ONE), T + timedelta(days=100))
    row = anchor(frame)
    assert row["status_3m"] == "window_open"  # T + 107 is after the data end
    assert row["growth_3m"] is None
    assert report.horizons["3m"]["usable"] == 0
    for name in ("3m", "1y", "3y"):
        counts = frame.group_by(f"status_{name}").len()
        expected = dict(counts.iter_rows())
        assert {s: report.horizons[name][s] for s in STATUSES if s in expected} == expected
        assert sum(report.horizons[name][s] for s in STATUSES) == report.rows


def test_report_lines_and_dict():
    _frame, report = build_targets(frame_of(AREA_ONE), T + timedelta(days=2000))
    assert report.horizons["3m"]["last_t"] == T
    assert report.lines()[0] == f"Target rows: {len(AREA_ONE):,}"
    assert any(line.strip().startswith("3m:") and str(T) in line for line in report.lines())
    assert report.to_dict()["horizons"]["3m"]["last_t"] == T.isoformat()
    assert report.to_dict()["horizons"]["1y"]["last_t"] is None


def test_history_targets_track_the_known_growth(history):
    data_end = history["instance_date"].max()
    rows = history.with_row_index("row_id").with_columns(
        pl.col("row_id").cast(pl.Int64),
        (pl.col("price_aed") / pl.col("area_sqm")).alias("ppsm"),
        market_kind().alias("market_kind"),
    )
    frame, report = build_targets(rows, data_end)
    usable = frame.filter(
        (pl.col("status_1y") == "usable")
        & (pl.col("area_id") == 1)
        & (pl.col("sub_kind") == "flat")
    )
    assert usable.height > 1000
    # The first BASE_DAYS of history cannot show a full building-level base window (e.g. a
    # building with only 1 prior sale falls back to the area, which already has 5+ sales
    # combined across the area's buildings), so a handful of the very earliest sales legitimately
    # settle at the area level. Check building attribution once a full base window exists.
    settled = usable.filter(
        pl.col("instance_date") >= history["instance_date"].min() + timedelta(days=BASE_DAYS)
    )
    assert (settled["base_level"] == "building").all()
    # area 1 grows 0.005 a month; base centre T-46 d to target centre T+365.5 d is ~13.5 months
    assert usable["growth_1y"].median() == pytest.approx(0.005 * 411.5 / 30.4375, abs=0.01)
    assert report.horizons["3y"]["last_t"] <= data_end - timedelta(days=1126)
