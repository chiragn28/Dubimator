import dataclasses
import math
import re

import numpy as np
import polars as pl
import pytest

from models.forecast.config import HORIZON_SPECS, ForecastConfig
from models.forecast.evaluate import (
    FLAG_MAPE,
    MODEL,
    PRIMARY_SEGMENTS,
    SEGMENTS,
    ape,
    bootstrap_mape_upper,
    format_table,
    gate,
    segment_masks,
    segment_table,
    table_mape,
    top_areas,
)
from models.forecast.intervals import (
    LOW_MESSAGE,
    POOLED,
    confidence,
    coverage_by_segment,
    fit_intervals,
    half_width,
    interval_segments,
    low_message,
)

CONFIG = ForecastConfig()
THREE_M = HORIZON_SPECS["3m"]


def test_interval_segments():
    frame = pl.DataFrame({"reg_type": ["ready", "off_plan"], "sub_kind": ["villa", "flat"]})
    assert interval_segments(frame).to_list() == ["ready_villa", "off_plan_unit"]


def test_conformal_coverage_is_close_to_eighty_percent():
    rng = np.random.default_rng(0)
    segments = ["ready_unit", "off_plan_unit"] * 2_500
    intervals = fit_intervals(segments, np.abs(rng.normal(0, 0.1, 5_000)), CONFIG)
    actual = rng.normal(0, 0.1, 4_000)
    coverage = coverage_by_segment(["ready_unit"] * 4_000, np.zeros(4_000), actual, intervals)
    assert 0.75 <= coverage["all"] <= 0.85
    assert coverage["ready_unit"] == coverage["all"]


def test_small_segments_use_the_pooled_width():
    errors = np.linspace(0.0, 1.0, 1_000)
    segments = ["ready_unit"] * 990 + ["ready_villa"] * 10
    intervals = fit_intervals(segments, errors, CONFIG)
    assert intervals["ready_villa"] == intervals[POOLED]
    assert intervals["ready_unit"] != intervals[POOLED]
    assert half_width(intervals, "off_plan_villa") == intervals[POOLED]
    with pytest.raises(ValueError, match="too few calibration rows"):
        fit_intervals(["ready_unit"] * 2, [0.1, 0.2], CONFIG)


@pytest.mark.parametrize(
    ("level", "base_n", "days", "area_rows", "expected"),
    [
        ("building", 10, 90, 50, "HIGH"),
        ("building", 9, 90, 50, "MEDIUM"),
        ("building", 10, 91, 50, "MEDIUM"),
        ("area", 30, 10, 500, "MEDIUM"),
        ("area", 5, 180, 500, "MEDIUM"),
        ("area", 5, 181, 500, "LOW"),
        ("area", 4, 10, 500, "LOW"),
        ("building", 30, 5, 49, "LOW"),
        ("building", 30, None, 500, "LOW"),
        (None, None, None, 500, "LOW"),
    ],
)  # fmt: skip
def test_confidence_rules(level, base_n, days, area_rows, expected):
    assert confidence(level, base_n, days, area_rows, CONFIG.low_confidence_area_rows) == expected


def test_low_message_states_the_half_width():
    assert low_message(1_000_000, 820_000, 1_180_000) == (
        "Limited comparable data for this property type/area. Estimate has wide uncertainty (±18%)."
    )
    assert LOW_MESSAGE.format(pct=5).endswith("(±5%).")


def test_ape_is_the_price_error():
    assert ape([0.1, 0.0], [0.0, 0.1]) == pytest.approx([math.exp(0.1) - 1, 1 - math.exp(-0.1)])


def eval_frame():
    return pl.DataFrame(
        {
            "reg_type": ["ready", "ready", "off_plan", "ready"],
            "area_id": [1, 2, 3, 3],
            "building_age_proxy_years": [0.5, 3.0, 7.0, None],
        },
        schema={"reg_type": pl.Utf8, "area_id": pl.Int64, "building_age_proxy_years": pl.Float64},
    )


def test_top_areas_break_ties_by_id():
    train = pl.DataFrame({"area_id": [3, 3, 1, 2, 2, 5]})
    assert top_areas(train, 2) == [2, 3]
    assert top_areas(train, 10) == [2, 3, 1, 5]


def test_segment_masks():
    masks = segment_masks(eval_frame(), [3])
    as_lists = {name: mask.tolist() for name, mask in masks.items()}
    assert list(masks) == list(SEGMENTS)
    assert as_lists["all"] == [True] * 4
    assert as_lists["ready"] == [True, True, False, True]
    assert as_lists["off_plan"] == [False, False, True, False]
    assert as_lists["top5"] == [False, False, True, True]
    assert as_lists["rest"] == [True, True, False, False]
    assert as_lists["age_lt1"] == [True, False, False, False]
    assert as_lists["age_1to5"] == [False, True, False, False]
    assert as_lists["age_gt5"] == [False, False, True, False]
    assert as_lists["age_unknown"] == [False, False, False, True]
    assert as_lists["age_gt2"] == [False, True, True, False]


def test_segment_table_scores_every_model_per_segment():
    actual = np.zeros(4)
    predictions = {MODEL: np.array([0.0, 0.0, 0.5, 0.0]), "no_change": np.zeros(4)}
    table = segment_table(eval_frame(), predictions, actual, [3])
    assert table.columns == ["segment", "model", "rows", "mape", "median_ape", "flagged"]
    assert table.height == len(SEGMENTS) * 2
    assert table_mape(table, "off_plan", MODEL) == pytest.approx(math.exp(0.5) - 1)
    assert table_mape(table, "all", MODEL) == pytest.approx((math.exp(0.5) - 1) / 4)
    flagged = table.filter(pl.col("flagged"))
    assert flagged["segment"].to_list() == ["off_plan", "top5", "age_gt5", "age_gt2"]
    # "all" pools all 4 rows, so its MAPE is diluted below the flag threshold.
    assert (math.exp(0.5) - 1) / 4 < FLAG_MAPE
    empty = segment_table(eval_frame().head(1), {MODEL: np.zeros(1)}, np.zeros(1), [3])
    assert math.isnan(table_mape(empty, "off_plan", MODEL))
    assert not empty.filter(pl.col("segment") == "off_plan")["flagged"][0]


def test_bootstrap_upper_bound():
    errors = np.random.default_rng(1).exponential(0.1, 2_000)
    upper = bootstrap_mape_upper(errors, 1_000, seed=7)
    assert upper == bootstrap_mape_upper(errors, 1_000, seed=7)
    assert errors.mean() < upper < errors.mean() + 0.01
    assert bootstrap_mape_upper(np.full(10, 0.2), 1_000, seed=7) == pytest.approx(0.2)
    assert math.isnan(bootstrap_mape_upper(np.array([]), 1_000, seed=7))


def table_of(model=0.05, no_change=0.10, area_trend=0.08, ready=None, rows=100):
    records = []
    for segment in SEGMENTS:
        for name, value in (("model", model), ("no_change", no_change), ("area_trend", area_trend)):
            if name == "model" and segment == "ready" and ready is not None:
                value = ready
            records.append(
                {"segment": segment, "model": name, "rows": rows, "mape": value,
                 "median_ape": value / 2, "flagged": value > FLAG_MAPE}
            )  # fmt: skip
    return pl.DataFrame(records)


def test_gate_passes_when_every_condition_holds():
    result = gate(table_of(), THREE_M, model_upper=0.06)
    assert result.passed
    assert result.reasons == ()
    assert result.checks["strongest_baseline"] == 0.08


def test_gate_fails_a_primary_segment_over_the_limit():
    result = gate(table_of(ready=0.16), THREE_M, model_upper=0.06)
    assert not result.passed
    assert result.reasons == ("3m resale MAPE 16.0% exceeds the 15% gate",)


def test_gate_fails_when_only_one_baseline_is_beaten():
    result = gate(table_of(model=0.09), THREE_M, model_upper=0.095)
    assert not result.passed
    assert "3m MAPE 9.0% does not beat the area_trend baseline (8.0%)" in result.reasons
    assert not any("no_change" in reason for reason in result.reasons)


def test_gate_fails_when_the_upper_bound_is_not_below_the_stronger_baseline():
    result = gate(table_of(), THREE_M, model_upper=0.08)
    assert result.reasons == (
        "3m MAPE 95% upper bound 8.0% is not below the stronger baseline (8.0%)",
    )


def test_gate_fails_without_primary_rows():
    table = table_of().with_columns(
        pl.when(pl.col("segment") == "age_gt2").then(0).otherwise(pl.col("rows")).alias("rows")
    )
    result = gate(table, THREE_M, model_upper=0.06)
    assert result.reasons == ("3m has no older-building (>2y) test rows",)
    assert set(PRIMARY_SEGMENTS) == {"ready", "top5", "age_gt2"}


def test_gate_uses_each_horizons_limit():
    three_y = dataclasses.replace(HORIZON_SPECS["3y"])
    result = gate(table_of(model=0.25, no_change=0.4, area_trend=0.35), three_y, model_upper=0.3)
    assert result.passed


def test_format_table_lines():
    lines = format_table("3m", table_of(model=0.3, no_change=0.4, area_trend=0.35))
    assert len(lines) == 1 + len(SEGMENTS)
    assert "no_change" in lines[0] and "area_trend" in lines[0]
    assert re.search(r"all\s+100\s+30\.00%\s+40\.00%\s+35\.00%\s+15\.00%\s+FLAG >25%", lines[1])
