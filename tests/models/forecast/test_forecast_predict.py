import dataclasses
import math
import re
from datetime import date, timedelta

import polars as pl
import pytest
from forecast_fixtures import (
    FakePrice,
    history_aliases,
    history_areas,
    prepared_history,
    project_table,
    small_model,
)

from models.forecast.config import ForecastConfig
from models.forecast.features import build_dataset
from models.forecast.predict import Forecaster, build_snapshot, describe, driver_text
from models.forecast.rows import EXCLUDED_SCHEMA
from models.price.predictor import PriceInputError

CONFIG = ForecastConfig(device="cpu")
MARINA_FLAT = {
    "property_id": "p-1",
    "area": "Dubai Marina",
    "building": "Tower 1-0",
    "property_kind": "apartment",
    "status": "ready",
    "size_sqm": 80.0,
    "bedrooms": 1,
}
GATES = {
    "3m": {"status": "passed", "reason": "passed"},
    "1y": {"status": "failed", "reason": "1y resale MAPE 31.2% exceeds the 20% gate"},
    "3y": {"status": "insufficient_data", "reason": "3y: only 0 walk-forward folds (needs 2)"},
}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    rows, data_end = prepared_history()
    projects = project_table(tmp_path_factory.mktemp("projects"))
    frame, _ = build_dataset(rows, data_end, projects)
    return rows, data_end, projects, small_model(frame)


def forecaster(world, models=None, gates=GATES, excluded=None, config=CONFIG):
    rows, data_end, projects, model = world
    models = models if models is not None else {"3m": (model, "4")}
    excluded = excluded if excluded is not None else pl.DataFrame(schema=EXCLUDED_SCHEMA)
    return Forecaster.build(
        rows, data_end, projects, history_areas(), history_aliases(), models, gates,
        FakePrice(), excluded, config,
    )  # fmt: skip


def test_snapshot_matches_a_direct_computation(world):
    rows, data_end, projects, _ = world
    snapshot = build_snapshot(rows, data_end, projects)
    tower = snapshot.filter(pl.col("building_key") == "1|tower 1 0").row(0, named=True)
    since = data_end + timedelta(days=1) - timedelta(days=92)
    recent = rows.filter(
        (pl.col("building_name") == "Tower 1-0") & (pl.col("instance_date") >= since)
    )
    assert tower["base_level"] == "building"
    assert tower["base_n"] == recent.height
    assert tower["base_ppsm"] == pytest.approx(recent["ppsm"].median())
    assert snapshot.filter(pl.col("building_key").is_null()).height == 6  # 3 areas x flat/villa
    assert snapshot.select("area_id", "sub_kind", "building_key").is_unique().all()


def test_forecast_json_shape(world):
    result = forecaster(world).forecast(MARINA_FLAT)
    assert list(result) == [
        "property_id", "as_of", "current_estimate_aed", "current_range_80",
        "forecast_3m", "forecast_1y", "forecast_3y",
        "key_drivers", "exclusions_applied", "model_versions",
    ]  # fmt: skip
    assert result["property_id"] == "p-1"
    assert result["as_of"] == world[1].isoformat()
    assert result["current_estimate_aed"] == 1_000_000
    assert result["current_range_80"] == [900_000, 1_100_000]
    three = result["forecast_3m"]
    assert set(three) == {"point", "ci_low", "ci_high", "confidence"}
    assert three["ci_low"] < three["point"] < three["ci_high"]
    growth = math.log(three["point"] / 1_000_000)
    assert three["ci_high"] == pytest.approx(1_000_000 * math.exp(growth + 0.04), abs=2)
    assert three["confidence"] == "HIGH"
    assert result["forecast_1y"] == {
        "status": "not_deployed", "reason": "1y resale MAPE 31.2% exceeds the 20% gate",
    }  # fmt: skip
    assert result["forecast_3y"]["reason"] == "3y: only 0 walk-forward folds (needs 2)"
    assert result["model_versions"] == {"price": "7", "forecast_3m": "4"}


def test_no_number_without_a_range(world):
    result = forecaster(world).forecast(MARINA_FLAT)
    for name in ("forecast_3m", "forecast_1y", "forecast_3y"):
        block = result[name]
        assert ("point" in block) == ("ci_low" in block) == ("ci_high" in block)
    assert len(result["current_range_80"]) == 2


def test_unknown_building_falls_back_to_the_area(world):
    result = forecaster(world).forecast({**MARINA_FLAT, "building": "Nowhere Tower"})
    assert result["forecast_3m"]["confidence"] == "MEDIUM"


def test_thin_areas_get_the_low_message(world):
    _rows, _data_end, _projects, model = world
    thin = dataclasses.replace(model, area_rows={})
    result = forecaster(world, models={"3m": (thin, "4")}).forecast(MARINA_FLAT)
    block = result["forecast_3m"]
    assert block["confidence"] == "LOW"
    pct = round(100 * (block["ci_high"] - block["ci_low"]) / 2 / block["point"])
    assert block["message"] == (
        "Limited comparable data for this property type/area. "
        f"Estimate has wide uncertainty (±{pct}%)."
    )
    assert re.fullmatch(r".*\(±\d+%\)\.", block["message"])


def test_key_drivers_come_from_shap(world):
    drivers = forecaster(world).forecast(MARINA_FLAT)["key_drivers"]
    assert 1 <= len(drivers) <= 3
    assert all(re.search(r"\([+-]\d+\.\d% to the 3m forecast\)$", text) for text in drivers)
    silent = dataclasses.replace(CONFIG, min_driver_contribution=10.0)
    assert forecaster(world, config=silent).forecast(MARINA_FLAT)["key_drivers"] == []
    nothing = forecaster(world, models={}).forecast(MARINA_FLAT)
    assert nothing["key_drivers"] == []
    assert nothing["forecast_3m"] == {"status": "not_deployed", "reason": "no registered 3m model"}


def test_driver_text_templates():
    assert (
        describe("area_mom_12m", math.log(1.14)) == "Area prices rose 14% over the last 12 months"
    )
    assert describe("city_mom_3m", math.log(0.9)) == (
        "Dubai-wide prices for this kind fell 10% over the last 3 months"
    )
    assert describe("base_level_building", 1.0) == "Recent prices come from the same building"
    assert (
        describe("infra_active_metro_rail", 2.0) == "2 metro or rail projects under way in the area"
    )
    assert (
        describe("days_since_building_sale", None) == "Days since the building's last sale: unknown"
    )
    assert driver_text("area_mom_12m", math.log(1.14), 0.0305, "1y") == (
        "Area prices rose 14% over the last 12 months (+3.1% to the 1y forecast)"
    )
    assert driver_text("off_plan", 1.0, -0.02, "3m").endswith("(-2.0% to the 3m forecast)")


def test_exclusions_name_the_area_and_segment(world):
    excluded = pl.DataFrame(
        {
            "area_id": [1, 1, 1, 2],
            "sub_kind": ["flat", "flat", "villa", "flat"],
            "reg_type": ["off_plan", "off_plan", "ready", "off_plan"],
            "month": [date(2022, 1, 1)] * 4,
            "reason": ["outlier"] * 4,
        },
        schema=EXCLUDED_SCHEMA,
    )
    result = forecaster(world, excluded=excluded).forecast(MARINA_FLAT)
    assert result["exclusions_applied"] == ["Dropped 2 off-plan outliers in Dubai Marina"]


def test_invalid_requests_raise_price_input_errors(world):
    engine = forecaster(world)
    with pytest.raises(PriceInputError, match="as_of"):
        engine.forecast(MARINA_FLAT, as_of=date(2020, 1, 1))
    with pytest.raises(PriceInputError, match="area"):
        engine.forecast({**MARINA_FLAT, "area": "Atlantis"})
    with pytest.raises(PriceInputError, match="size_sqm"):
        engine.forecast({**MARINA_FLAT, "size_sqm": -1})
    with pytest.raises(PriceInputError, match="no sales history"):
        engine.forecast({**MARINA_FLAT, "property_kind": "townhouse"})
    assert engine.forecast(MARINA_FLAT, as_of=world[1])["as_of"] == world[1].isoformat()
