import dataclasses
import math
import re
from datetime import date, timedelta

import numpy as np
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

from models.forecast.config import FEATURES, ForecastConfig
from models.forecast.features import build_dataset
from models.forecast.predict import (
    SNAPSHOT_COLUMNS,
    Forecaster,
    build_snapshot,
    describe,
    driver_text,
)
from models.forecast.rows import EXCLUDED_SCHEMA
from models.price.predictor import PriceInputError, PriceRequest

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
PLAIN_FLAT = {k: v for k, v in MARINA_FLAT.items() if k != "property_id"}
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
    return rows, data_end, projects, small_model(frame, data_end=data_end)


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
    assert snapshot.select("area_id", "market_kind", "building_key").is_unique().all()


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


def test_a_champion_whose_latest_gate_failed_is_not_served(world):
    model = world[3]
    gates = {**GATES, "3m": {"status": "failed", "reason": "3m resale MAPE 16.0% exceeds"}}
    result = forecaster(world, models={"3m": (model, "4")}, gates=gates).forecast(MARINA_FLAT)
    assert result["forecast_3m"] == {
        "status": "not_deployed", "reason": "3m resale MAPE 16.0% exceeds",
    }  # fmt: skip
    assert result["model_versions"] == {"price": "7"}
    assert result["key_drivers"] == []
    short = {**GATES, "3m": {"status": "insufficient_data", "reason": "3m: only 1 folds"}}
    blocked = forecaster(world, gates=short).forecast(MARINA_FLAT)["forecast_3m"]
    assert blocked == {"status": "not_deployed", "reason": "3m: only 1 folds"}


def test_a_stale_champion_is_not_served(world):
    _rows, data_end, _projects, model = world
    stale = dataclasses.replace(model, metadata={"data_end": "2022-12-31"})
    result = forecaster(world, models={"3m": (stale, "4")}).forecast(MARINA_FLAT)
    assert result["forecast_3m"] == {
        "status": "not_deployed",
        "reason": (
            "the 3m model was trained on data ending 2022-12-31; "
            f"retrain it on data ending {data_end.isoformat()}"
        ),
    }
    assert result["model_versions"] == {"price": "7"}
    assert result["key_drivers"] == []


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


def test_key_drivers_come_from_shap(world, monkeypatch):
    from models.forecast import predict as predict_module

    engine = forecaster(world)
    features = engine._features(PriceRequest.parse(PLAIN_FLAT), 1, "flat")
    monkeypatch.setattr(predict_module, "FEATURES", ("area_code", *FEATURES[1:]))
    monkeypatch.setattr(
        type(engine._model("3m")),
        "contributions",
        lambda self, frame: np.array([[0.5] + [0.0] * len(FEATURES)]),
    )
    assert engine._drivers(features) == ["Location: Dubai Marina (+64.9% to the 3m forecast)"]
    monkeypatch.undo()
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
    names = {1: "Dubai Marina"}
    assert describe("area_code", "1", names) == "Location: Dubai Marina"
    assert describe("area_code", "9", names) == "Location: area 9"
    assert driver_text("area_code", "1", 0.01, "1y", names) == (
        "Location: Dubai Marina (+1.0% to the 1y forecast)"
    )
    kinds = {
        "flat": "apartment",
        "hotel_apartment": "hotel apartment",
        "townhouse": "townhouse",
        "villa": "villa (built-up area)",
        "villa_plot": "villa (plot area)",
    }
    for kind, text in kinds.items():
        assert describe("market_kind", kind) == f"Property kind: {text}"
    assert describe("project_code", "Marina Gate") == "Project: Marina Gate"
    assert describe("project_code", None) == "Project: unknown"


def test_exclusions_name_the_area_and_segment(world):
    excluded = pl.DataFrame(
        {
            "area_id": [1, 1, 1, 2],
            "sub_kind": ["flat", "flat", "villa", "flat"],
            "market_kind": ["flat", "flat", "villa", "flat"],
            "reg_type": ["off_plan", "off_plan", "ready", "off_plan"],
            "month": [date(2022, 1, 1)] * 4,
            "reason": ["outlier"] * 4,
        },
        schema=EXCLUDED_SCHEMA,
    )
    repeats = pl.DataFrame(
        {
            "area_id": [1, 1, 1, 1],
            "sub_kind": ["flat"] * 4,
            "market_kind": ["flat"] * 4,
            "reg_type": ["off_plan", "off_plan", "off_plan", "ready"],
            "month": [date(2022, 2, 1)] * 4,
            "reason": ["repeat_sale"] * 4,
        },
        schema=EXCLUDED_SCHEMA,
    )
    result = forecaster(world, excluded=excluded).forecast(MARINA_FLAT)
    assert result["exclusions_applied"] == ["Dropped 2 off-plan outliers in Dubai Marina"]
    both = forecaster(world, excluded=pl.concat([excluded, repeats])).forecast(MARINA_FLAT)
    assert both["exclusions_applied"] == [
        "Dropped 2 off-plan outliers in Dubai Marina",
        "Dropped 3 off-plan repeat sales in Dubai Marina",
        "Dropped 1 ready repeat sale in Dubai Marina",
    ]


def test_plot_villas_use_the_plot_market(tmp_path, monkeypatch):
    rows, data_end = prepared_history(
        areas=1, buildings_per_area=1, start=date(2022, 1, 3), end=date(2023, 3, 13),
        plot_villas=True,
    )  # fmt: skip
    engine = Forecaster.build(
        rows, data_end, project_table(tmp_path), history_areas(), history_aliases(), {}, {},
        FakePrice(), pl.DataFrame(schema=EXCLUDED_SCHEMA), CONFIG,
    )  # fmt: skip
    kinds = set(engine.snapshot.filter(pl.col("building_key").is_null())["market_kind"])
    assert kinds == {"flat", "villa", "villa_plot"}
    seen = []
    features = Forecaster._features

    def spy(self, request, area_id, market_kind):
        frame = features(self, request, area_id, market_kind)
        seen.append(frame.row(0, named=True))
        return frame

    monkeypatch.setattr(Forecaster, "_features", spy)
    villa = {"area": "Dubai Marina", "property_kind": "villa", "status": "ready",
             "size_sqm": 600.0, "bedrooms": 4}  # fmt: skip
    engine.forecast({**villa, "size_basis": "plot"})
    engine.forecast(villa)
    plot, built = seen
    assert plot["market_kind"] == "villa_plot"
    assert built["market_kind"] == "villa"
    since = data_end + timedelta(days=1) - timedelta(days=92)
    recent = rows.filter(
        (pl.col("market_kind") == "villa_plot") & (pl.col("instance_date") >= since)
    )
    assert plot["base_ppsm"] == pytest.approx(recent["ppsm"].median())
    assert plot["base_ppsm"] < 0.6 * built["base_ppsm"]  # plot prices per m2 are about half


def test_penthouses_have_no_bedroom_count(world):
    engine = forecaster(world)
    request = PriceRequest.parse({**PLAIN_FLAT, "bedrooms": 3, "is_penthouse": True})
    assert engine._features(request, 1, "flat")["bedrooms"].to_list() == [None]
    plain = PriceRequest.parse({**PLAIN_FLAT, "bedrooms": 3})
    assert engine._features(plain, 1, "flat")["bedrooms"].to_list() == [3.0]


def test_the_snapshot_matches_a_real_sale_the_day_after_the_data_end(world):
    rows, data_end, projects, _ = world
    day = data_end + timedelta(days=1)
    template = rows.filter(pl.col("building_name") == "Tower 1-0").row(-1, named=True)
    sale = {**template, "transaction_id": "next-day", "instance_date": day,
            "row_id": rows.height}  # fmt: skip
    frame, _ = build_dataset(pl.concat([rows, pl.DataFrame([sale], schema=rows.schema)]),
                             data_end, projects)  # fmt: skip
    real = frame.filter(pl.col("transaction_id") == "next-day").row(0, named=True)
    snapshot = build_snapshot(rows, data_end, projects)
    row = snapshot.filter(pl.col("building_key") == real["building_key"]).row(0, named=True)
    assert row["market_kind"] == real["market_kind"]
    for column in SNAPSHOT_COLUMNS:
        if isinstance(real[column], float):
            assert row[column] == pytest.approx(real[column], nan_ok=True), column
        else:
            assert row[column] == real[column], column


def test_area_id_requests_match_area_name_requests(world):
    engine = forecaster(world)
    by_id = {k: v for k, v in MARINA_FLAT.items() if k != "area"} | {"area_id": 1}
    assert engine.forecast(by_id) == engine.forecast(MARINA_FLAT)
    with pytest.raises(PriceInputError, match="area_id"):
        engine.forecast({**by_id, "area_id": 99})


def test_areas_without_recent_sales_cannot_be_forecast(world):
    rows, data_end, projects, model = world
    cut = data_end - timedelta(days=120)
    quiet_villas = (pl.col("area_id") == 3) & (pl.col("sub_kind") == "villa")
    thinned = rows.filter(~(quiet_villas & (pl.col("instance_date") > cut)))
    engine = forecaster((thinned, data_end, projects, model))
    villa = {"area": "Arabian Ranches", "property_kind": "villa", "status": "ready",
             "size_sqm": 300.0, "bedrooms": 4}  # fmt: skip
    with pytest.raises(PriceInputError, match="not enough sales in the last 3 months"):
        engine.forecast(villa)


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
