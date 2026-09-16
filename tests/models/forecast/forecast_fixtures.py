"""Synthetic prepared home rows with known growth, shaped like models.price.data.load_homes."""

import math
from datetime import date, timedelta

import numpy as np
import polars as pl

from models.price.data import RAW_SCHEMA
from models.price.features import derive_segments

AREA_NAMES = {1: "Dubai Marina", 2: "Jumeirah Village Circle", 3: "Arabian Ranches"}


def monthly_growth(area_id: int) -> float:
    return 0.003 + 0.002 * area_id


def build_history(
    seed: int = 0,
    areas: int = 3,
    buildings_per_area: int = 3,
    per_building_per_week: int = 2,
    start: date = date(2015, 1, 5),
    end: date = date(2023, 3, 13),
    plot_villas: bool = False,
) -> pl.DataFrame:
    """Weekly sales per building, plus one villa sale per area per week (area-level only).

    ln ppsm = ln(9,000) + 0.1 * area + monthly_growth(area) * months + N(0, 0.03).
    Off-plan units are every third building; bedrooms cycle 0..3. With plot_villas, each area
    also sells one plot-priced villa a week (null sub-type, 600 m2 plot, about half the ppsm).
    """
    rng = np.random.default_rng(seed)
    records = []
    day = start
    while day <= end:
        months = (day - start).days / 30.4375
        for area_id in range(1, areas + 1):
            level = math.log(9_000.0) + 0.1 * area_id + monthly_growth(area_id) * months
            for building in range(buildings_per_area):
                for sale in range(per_building_per_week):
                    size = float(60 + 20 * ((building + sale) % 4))
                    ppsm = math.exp(level + rng.normal(0.0, 0.03))
                    records.append(
                        {
                            "transaction_id": f"s{len(records)}",
                            "instance_date": day + timedelta(days=int(rng.integers(0, 7))),
                            "property_type": "unit",
                            "property_sub_type": "Flat",
                            "reg_type": "off_plan" if building % 3 == 0 else "ready",
                            "area_id": area_id,
                            "building_name": f"Tower {area_id}-{building}",
                            "project_name": f"Project {area_id}",
                            "rooms": "Studio" if sale % 4 == 0 else f"{1 + sale % 3} B/R",
                            "has_parking": True,
                            "area_sqm": size,
                            "price_aed": round(ppsm * size, 0),
                            "ingest_run_id": 1,
                            "is_clean": True,
                        }
                    )
            villa_ppsm = math.exp(level + 0.2 + rng.normal(0.0, 0.03))
            records.append(
                {
                    "transaction_id": f"v{len(records)}",
                    "instance_date": day,
                    "property_type": "villa",
                    "property_sub_type": "Villa",
                    "reg_type": "ready",
                    "area_id": area_id,
                    "building_name": None,
                    "project_name": f"Project {area_id}",
                    "rooms": "4 B/R",
                    "has_parking": True,
                    "area_sqm": 300.0,
                    "price_aed": round(villa_ppsm * 300.0, 0),
                    "ingest_run_id": 1,
                    "is_clean": True,
                }
            )
            if plot_villas:
                records.append(
                    {
                        **records[-1],
                        "transaction_id": f"p{len(records)}",
                        "property_sub_type": None,
                        "area_sqm": 600.0,
                        "price_aed": round(villa_ppsm * 300.0 * 1.02, 0),
                    }
                )
        day += timedelta(days=7)
    frame = pl.DataFrame(records, schema=RAW_SCHEMA)
    return derive_segments(frame).sort("instance_date", "transaction_id")


def prepared_history(**kwargs):
    """build_history rows run through prepare_rows: (rows with ppsm and row_id, data_end)."""
    from models.forecast.config import ForecastConfig
    from models.forecast.rows import prepare_rows

    rows, _ = prepare_rows(build_history(**kwargs), ForecastConfig())
    return rows, rows["instance_date"].max()


def history_areas():
    return pl.DataFrame(
        {"area_id": list(AREA_NAMES), "name_en": list(AREA_NAMES.values())},
        schema={"area_id": pl.Int64, "name_en": pl.Utf8},
    )


def history_aliases():
    from ingestion.normalize import match_key

    return pl.DataFrame(
        {
            "alias_key": [match_key(name) for name in AREA_NAMES.values()],
            "area_id": list(AREA_NAMES),
        },
        schema={"alias_key": pl.Utf8, "area_id": pl.Int64},
    )


def fake_load_rows(rows_and_end, quality=None):
    """A stand-in for models.forecast.rows.load_rows over prepared synthetic rows."""
    from models.forecast.rows import EXCLUDED_SCHEMA, DataQuality

    rows, data_end = rows_and_end
    if quality is None:
        quality = DataQuality(
            rows.height, {"bad_price": 0}, rows.height, pl.DataFrame(schema=EXCLUDED_SCHEMA)
        )

    def load(settings, config):
        return rows, quality, data_end, history_areas(), history_aliases()

    return load


def project_record(index, **overrides):
    from models.forecast.config import INFRA_TYPES

    record = {
        "project_id": f"P{index:02d}",
        "name": f"Project number {index}",
        "type": INFRA_TYPES[index % len(INFRA_TYPES)],
        "announced_date": "2016-01-01",
        "planned_completion_date": "2019-01-01",
        "actual_completion_date": "2019-06-01" if index % 2 else "",
        "affected_area_ids": "1;2",
        "source_url": f"https://example.org/projects/{index}",
        "source_accessed": "2026-09-16",
        "notes": "synthetic test row",
    }
    return {**record, **overrides}


def write_projects(directory, records):
    from models.forecast.infra import INFRA_COLUMNS

    path = directory / "projects.csv"
    pl.DataFrame(records, schema=dict.fromkeys(INFRA_COLUMNS, pl.Utf8)).write_csv(path)
    return path


def project_table(directory, count=15, **overrides):
    from models.forecast.infra import load_projects

    records = [project_record(index, **overrides) for index in range(count)]
    return load_projects(write_projects(directory, records))


class FakePrice:
    """Stands in for the Phase 3 PricePredictor: a fixed 1,000,000 AED estimate."""

    def __init__(self):
        self.requests = []

    def predict_one(self, request):
        from types import SimpleNamespace

        self.requests.append(request)
        return SimpleNamespace(
            estimate_aed=1_000_000.0, range_80=(900_000.0, 1_100_000.0), model_version="7"
        )


def small_model(frame, horizon="3m", rounds=30, data_end=None):
    from models.forecast.config import ForecastConfig
    from models.forecast.features import fit_categories
    from models.forecast.model import ForecastModel
    from models.forecast.train import fit_booster

    config = ForecastConfig(device="cpu")
    train = frame.filter(pl.col(f"growth_{horizon}").is_not_null())
    categories = fit_categories(train, config)
    booster, used = fit_booster(
        train, None, {}, categories, "cpu", config, f"growth_{horizon}", rounds
    )
    return ForecastModel(
        horizon=horizon,
        booster=booster,
        rounds=used,
        categories=categories,
        intervals={"_pooled": 0.05, "ready_unit": 0.04},
        area_rows=dict(train.group_by("area_id").len().iter_rows()),
        metadata={"data_end": data_end.isoformat()} if data_end else {},
    )
