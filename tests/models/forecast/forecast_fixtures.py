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
) -> pl.DataFrame:
    """Weekly sales per building, plus one villa sale per area per week (area-level only).

    ln ppsm = ln(9,000) + 0.1 * area + monthly_growth(area) * months + N(0, 0.03).
    Off-plan units are every third building; bedrooms cycle 0..3.
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
        day += timedelta(days=7)
    frame = pl.DataFrame(records, schema=RAW_SCHEMA)
    return derive_segments(frame).sort("instance_date", "transaction_id")
