import math
from datetime import date, timedelta

# Import lightgbm before psycopg2 (pulled in transitively below via models.price.data) has a
# chance to load: on this Windows machine, psycopg2's native libpq loading ahead of scikit-learn
# (imported by models.price.features -> models.price.split) corrupts state that crashes a later
# lightgbm Dataset construction with an access violation. Harmless once lightgbm is loaded first.
import lightgbm  # noqa: F401
import numpy as np
import polars as pl
import pytest

from models.price.data import RAW_SCHEMA
from models.price.features import derive_segments

BASE_HOME = {
    "transaction_id": "t",
    "instance_date": date(2020, 1, 15),
    "property_type": "unit",
    "property_sub_type": "Flat",
    "reg_type": "ready",
    "area_id": 1,
    "building_name": "Tower A",
    "project_name": "Project P",
    "rooms": "1 B/R",
    "has_parking": True,
    "area_sqm": 80.0,
    "price_aed": 800_000.0,
    "ingest_run_id": 1,
    "is_clean": True,
}


def build_raw_homes(*overrides: dict) -> pl.DataFrame:
    """One raw home row per override dict, each starting from BASE_HOME."""
    records = [{**BASE_HOME, "transaction_id": f"t{i}", **row} for i, row in enumerate(overrides)]
    return pl.DataFrame(records, schema=RAW_SCHEMA)


@pytest.fixture
def raw_homes():
    return build_raw_homes


def build_synthetic_homes(seed: int = 0, per_month: int = 6) -> pl.DataFrame:
    """Prepared home rows 2014-01..2023-03: prices rise 0.4%/month, area and villa premia."""
    rng = np.random.default_rng(seed)
    records = []
    month = date(2014, 1, 1)
    while month <= date(2023, 3, 1):
        months_since = (month.year - 2014) * 12 + month.month - 1
        level = math.log(9_000.0) + 0.004 * months_since
        for j in range(per_month):
            villa = j % 3 == 2
            area_id = int(rng.integers(1, 4))
            size = float(rng.uniform(250.0, 600.0) if villa else rng.uniform(40.0, 150.0))
            ln_ppsqm = level + 0.15 * area_id + (0.2 if villa else 0.0) + rng.normal(0.0, 0.1)
            records.append(
                {
                    "transaction_id": f"s{len(records)}",
                    "instance_date": month.replace(day=int(rng.integers(1, 28))),
                    "property_type": "villa" if villa else "unit",
                    "property_sub_type": "Villa" if villa else "Flat",
                    "reg_type": "ready",
                    "area_id": area_id,
                    "building_name": None if villa else f"Tower {area_id}{int(rng.integers(0, 3))}",
                    "project_name": f"Project {area_id}",
                    "rooms": f"{int(rng.integers(1, 4))} B/R",
                    "has_parking": bool(rng.integers(0, 2)),
                    "area_sqm": round(size, 1),
                    "price_aed": round(math.exp(ln_ppsqm) * size, -2),
                    "ingest_run_id": 1,
                    "is_clean": not (month >= date(2022, 11, 1) and j == 0),
                }
            )
        month = (month.replace(day=28) + timedelta(days=4)).replace(day=1)
    return derive_segments(pl.DataFrame(records, schema=RAW_SCHEMA))


@pytest.fixture
def synthetic_homes():
    return build_synthetic_homes
