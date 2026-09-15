from datetime import date

import polars as pl
import pytest

from models.price.data import RAW_SCHEMA

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
