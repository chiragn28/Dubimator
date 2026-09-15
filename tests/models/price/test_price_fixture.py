from datetime import date
from pathlib import Path

import polars as pl

from ingestion.schema import EXPECTED_COLUMNS

FIXTURE = Path("tests/fixtures/price_sample.csv")


def test_price_fixture_is_real_dld_shaped_and_spans_every_split():
    frame = pl.read_csv(FIXTURE, infer_schema=False)
    assert set(frame.columns) == set(EXPECTED_COLUMNS)
    assert 2_500 <= frame.height <= 3_100
    days = frame["instance_date"].str.strptime(pl.Date, "%d-%m-%Y")
    assert days.min() < date(2015, 1, 1)
    assert days.max() >= date(2022, 11, 1)
    assert set(frame["property_type_en"].unique()) == {"Unit", "Villa"}
