from pathlib import Path

import polars as pl

from ingestion.normalize import to_typed
from ingestion.rules import classify
from ingestion.schema import read_raw

FIXTURE = Path(__file__).parent.parent / "fixtures" / "dld_sample.csv"


def test_fixture_is_valid_and_covers_the_rules():
    raw = read_raw(FIXTURE)
    assert 250 <= raw.height <= 400
    result = classify(to_typed(raw))
    reasons = set(result["exclusion_reason"].drop_nulls().to_list())
    assert {
        "mortgage",
        "gift",
        "non_market_procedure",
        "missing_price",
        "price_below_floor",
    } <= reasons
    assert result["exclusion_reason"].null_count() > 100
    assert result.filter(pl.col("instance_date").is_null()).height == 5
    assert set(result["property_type"].to_list()) == {"unit", "villa", "land", "building"}
