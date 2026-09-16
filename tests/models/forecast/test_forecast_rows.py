import dataclasses
from datetime import date

import polars as pl
import pytest
from forecast_fixtures import build_history

from models.forecast.config import (
    FEATURES,
    FORBIDDEN_FEATURES,
    HORIZON_SPECS,
    HORIZONS,
    ForecastConfig,
)
from models.forecast.rows import DataQuality, prepare_rows, screen_outliers, validate_rows

SMALL = build_history(areas=2, buildings_per_area=2, start=date(2020, 1, 6), end=date(2020, 6, 29))


def test_config_is_consistent():
    assert HORIZONS == ("3m", "1y", "3y")
    assert [HORIZON_SPECS[h].end_days for h in HORIZONS] == [107, 396, 1126]
    assert not set(FEATURES) & set(FORBIDDEN_FEATURES)
    assert {"nearest_metro", "growth_1y", "ppsm"} <= set(FORBIDDEN_FEATURES)
    assert len(FEATURES) == len(set(FEATURES))


def _with(frame: pl.DataFrame, index: int, **values) -> pl.DataFrame:
    rows = frame.to_dicts()
    rows[index] = {**rows[index], **values}
    return pl.DataFrame(rows, schema=frame.schema)


def test_validate_rows_drops_each_bad_value_with_its_reason():
    frame = SMALL.head(10)
    frame = _with(frame, 0, price_aed=0.0)
    frame = _with(frame, 1, area_sqm=None)
    frame = _with(frame, 2, instance_date=None)
    frame = _with(frame, 3, area_id=None)
    frame = _with(frame, 4, transaction_id=frame["transaction_id"][5])
    frame = _with(frame, 6, is_clean=False)
    kept, dropped = validate_rows(frame)
    assert dropped == {
        "not_clean": 1,
        "bad_price": 1,
        "bad_size": 1,
        "bad_date": 1,
        "missing_area": 1,
        "duplicate_transaction_id": 1,
        "repeat_sale": 0,
    }
    assert kept.height == 4


def test_validate_rows_keeps_the_last_of_a_repeat_sale():
    frame = SMALL.filter(pl.col("building_name").is_not_null()).head(3)
    first = frame.row(0, named=True)
    copy = {**first, "transaction_id": "copy"}
    frame = pl.concat([frame, pl.DataFrame([copy], schema=frame.schema)])
    kept, dropped = validate_rows(frame)
    assert dropped["repeat_sale"] == 1
    assert "copy" in kept["transaction_id"].to_list()
    assert first["transaction_id"] not in kept["transaction_id"].to_list()


def test_outliers_are_screened_within_area_kind_month():
    frame = SMALL.with_columns((pl.col("price_aed") / pl.col("area_sqm")).alias("ppsm"))
    in_march = (pl.col("instance_date").dt.month() == 3) & pl.col("building_name").is_not_null()
    victim = frame.filter(in_march).row(0, named=True)
    frame = frame.with_columns(
        pl.when(pl.col("transaction_id") == victim["transaction_id"])
        .then(pl.col("ppsm") * 5)
        .otherwise(pl.col("ppsm"))
        .alias("ppsm")
    )
    kept, excluded, unscreened = screen_outliers(frame, ForecastConfig())
    assert excluded.height >= 1  # the planted one; N(0, 0.03) noise rarely adds another
    assert (excluded["area_id"] == victim["area_id"]).any()
    assert set(excluded["reason"]) == {"outlier"}
    assert victim["transaction_id"] not in kept["transaction_id"].to_list()
    assert unscreened > 0  # villa groups have ~4 sales a month, under the 10-sale minimum


def test_prepare_rows_reports_drops_and_adds_ids():
    frame = _with(SMALL, 0, price_aed=-1.0)
    rows, quality = prepare_rows(frame, ForecastConfig())
    assert isinstance(quality, DataQuality)
    assert quality.loaded == SMALL.height
    assert quality.kept == rows.height == SMALL.height - sum(quality.dropped.values())
    assert quality.dropped["bad_price"] == 1
    assert rows["row_id"].to_list() == list(range(rows.height))
    assert (rows["ppsm"] > 0).all()
    lines = quality.lines()
    total = sum(quality.dropped.values())
    share = 100 * total / quality.loaded
    assert lines[0] == f"Dropped {total:,} of {quality.loaded:,} rows ({share:.1f}%)"
    assert any(line.strip().startswith("bad_price") for line in lines)
    assert quality.drop_share == pytest.approx(total / quality.loaded)
    assert quality.to_dict()["dropped"]["bad_price"] == 1


def test_sampling_is_seeded():
    config = dataclasses.replace(ForecastConfig(), sample_rows=50)
    first, _ = prepare_rows(SMALL, config)
    again, _ = prepare_rows(SMALL, config)
    other, _ = prepare_rows(SMALL, dataclasses.replace(config, seed=8))
    assert first.height <= 50
    assert first["transaction_id"].to_list() == again["transaction_id"].to_list()
    assert first["transaction_id"].to_list() != other["transaction_id"].to_list()


def test_load_rows_reads_postgres(pg_test_db):
    from pathlib import Path

    from ingestion.pipeline import run_pipeline
    from models.forecast.rows import load_rows

    run_pipeline(Path("tests/fixtures/price_sample.csv"), pg_test_db)
    rows, quality, data_end, areas, aliases = load_rows(pg_test_db, ForecastConfig())
    assert rows.height == quality.kept > 0
    assert data_end == rows["instance_date"].max()
    assert areas.height > 0 and aliases.height > 0
    assert {"ppsm", "row_id", "sub_kind", "reg_type"} <= set(rows.columns)
