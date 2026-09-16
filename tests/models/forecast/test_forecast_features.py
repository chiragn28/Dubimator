import math
from datetime import date, timedelta

import pandas as pd
import polars as pl
import pytest
from forecast_fixtures import prepared_history

from models.forecast.config import CATEGORICAL, ForecastConfig
from models.forecast.features import (
    CORE_FEATURES,
    OTHER_PROJECT,
    add_core_features,
    feature_coverage,
    fit_categories,
    to_matrix,
)
from models.forecast.targets import build_targets

ROWS, DATA_END = prepared_history()


def core(rows, data_end=DATA_END):
    frame, _ = build_targets(rows, data_end)
    return add_core_features(frame)


FRAME = core(ROWS)


def test_core_features_exist_and_keep_row_order():
    assert set(CORE_FEATURES) <= set(FRAME.columns)
    assert FRAME["row_id"].to_list() == ROWS["row_id"].to_list()
    assert not any(name.startswith("infra_") for name in CORE_FEATURES)


def test_area_momentum_matches_the_known_growth():
    late = FRAME.filter(
        (pl.col("area_id") == 1)
        & (pl.col("sub_kind") == "flat")
        & (pl.col("instance_date") > date(2019, 1, 1))
    )
    # area 1 grows 0.005 a month in log price
    assert late["area_mom_3m"].median() == pytest.approx(0.005 * 91 / 30.4375, abs=0.01)
    assert late["area_mom_12m"].median() == pytest.approx(0.005 * 365 / 30.4375, abs=0.01)
    assert late["area_mom_36m"].median() == pytest.approx(0.005 * 1096 / 30.4375, abs=0.02)
    early = FRAME.filter(pl.col("instance_date") < date(2017, 1, 1))
    assert early["area_mom_36m"].null_count() == early.height  # no window 3 years back yet
    city = FRAME.filter(
        (pl.col("sub_kind") == "flat") & (pl.col("instance_date") > date(2019, 1, 1))
    )
    assert 0.005 * 12 - 0.01 < city["city_mom_12m"].median() < 0.009 * 12 + 0.01


def test_activity_counts_and_recency():
    late = FRAME.filter(
        (pl.col("sub_kind") == "flat") & (pl.col("instance_date") > date(2019, 1, 1))
    )
    assert late["building_sales_12m"].median() == pytest.approx(104, abs=6)  # 2 a week
    assert late["area_share_12m"].median() == pytest.approx(1 / 3, abs=0.02)
    assert late["days_since_building_sale"].max() <= 13
    assert late["days_since_building_sale"].min() >= 1
    villas = FRAME.filter(pl.col("sub_kind") == "villa")
    assert villas["building_sales_12m"].null_count() == villas.height
    assert villas["days_since_building_sale"].null_count() == villas.height
    assert (
        villas.filter(pl.col("instance_date") > date(2016, 1, 1))["days_since_area_sale"].max()
        <= 13
    )


def test_building_age_proxy_uses_only_earlier_sales():
    tower = FRAME.filter(pl.col("building_key") == "1|tower 1 0").sort("instance_date")
    first_day = tower["instance_date"][0]
    assert (
        tower.filter(pl.col("instance_date") == first_day)["building_age_proxy_years"].null_count()
        > 0
    )
    later = tower.filter(pl.col("instance_date") > date(2017, 1, 1)).row(0, named=True)
    expected = (later["instance_date"] - first_day).days / 365.25
    assert later["building_age_proxy_years"] == pytest.approx(expected)


def test_property_columns():
    row = FRAME.filter(pl.col("base_level").is_not_null()).row(0, named=True)
    assert row["ln_base_ppsm"] == pytest.approx(math.log(row["base_ppsm"]))
    assert row["base_level_building"] in (0.0, 1.0)
    assert row["log_area_sqm"] == pytest.approx(math.log(row["area_sqm"]))
    assert row["off_plan"] == float(row["reg_type"] == "off_plan")
    assert row["area_code"] == str(row["area_id"])
    assert row["project_code"] == row["project_name"]


def test_features_never_look_at_sales_on_or_after_t():
    cutoff = date(2019, 6, 1)
    later = pl.col("instance_date") >= cutoff
    changed = ROWS.with_columns(
        pl.when(later).then(pl.col("ppsm") * 3).otherwise(pl.col("ppsm")).alias("ppsm")
    ).filter(~later | (pl.col("row_id") % 2 == 0))
    before = FRAME.filter(~later).select("transaction_id", *CORE_FEATURES).sort("transaction_id")
    after = (
        core(changed).filter(~later).select("transaction_id", *CORE_FEATURES).sort("transaction_id")
    )
    assert before.equals(after)


def test_query_rows_see_history_but_change_nothing():
    day = DATA_END + timedelta(days=1)
    template = ROWS.filter(pl.col("building_name").is_not_null()).row(-1, named=True)
    query = pl.DataFrame(
        [{**template, "transaction_id": "query", "instance_date": day, "ppsm": None,
          "row_id": ROWS.height}],
        schema=ROWS.schema,
    )  # fmt: skip
    with_query = core(pl.concat([ROWS, query]))
    others = with_query.filter(pl.col("transaction_id") != "query").select(
        "transaction_id", *CORE_FEATURES
    )
    assert others.equals(FRAME.select("transaction_id", *CORE_FEATURES))
    row = with_query.filter(pl.col("transaction_id") == "query").row(0, named=True)
    assert row["base_level"] == "building"
    recent = ROWS.filter(
        (pl.col("building_name") == template["building_name"])
        & (pl.col("instance_date") >= day - timedelta(days=92))
    )
    assert row["base_n"] == recent.height
    assert row["base_ppsm"] == pytest.approx(recent["ppsm"].median())


def test_to_matrix_orders_columns_and_fixes_categories():
    config = ForecastConfig()
    categories = fit_categories(FRAME, config)
    assert categories["area_code"] == ["1", "2", "3"]
    assert categories["project_code"][-1] == OTHER_PROJECT
    odd = FRAME.head(3).with_columns(
        pl.Series("area_code", ["1", "99", None]),
        pl.Series("project_code", ["Project 1", "Nowhere", None]),
    )
    matrix = to_matrix(odd, categories, CORE_FEATURES)
    assert list(matrix.columns) == list(CORE_FEATURES)
    for name in CATEGORICAL:
        assert isinstance(matrix[name].dtype, pd.CategoricalDtype)
        assert list(matrix[name].cat.categories) == categories[name]
    assert matrix["area_code"].isna().tolist() == [False, True, True]
    assert matrix["project_code"].tolist()[:2] == ["Project 1", OTHER_PROJECT]
    assert pd.isna(matrix["project_code"].tolist()[2])
    numeric = [name for name in CORE_FEATURES if name not in CATEGORICAL]
    assert all(matrix[name].dtype == "float64" for name in numeric)
    with pytest.raises(KeyError, match="infra_active_mall"):
        to_matrix(FRAME, categories)


def test_rare_projects_become_other():
    config = ForecastConfig(min_project_rows=10**9)
    assert fit_categories(FRAME, config)["project_code"] == [OTHER_PROJECT]


def test_feature_coverage():
    coverage = feature_coverage(FRAME, CORE_FEATURES)
    assert set(coverage) == set(CORE_FEATURES)
    assert coverage["log_area_sqm"] == 1.0
    assert 0.0 < coverage["area_mom_36m"] < 1.0
