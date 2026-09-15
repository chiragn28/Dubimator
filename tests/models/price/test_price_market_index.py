import math
from datetime import date

import polars as pl
import pytest

from models.price.features import MarketIndex, MarketIndexError

U = "unit_ready_built_up"
SCHEMA = {
    "segment": pl.Utf8,
    "property_type": pl.Utf8,
    "size_basis": pl.Utf8,
    "instance_date": pl.Date,
    "price_aed": pl.Float64,
    "area_sqm": pl.Float64,
}


def make_sales(entries):
    """entries: (segment, day, ln_price_per_sqm) triples."""
    records = []
    for segment, day, value in entries:
        size_basis = "plot" if segment.endswith("_plot") else "built_up"
        records.append(
            {
                "segment": segment,
                "property_type": segment.split("_", 1)[0],
                "size_basis": size_basis,
                "instance_date": day,
                "price_aed": math.exp(value) * 100.0,
                "area_sqm": 100.0,
            }
        )
    return pl.DataFrame(records, schema=SCHEMA)


def index_row(index, segment, month):
    row = index.table.filter((pl.col("segment") == segment) & (pl.col("month") == month))
    assert row.height == 1
    return row["value"][0], row["source"][0]


def test_index_is_the_median_of_the_three_prior_months():
    sales = make_sales(
        [(U, date(2020, 1, 10), 1.0), (U, date(2020, 2, 10), 2.0),
         (U, date(2020, 3, 10), 3.0), (U, date(2020, 4, 10), 100.0)]
    )  # fmt: skip
    index = MarketIndex.fit(sales, date(2020, 4, 1), date(2020, 4, 1), min_sales=3)
    value, source = index_row(index, U, date(2020, 4, 1))
    assert value == pytest.approx(2.0)
    assert source == "segment_3"


def test_index_never_sees_same_month_or_later_prices():
    entries = [(U, date(2020, m, d), 1.0 + m / 10) for m in range(1, 7) for d in (5, 20)]
    base = MarketIndex.fit(make_sales(entries), date(2020, 3, 1), date(2020, 6, 1), min_sales=1)
    # Negative, not positive: with ln-price increasing monotonically each month, a
    # `+5.0` bump to the already-largest (most recent) month in a 3-month window can
    # never move a median (an order statistic) — it stays the largest either way, so
    # June's index would be identical before and after. `-5.0` moves May below the
    # rest of the window instead, which does change June's median, genuinely
    # exercising the invariant this test checks.
    perturbed_entries = [
        (s, day, v - (5.0 if day >= date(2020, 5, 1) else 0.0)) for s, day, v in entries
    ]
    perturbed = MarketIndex.fit(
        make_sales(perturbed_entries), date(2020, 3, 1), date(2020, 6, 1), min_sales=1
    )
    for month in (date(2020, 3, 1), date(2020, 4, 1), date(2020, 5, 1)):
        assert index_row(perturbed, U, month)[0] == pytest.approx(index_row(base, U, month)[0])
    assert index_row(perturbed, U, date(2020, 6, 1))[0] != pytest.approx(
        index_row(base, U, date(2020, 6, 1))[0]
    )


def test_window_widens_when_three_months_are_thin():
    entries = [(U, date(2019, 11, 1 + i), 1.0) for i in range(5)] + [(U, date(2020, 2, 1), 9.0)]
    index = MarketIndex.fit(make_sales(entries), date(2020, 4, 1), date(2020, 4, 1), min_sales=3)
    value, source = index_row(index, U, date(2020, 4, 1))
    assert (value, source) == (pytest.approx(1.0), "segment_6")


def test_pools_over_reg_type_when_the_segment_has_no_history():
    ready, off_plan = "villa_ready_built_up", "villa_off_plan_built_up"
    entries = [(ready, date(2020, m, 1), 4.0) for m in (1, 2, 3)] + [
        (off_plan, date(2020, 4, 2), 7.0)
    ]
    index = MarketIndex.fit(make_sales(entries), date(2020, 4, 1), date(2020, 4, 1), min_sales=3)
    value, source = index_row(index, off_plan, date(2020, 4, 1))
    assert (value, source) == (pytest.approx(4.0), "pooled_3")


def test_carries_the_last_value_forward_when_history_runs_out():
    entries = [(U, date(2020, m, d), 2.0) for m in (1, 2, 3) for d in (1, 2, 3)]
    index = MarketIndex.fit(make_sales(entries), date(2020, 4, 1), date(2021, 6, 1), min_sales=3)
    value, source = index_row(index, U, date(2021, 6, 1))
    assert (value, source) == (pytest.approx(2.0), "carried")


def test_raises_when_there_is_no_earlier_sale_at_all():
    sales = make_sales([(U, date(2020, 4, 10), 1.0)])
    with pytest.raises(MarketIndexError, match="no earlier sales"):
        MarketIndex.fit(sales, date(2020, 4, 1), date(2020, 4, 1), min_sales=1)


def test_lookup_aligns_rows_and_rejects_months_outside_the_fit():
    entries = [(U, date(2020, m, 1), float(m)) for m in (1, 2, 3)]
    index = MarketIndex.fit(make_sales(entries), date(2020, 4, 1), date(2020, 5, 1), min_sales=1)
    rows = make_sales([(U, date(2020, 5, 20), 0.0), (U, date(2020, 4, 3), 0.0)])
    assert index.lookup(rows).to_list() == pytest.approx([2.5, 2.0])
    assert index.lookup(rows).name == "market_index"
    assert index.value_at(U, date(2020, 5, 17)) == pytest.approx(2.5)
    with pytest.raises(MarketIndexError):
        index.lookup(make_sales([(U, date(2020, 9, 1), 0.0)]))
