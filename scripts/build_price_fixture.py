"""Regenerate tests/fixtures/price_sample.csv: ~3,000 real DLD home sales for the Phase 3
integration test, stratified by split period and property kind.

Run from the repo root: uv run python scripts/build_price_fixture.py
"""

from datetime import date
from pathlib import Path

import polars as pl

SEED = 42
SOURCE = Path("data/raw/Transactions.csv")
TARGET = Path("tests/fixtures/price_sample.csv")
MARKET = ["Sell", "Sell - Pre registration", "Delayed Sell", "Sale On Payment Plan"]
PERIODS = {  # name: (start inclusive, end exclusive, rows)
    "seed": (date(2014, 1, 1), date(2015, 1, 1), 300),
    "train": (date(2015, 1, 1), date(2022, 7, 1), 1_500),
    "val": (date(2022, 7, 1), date(2022, 11, 1), 500),
    "test": (date(2022, 11, 1), date(2023, 4, 1), 700),
}


def main() -> None:
    # Keep "null" as literal text so the fixture round-trips in the source format.
    df = pl.read_csv(SOURCE, infer_schema=False)
    day = pl.col("instance_date").str.strptime(pl.Date, "%d-%m-%Y", strict=False)
    worth = pl.col("actual_worth").cast(pl.Float64, strict=False)
    size = pl.col("procedure_area").cast(pl.Float64, strict=False)
    ptype, sub = pl.col("property_type_en"), pl.col("property_sub_type_en").fill_null("")
    market = (
        (pl.col("trans_group_en") == "Sales")
        & pl.col("procedure_name_en").is_in(MARKET)
        & (worth >= 10_000)
        & (size > 0)
    )
    kinds = {  # name: (filter, share of each period)
        "flat": ((ptype == "Unit") & (sub == "Flat"), 0.50),
        "hotel_apartment": ((ptype == "Unit") & (sub == "Hotel Apartment"), 0.10),
        "townhouse": ((ptype == "Unit") & (sub == "Stacked Townhouses"), 0.05),
        "villa_built_up": ((ptype == "Villa") & (sub == "Villa"), 0.20),
        "villa_plot": ((ptype == "Villa") & (sub == ""), 0.15),
    }
    picks = []
    for start, end, rows in PERIODS.values():
        in_period = market & (day >= start) & (day < end)
        for condition, share in kinds.values():
            subset = df.filter(in_period & condition)
            picks.append(subset.sample(n=min(round(rows * share), subset.height), seed=SEED))

    sample = pl.concat(picks).unique(subset="transaction_id", keep="first", maintain_order=True)
    if not 2_500 <= sample.height <= 3_100:
        raise SystemExit(f"fixture has {sample.height} rows, expected 2,500-3,100")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    sample.write_csv(TARGET)
    print(f"wrote {sample.height} rows to {TARGET}")


if __name__ == "__main__":
    main()
