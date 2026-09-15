"""Regenerate tests/fixtures/dld_sample.csv from the real DLD file.

Run from the repo root: uv run python scripts/build_dld_fixture.py
"""

from pathlib import Path

import polars as pl

SEED = 42
SOURCE = Path("data/raw/Transactions.csv")
TARGET = Path("tests/fixtures/dld_sample.csv")
MARKET = ["Sell", "Sell - Pre registration", "Delayed Sell", "Sale On Payment Plan"]


def main() -> None:
    # Keep "null" as literal text so the fixture round-trips in the source format.
    df = pl.read_csv(SOURCE, infer_schema=False)
    group = pl.col("trans_group_en")
    procedure = pl.col("procedure_name_en")
    worth = pl.col("actual_worth").cast(pl.Float64, strict=False)
    area = pl.col("procedure_area").cast(pl.Float64, strict=False)
    market = (group == "Sales") & procedure.is_in(MARKET)
    priced = market & (pl.col("instance_date") != "") & (worth >= 10_000) & (area > 0)

    picks = []

    def take(condition: pl.Expr, n: int) -> None:
        subset = df.filter(condition)
        picks.append(subset.sample(n=min(n, subset.height), seed=SEED))

    take(group == "Mortgages", 25)
    take(group == "Gifts", 20)
    take((group == "Sales") & ~procedure.is_in(MARKET), 25)
    take(pl.col("instance_date") == "", 5)
    take(market & (pl.col("actual_worth") == "null"), 10)
    take(market & (worth < 10_000), 10)
    for name in MARKET:
        take(priced & (procedure == name), 15)
    for property_type in ["Unit", "Villa", "Land", "Building"]:
        take(priced & (pl.col("property_type_en") == property_type), 40)

    sample = pl.concat(picks).unique(subset="transaction_id", keep="first", maintain_order=True)
    if not 250 <= sample.height <= 400:
        raise SystemExit(f"fixture has {sample.height} rows, expected 250-400")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    sample.write_csv(TARGET)
    print(f"wrote {sample.height} rows to {TARGET}")


if __name__ == "__main__":
    main()
