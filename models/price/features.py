"""Feature engineering for the price model: segments, market index, location priors."""

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

SUB_KINDS = {
    "Flat": "flat",
    "Hotel Apartment": "hotel_apartment",
    "Stacked Townhouses": "townhouse",
}
_BEDROOMS = r"^(\d) B/R$"


def derive_segments(frame: pl.DataFrame) -> pl.DataFrame:
    """Add size_basis, sub_kind, room_kind, bedrooms and segment (pure; no target use)."""
    villa = pl.col("property_type") == "villa"
    rooms = pl.col("rooms")
    bedroom_count = rooms.str.extract(_BEDROOMS, 1).cast(pl.Float64)
    size_basis = (
        pl.when(villa & pl.col("property_sub_type").is_null())
        .then(pl.lit("plot"))
        .otherwise(pl.lit("built_up"))
    )
    sub_kind = (
        pl.when(villa)
        .then(pl.lit("villa"))
        .otherwise(
            pl.col("property_sub_type").replace_strict(
                SUB_KINDS, default=None, return_dtype=pl.Utf8
            )
        )
    )
    room_kind = (
        pl.when(rooms == "Studio")
        .then(pl.lit("studio"))
        .when(bedroom_count.is_not_null())
        .then(pl.lit("bedrooms"))
        .when(rooms == "Penthouse")
        .then(pl.lit("penthouse"))
        .when(rooms == "Single Room")
        .then(pl.lit("single_room"))
        .otherwise(pl.lit("unknown"))
    )
    bedrooms = pl.when(rooms == "Studio").then(pl.lit(0.0)).otherwise(bedroom_count)
    return frame.with_columns(
        size_basis.alias("size_basis"),
        sub_kind.alias("sub_kind"),
        room_kind.alias("room_kind"),
        bedrooms.alias("bedrooms"),
    ).with_columns(
        pl.concat_str(
            [pl.col("property_type"), pl.col("reg_type"), pl.col("size_basis")], separator="_"
        ).alias("segment")
    )


INDEX_WINDOWS = (3, 6, 12)
INDEX_SCHEMA = {"segment": pl.Utf8, "month": pl.Date, "value": pl.Float64, "source": pl.Utf8}


class MarketIndexError(RuntimeError):
    """The market index cannot be computed or looked up for a segment/month."""


def month_number(day: date) -> int:
    return day.year * 12 + day.month - 1


def month_from_number(number: int) -> date:
    return date(number // 12, number % 12 + 1, 1)


def _window_median(
    months: np.ndarray, values: np.ndarray, target: int, width: int, min_sales: int
) -> float | None:
    """Median of values in months [target - width, target - 1]; None if too few sales."""
    low = np.searchsorted(months, target - width, side="left")
    high = np.searchsorted(months, target, side="left")
    if high - low < min_sales:
        return None
    return float(np.median(values[low:high]))


def _resolve(candidates, target: int, min_sales: int) -> tuple[float | None, str | None]:
    for label, months, values in candidates:
        for width in INDEX_WINDOWS:
            value = _window_median(months, values, target, width, min_sales)
            if value is not None:
                return value, f"{label}_{width}"
    return None, None


@dataclass(frozen=True)
class MarketIndex:
    """Median ln(price per m²) per segment over the months strictly before each month."""

    table: pl.DataFrame

    @classmethod
    def fit(
        cls, sales: pl.DataFrame, first_month: date, last_month: date, min_sales: int
    ) -> "MarketIndex":
        day = pl.col("instance_date")
        data = sales.select(
            "segment",
            pl.concat_str([pl.col("property_type"), pl.col("size_basis")], separator="_").alias(
                "pool"
            ),
            (day.dt.year().cast(pl.Int64) * 12 + day.dt.month().cast(pl.Int64) - 1).alias("m"),
            (pl.col("price_aed") / pl.col("area_sqm")).log().alias("v"),
        ).sort("m")

        def history(column: str, key: str) -> tuple[np.ndarray, np.ndarray]:
            subset = data.filter(pl.col(column) == key)
            return subset["m"].to_numpy(), subset["v"].to_numpy()

        records = []
        for segment, pool in data.select("segment", "pool").unique().sort("segment").iter_rows():
            candidates = (
                ("segment", *history("segment", segment)),
                ("pooled", *history("pool", pool)),
            )
            last = None
            for target in range(month_number(first_month), month_number(last_month) + 1):
                value, source = _resolve(candidates, target, min_sales)
                if value is None:
                    if last is None:
                        raise MarketIndexError(
                            f"no market index for {segment} in "
                            f"{month_from_number(target):%Y-%m}: no earlier sales"
                        )
                    value, source = last, "carried"
                last = value
                records.append((segment, month_from_number(target), value, source))
        return cls(pl.DataFrame(records, schema=INDEX_SCHEMA, orient="row"))

    def lookup(self, frame: pl.DataFrame) -> pl.Series:
        keyed = frame.select(
            pl.col("segment"), pl.col("instance_date").dt.truncate("1mo").alias("month")
        ).with_row_index("__row")
        joined = keyed.join(
            self.table.select("segment", "month", "value"), on=["segment", "month"], how="left"
        ).sort("__row")
        if joined["value"].null_count():
            raise MarketIndexError(
                "market index missing for some rows: segment or month outside the fitted range"
            )
        return joined["value"].alias("market_index")

    def value_at(self, segment: str, month: date) -> float:
        first = month.replace(day=1)
        match = self.table.filter((pl.col("segment") == segment) & (pl.col("month") == first))
        if match.height != 1:
            raise MarketIndexError(f"no market index for {segment} in {first:%Y-%m}")
        return float(match["value"][0])
