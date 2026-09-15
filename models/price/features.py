"""Feature engineering for the price model: segments, market index, location priors."""

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

from ingestion.normalize import map_unique, match_key

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


LEVEL_KEYS = {
    "city": ("property_type",),
    "area": ("property_type", "area_id"),
    "project": ("property_type", "area_id", "project_key"),
    "building": ("property_type", "area_id", "building_key"),
}
PRIOR_COLUMNS = (
    "prior_area", "prior_project", "prior_building",
    "n_area", "n_project", "n_building", "loc_level",
)  # fmt: skip


class LocationPriorError(RuntimeError):
    """A row's property type has no prior fit data."""


def add_location_keys(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        map_unique(frame["project_name"], match_key, pl.Utf8).alias("project_key"),
        map_unique(frame["building_name"], match_key, pl.Utf8).alias("building_key"),
    )


@dataclass(frozen=True)
class LocationPriors:
    """Bulk-weighted relative-price statistics per location level, shrunk down the chain."""

    stats: dict[str, pl.DataFrame]
    shrink_k: float
    min_level_n: float

    @classmethod
    def fit(cls, frame: pl.DataFrame, shrink_k: float, min_level_n: float) -> "LocationPriors":
        stats = {}
        for level, keys in LEVEL_KEYS.items():
            stats[level] = (
                frame.drop_nulls(list(keys))
                .group_by(list(keys))
                .agg(
                    (pl.col("y") * pl.col("bulk_weight")).sum().alias("sum_wy"),
                    pl.col("bulk_weight").sum().alias("sum_w"),
                )
                .sort(list(keys))
            )
        return cls(stats, shrink_k, min_level_n)

    def transform(self, frame: pl.DataFrame) -> pl.DataFrame:
        out = frame.with_row_index("__p_row")
        for level, keys in LEVEL_KEYS.items():
            table = self.stats[level].rename(
                {"sum_wy": f"__p_wy_{level}", "sum_w": f"__p_w_{level}"}
            )
            out = out.join(table, on=list(keys), how="left")
        out = out.sort("__p_row")
        if out["__p_w_city"].null_count():
            missing = sorted(out.filter(pl.col("__p_w_city").is_null())["property_type"].unique())
            raise LocationPriorError(f"no prior fit data for property type(s) {missing}")

        def wy(level: str) -> pl.Expr:
            return pl.col(f"__p_wy_{level}").fill_null(0.0)

        def w(level: str) -> pl.Expr:
            return pl.col(f"__p_w_{level}").fill_null(0.0)

        k = self.shrink_k
        out = out.with_columns((pl.col("__p_wy_city") / pl.col("__p_w_city")).alias("__p_city"))
        out = out.with_columns(
            ((wy("area") + k * pl.col("__p_city")) / (w("area") + k)).alias("prior_area")
        )
        out = out.with_columns(
            ((wy("project") + k * pl.col("prior_area")) / (w("project") + k)).alias("prior_project")
        )
        out = out.with_columns(
            ((wy("building") + k * pl.col("prior_project")) / (w("building") + k)).alias(
                "prior_building"
            )
        )
        n = self.min_level_n
        out = out.with_columns(
            w("area").log1p().alias("n_area"),
            w("project").log1p().alias("n_project"),
            w("building").log1p().alias("n_building"),
            pl.when(w("building") >= n)
            .then(3)
            .when(w("project") >= n)
            .then(2)
            .when(w("area") >= n)
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("loc_level"),
        )
        return out.drop([name for name in out.columns if name.startswith("__p_")])


def oof_priors(
    frame: pl.DataFrame, folds: np.ndarray, shrink_k: float, min_level_n: float
) -> pl.DataFrame:
    """Transform each fold with priors fitted on the other folds (no row sees its own price)."""
    indexed = frame.with_columns(pl.Series("__oof_fold", folds)).with_row_index("__oof_row")
    parts = []
    for fold in np.unique(folds):
        in_fold = pl.col("__oof_fold") == fold
        priors = LocationPriors.fit(indexed.filter(~in_fold), shrink_k, min_level_n)
        parts.append(priors.transform(indexed.filter(in_fold)))
    return pl.concat(parts).sort("__oof_row").drop("__oof_row", "__oof_fold")
