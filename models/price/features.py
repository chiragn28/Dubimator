"""Feature engineering for the price model: segments, market index, location priors."""

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
