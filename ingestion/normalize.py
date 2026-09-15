import math
import re
from datetime import date, datetime

import polars as pl

TRANS_GROUPS = {"Sales": "sales", "Mortgages": "mortgages", "Gifts": "gifts"}
REG_TYPES = {"Existing Properties": "ready", "Off-Plan Properties": "off_plan"}
PARKING = {"1": True, "0": False}

_BEDROOMS = re.compile(r"^([1-9]) B/R$", re.IGNORECASE)
_ROOM_LABELS = {
    "STUDIO": "Studio",
    "PENTHOUSE": "Penthouse",
    "GYM": "Gym",
    "OFFICE": "Office",
    "SHOP": "Shop",
    "SINGLE ROOM": "Single Room",
    "STORE": "Store",
}
_NON_ALNUM = re.compile(r"[^0-9a-z]+")

TYPED_SCHEMA = {
    "transaction_id": pl.Utf8,
    "source_row": pl.Int64,
    "trans_group": pl.Utf8,
    "procedure_name": pl.Utf8,
    "instance_date": pl.Date,
    "year": pl.Int16,
    "property_type": pl.Utf8,
    "property_sub_type": pl.Utf8,
    "property_usage": pl.Utf8,
    "reg_type": pl.Utf8,
    "area_id": pl.Int64,
    "area_name": pl.Utf8,
    "area_name_ar": pl.Utf8,
    "building_name": pl.Utf8,
    "project_number": pl.Int64,
    "project_name": pl.Utf8,
    "master_project": pl.Utf8,
    "nearest_landmark": pl.Utf8,
    "nearest_metro": pl.Utf8,
    "nearest_mall": pl.Utf8,
    "rooms": pl.Utf8,
    "bedrooms": pl.Int16,
    "has_parking": pl.Boolean,
    "area_sqm": pl.Float64,
    "price_aed": pl.Float64,
    "price_per_sqm_aed": pl.Float64,
    "parties_role_1": pl.Int16,
    "parties_role_2": pl.Int16,
    "parties_role_3": pl.Int16,
}


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), "%d-%m-%Y").date()  # noqa: DTZ007
    except ValueError:
        return None


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def parse_int(value: str | None) -> int | None:
    number = parse_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def parse_rooms(value: str | None) -> tuple[str | None, int | None]:
    text = " ".join(value.split()) if value else ""
    if not text:
        return (None, None)
    bedrooms = _BEDROOMS.match(text)
    if bedrooms:
        count = int(bedrooms.group(1))
        return (f"{count} B/R", count)
    label = _ROOM_LABELS.get(text.upper())
    if label == "Studio":
        return (label, 0)
    return (label or text, None)


def clean_display_name(value: str | None) -> str | None:
    text = " ".join(value.split()) if value else ""
    if not text:
        return None
    if len(text) > 5 and text.upper() == text and any(ch.isalpha() for ch in text):
        return text.title()
    return text


def match_key(value: str | None) -> str | None:
    if value is None:
        return None
    tokens = _NON_ALNUM.sub(" ", value.lower()).split()
    return " ".join(token for token in tokens if token != "al") or None


def _strip(value: str | None) -> str | None:
    return value.strip() or None if value is not None else None


def map_unique(series: pl.Series, fn, dtype) -> pl.Series:
    uniques = [value for value in series.unique().to_list() if value is not None]
    if not uniques:
        return pl.Series(series.name, [None] * series.len(), dtype=dtype)
    return series.replace_strict(
        uniques, [fn(value) for value in uniques], default=None, return_dtype=dtype
    )


def to_typed(raw: pl.DataFrame) -> pl.DataFrame:
    def text(column: str) -> pl.Series:
        return map_unique(raw[column], clean_display_name, pl.Utf8)

    typed = pl.DataFrame(
        {
            "transaction_id": map_unique(raw["transaction_id"], _strip, pl.Utf8),
            "source_row": pl.Series(range(1, raw.height + 1), dtype=pl.Int64),
            "trans_group": raw["trans_group_en"].replace_strict(TRANS_GROUPS, return_dtype=pl.Utf8),
            "procedure_name": map_unique(raw["procedure_name_en"], _strip, pl.Utf8),
            "instance_date": map_unique(raw["instance_date"], parse_date, pl.Date),
            "property_type": raw["property_type_en"].str.to_lowercase(),
            "property_sub_type": text("property_sub_type_en"),
            "property_usage": text("property_usage_en"),
            "reg_type": raw["reg_type_en"].replace_strict(REG_TYPES, return_dtype=pl.Utf8),
            "area_id": map_unique(raw["area_id"], parse_int, pl.Int64),
            "area_name": text("area_name_en"),
            "area_name_ar": text("area_name_ar"),
            "building_name": text("building_name_en"),
            "project_number": map_unique(raw["project_number"], parse_int, pl.Int64),
            "project_name": text("project_name_en"),
            "master_project": text("master_project_en"),
            "nearest_landmark": text("nearest_landmark_en"),
            "nearest_metro": text("nearest_metro_en"),
            "nearest_mall": text("nearest_mall_en"),
            "rooms": map_unique(raw["rooms_en"], lambda v: parse_rooms(v)[0], pl.Utf8),
            "bedrooms": map_unique(raw["rooms_en"], lambda v: parse_rooms(v)[1], pl.Int16),
            "has_parking": map_unique(raw["has_parking"], PARKING.get, pl.Boolean),
            "area_sqm": map_unique(raw["procedure_area"], parse_float, pl.Float64),
            "price_aed": map_unique(raw["actual_worth"], parse_float, pl.Float64),
            "parties_role_1": map_unique(raw["no_of_parties_role_1"], parse_int, pl.Int16),
            "parties_role_2": map_unique(raw["no_of_parties_role_2"], parse_int, pl.Int16),
            "parties_role_3": map_unique(raw["no_of_parties_role_3"], parse_int, pl.Int16),
        }
    )
    typed = typed.with_columns(
        pl.col("instance_date").dt.year().cast(pl.Int16).alias("year"),
        pl.when((pl.col("area_sqm") > 0) & pl.col("price_aed").is_not_null())
        .then(pl.col("price_aed") / pl.col("area_sqm"))
        .alias("price_per_sqm_aed"),
    )
    return typed.select(list(TYPED_SCHEMA)).cast(TYPED_SCHEMA)
