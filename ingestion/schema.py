from collections.abc import Iterable
from pathlib import Path

import polars as pl

EXPECTED_COLUMNS = frozenset(
    {
        "transaction_id",
        "procedure_id",
        "trans_group_id",
        "trans_group_ar",
        "trans_group_en",
        "procedure_name_ar",
        "procedure_name_en",
        "instance_date",
        "property_type_id",
        "property_type_ar",
        "property_type_en",
        "property_sub_type_id",
        "property_sub_type_ar",
        "property_sub_type_en",
        "property_usage_ar",
        "property_usage_en",
        "reg_type_id",
        "reg_type_ar",
        "reg_type_en",
        "area_id",
        "area_name_ar",
        "area_name_en",
        "building_name_ar",
        "building_name_en",
        "project_number",
        "project_name_ar",
        "project_name_en",
        "master_project_en",
        "master_project_ar",
        "nearest_landmark_ar",
        "nearest_landmark_en",
        "nearest_metro_ar",
        "nearest_metro_en",
        "nearest_mall_ar",
        "nearest_mall_en",
        "rooms_ar",
        "rooms_en",
        "has_parking",
        "procedure_area",
        "actual_worth",
        "meter_sale_price",
        "rent_value",
        "meter_rent_price",
        "no_of_parties_role_1",
        "no_of_parties_role_2",
        "no_of_parties_role_3",
    }
)

CLOSED_DOMAINS = {
    "trans_group_en": frozenset({"Sales", "Mortgages", "Gifts"}),
    "reg_type_en": frozenset({"Existing Properties", "Off-Plan Properties"}),
    "property_type_en": frozenset({"Unit", "Villa", "Land", "Building"}),
}


class SourceFileError(ValueError):
    pass


class SchemaDriftError(SourceFileError):
    pass


def validate_columns(columns: Iterable[str]) -> None:
    actual = set(columns)
    missing = sorted(EXPECTED_COLUMNS - actual)
    unexpected = sorted(actual - EXPECTED_COLUMNS)
    if missing or unexpected:
        raise SchemaDriftError(
            f"DLD CSV columns changed. Missing: {missing}. Unexpected: {unexpected}."
        )


def validate_domains(df: pl.DataFrame) -> None:
    for column, allowed in CLOSED_DOMAINS.items():
        unexpected = set(df[column].unique().to_list()) - allowed
        if unexpected:
            raise SchemaDriftError(
                f"Unexpected values in {column}: {sorted(map(repr, unexpected))}"
            )


def read_raw(path: Path) -> pl.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"DLD CSV not found: {path}")
    try:
        df = pl.read_csv(path, infer_schema=False, null_values=["null"], encoding="utf8")
    except pl.exceptions.PolarsError as exc:
        raise SourceFileError(f"Could not read {path}: {exc}") from exc
    validate_columns(df.columns)
    validate_domains(df)
    return df
