import csv
from datetime import date

import polars as pl
import pytest

BASE_RAW_ROW = {
    "transaction_id": "1-11-2020-100",
    "procedure_id": "11",
    "trans_group_id": "1",
    "trans_group_ar": "مبايعات",
    "trans_group_en": "Sales",
    "procedure_name_ar": "بيع",
    "procedure_name_en": "Sell",
    "instance_date": "15-06-2020",
    "property_type_id": "3",
    "property_type_ar": "وحدة",
    "property_type_en": "Unit",
    "property_sub_type_id": "60",
    "property_sub_type_ar": "شقة",
    "property_sub_type_en": "Flat",
    "property_usage_ar": "سكني",
    "property_usage_en": "Residential",
    "reg_type_id": "1",
    "reg_type_ar": "العقارات القائمة",
    "reg_type_en": "Existing Properties",
    "area_id": "364",
    "area_name_ar": "مرسى دبي",
    "area_name_en": "Marsa Dubai",
    "building_name_ar": "",
    "building_name_en": "MARINA TOWER",
    "project_number": "1234",
    "project_name_ar": "",
    "project_name_en": "Marina Tower",
    "master_project_en": "Dubai Marina",
    "master_project_ar": "",
    "nearest_landmark_ar": "",
    "nearest_landmark_en": "Burj Al Arab",
    "nearest_metro_ar": "",
    "nearest_metro_en": "DAMAC Properties",
    "nearest_mall_ar": "",
    "nearest_mall_en": "Marina Mall",
    "rooms_ar": "",
    "rooms_en": "1 B/R",
    "has_parking": "1",
    "procedure_area": "80.5",
    "actual_worth": "1200000",
    "meter_sale_price": "14906.83",
    "rent_value": "null",
    "meter_rent_price": "null",
    "no_of_parties_role_1": "1",
    "no_of_parties_role_2": "1",
    "no_of_parties_role_3": "0",
}


@pytest.fixture
def write_dld_csv(tmp_path):
    def write(rows, columns=None, name="dld.csv"):
        columns = columns or list(BASE_RAW_ROW)
        path = tmp_path / name
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(columns)
            for overrides in rows:
                record = {**BASE_RAW_ROW, **overrides}
                writer.writerow([record.get(column, "") for column in columns])
        return path

    return write


@pytest.fixture
def make_raw():
    def build(rows):
        records = []
        for overrides in rows:
            record = {**BASE_RAW_ROW, **overrides}
            records.append({k: (None if v == "null" else v) for k, v in record.items()})
        return pl.DataFrame(records, schema={column: pl.Utf8 for column in BASE_RAW_ROW})

    return build


TYPED_DEFAULTS = {
    "trans_group": "sales",
    "procedure_name": "Sell",
    "instance_date": date(2020, 6, 1),
    "property_type": "unit",
    "property_sub_type": "Flat",
    "property_usage": "Residential",
    "reg_type": "ready",
    "area_id": 1,
    "area_name": "Area One",
    "area_name_ar": None,
    "building_name": None,
    "project_number": None,
    "project_name": None,
    "master_project": None,
    "nearest_landmark": None,
    "nearest_metro": None,
    "nearest_mall": None,
    "rooms": "1 B/R",
    "bedrooms": 1,
    "has_parking": True,
    "area_sqm": 100.0,
    "price_aed": 1_000_000.0,
    "parties_role_1": 1,
    "parties_role_2": 1,
    "parties_role_3": 0,
}


@pytest.fixture
def make_typed():
    from ingestion.normalize import TYPED_SCHEMA

    def build(rows):
        records = []
        for n, overrides in enumerate(rows, start=1):
            record = {"transaction_id": f"T-{n}", "source_row": n, **TYPED_DEFAULTS, **overrides}
            if "year" not in overrides:
                record["year"] = record["instance_date"].year if record["instance_date"] else None
            if "price_per_sqm_aed" not in overrides:
                price, area = record["price_aed"], record["area_sqm"]
                record["price_per_sqm_aed"] = (
                    price / area if price is not None and area and area > 0 else None
                )
            records.append({column: record[column] for column in TYPED_SCHEMA})
        return pl.DataFrame(records, schema=TYPED_SCHEMA)

    return build
