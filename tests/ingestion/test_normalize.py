from datetime import date

import polars as pl
import pytest

from ingestion.normalize import (
    TYPED_SCHEMA,
    clean_display_name,
    map_unique,
    match_key,
    parse_date,
    parse_float,
    parse_int,
    parse_rooms,
    to_typed,
)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("15-06-2020", date(2020, 6, 15)),
        (" 01-01-1995 ", date(1995, 1, 1)),
        ("31-02-2020", None),
        ("2020-06-15", None),
        ("", None),
        (None, None),
        ("garbage", None),
    ],
)
def test_parse_date(value, expected):
    assert parse_date(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("1200000", 1200000.0),
        ("80.5", 80.5),
        (" 12 ", 12.0),
        ("", None),
        (None, None),
        ("abc", None),
        ("nan", None),
        ("inf", None),
    ],
)
def test_parse_float(value, expected):
    assert parse_float(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [("364", 364), ("7.0", 7), ("7.5", None), ("", None), (None, None), ("x", None)],
)
def test_parse_int(value, expected):
    assert parse_int(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("Studio", ("Studio", 0)),
        ("1 B/R", ("1 B/R", 1)),
        ("9 B/R", ("9 B/R", 9)),
        ("2 b/r", ("2 B/R", 2)),
        ("PENTHOUSE", ("Penthouse", None)),
        ("GYM", ("Gym", None)),
        ("Office", ("Office", None)),
        ("Shop", ("Shop", None)),
        ("Single Room", ("Single Room", None)),
        ("Store", ("Store", None)),
        ("", (None, None)),
        (None, (None, None)),
        ("  Loft  ", ("Loft", None)),
    ],
)
def test_parse_rooms(value, expected):
    assert parse_rooms(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("Al Khairan  Second", "Al Khairan Second"),
        ("MADINAT HIND 2", "Madinat Hind 2"),
        ("DIFC", "DIFC"),
        ("JBR", "JBR"),
        ("Al-Nahdah", "Al-Nahdah"),
        ("  Marsa Dubai ", "Marsa Dubai"),
        ("مرسى دبي", "مرسى دبي"),
        ("", None),
        ("   ", None),
        (None, None),
    ],
)
def test_clean_display_name(value, expected):
    assert clean_display_name(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("Al-Nahdah", "nahdah"),
        ("Jumeriah Beach Residence  - JBR", "jumeriah beach residence jbr"),
        ("Al Barsha South Fourth", "barsha south fourth"),
        ("Me'Aisem First", "me aisem first"),
        ("مرسى دبي", None),
        ("", None),
        (None, None),
    ],
)
def test_match_key(value, expected):
    assert match_key(value) == expected


def test_map_unique_applies_function_and_keeps_nulls():
    series = pl.Series("s", ["2", None, "2", "x"])
    assert map_unique(series, parse_int, pl.Int64).to_list() == [2, None, 2, None]


def test_map_unique_on_all_null_series():
    series = pl.Series("s", [None, None], dtype=pl.Utf8)
    assert map_unique(series, parse_int, pl.Int64).to_list() == [None, None]


def test_to_typed_schema_and_values(make_raw):
    raw = make_raw(
        [
            {"procedure_name_en": "  Sell ", "area_name_en": "Al Khairan  Second"},
            {
                "trans_group_en": "Mortgages",
                "reg_type_en": "Off-Plan Properties",
                "property_type_en": "Villa",
                "instance_date": "",
                "actual_worth": "null",
                "has_parking": "0",
                "rooms_en": "Studio",
                "master_project_en": "",
            },
        ]
    )
    typed = to_typed(raw)
    assert typed.schema == pl.Schema(TYPED_SCHEMA)
    first, second = typed.to_dicts()
    assert first["source_row"] == 1 and second["source_row"] == 2
    assert first["procedure_name"] == "Sell"
    assert first["trans_group"] == "sales" and second["trans_group"] == "mortgages"
    assert first["reg_type"] == "ready" and second["reg_type"] == "off_plan"
    assert second["property_type"] == "villa"
    assert first["instance_date"] == date(2020, 6, 15) and first["year"] == 2020
    assert second["instance_date"] is None and second["year"] is None
    assert first["area_name"] == "Al Khairan Second"
    assert first["building_name"] == "Marina Tower"
    assert first["area_id"] == 364 and first["project_number"] == 1234
    assert first["rooms"] == "1 B/R" and first["bedrooms"] == 1
    assert second["rooms"] == "Studio" and second["bedrooms"] == 0
    assert first["has_parking"] is True and second["has_parking"] is False
    assert first["area_sqm"] == 80.5 and first["price_aed"] == 1_200_000.0
    assert first["price_per_sqm_aed"] == pytest.approx(1_200_000 / 80.5)
    assert second["price_aed"] is None and second["price_per_sqm_aed"] is None
    assert second["master_project"] is None
    assert first["parties_role_1"] == 1 and first["parties_role_3"] == 0


def test_to_typed_zero_area_gives_null_price_per_sqm(make_raw):
    typed = to_typed(make_raw([{"procedure_area": "0"}]))
    assert typed["area_sqm"][0] == 0.0
    assert typed["price_per_sqm_aed"][0] is None
