import polars as pl
import pytest

from ingestion.schema import (
    EXPECTED_COLUMNS,
    SchemaDriftError,
    SourceFileError,
    read_raw,
    validate_columns,
)


def test_expected_columns_has_46_names():
    assert len(EXPECTED_COLUMNS) == 46


def test_reads_all_columns_as_strings(write_dld_csv):
    df = read_raw(write_dld_csv([{}, {"transaction_id": "x-2", "actual_worth": "null"}]))
    assert df.shape == (2, 46)
    assert all(dtype == pl.Utf8 for dtype in df.dtypes)
    assert df["actual_worth"].to_list() == ["1200000", None]
    assert df["area_name_ar"][0] == "مرسى دبي"


def test_column_order_does_not_matter(write_dld_csv):
    df = read_raw(write_dld_csv([{}], columns=sorted(EXPECTED_COLUMNS)))
    assert set(df.columns) == EXPECTED_COLUMNS


def test_missing_column_raises(write_dld_csv):
    columns = [c for c in sorted(EXPECTED_COLUMNS) if c != "actual_worth"]
    with pytest.raises(SchemaDriftError, match="actual_worth"):
        read_raw(write_dld_csv([{}], columns=columns))


def test_extra_column_raises(write_dld_csv):
    with pytest.raises(SchemaDriftError, match="amount"):
        read_raw(write_dld_csv([{}], columns=[*sorted(EXPECTED_COLUMNS), "amount"]))


def test_renamed_column_reports_both_names():
    renamed = [c if c != "procedure_area" else "procedure_area_sqft" for c in EXPECTED_COLUMNS]
    with pytest.raises(SchemaDriftError) as info:
        validate_columns(renamed)
    assert "'procedure_area'" in str(info.value)
    assert "'procedure_area_sqft'" in str(info.value)


@pytest.mark.parametrize(
    "column, value",
    [("trans_group_en", "Leases"), ("reg_type_en", "Planned"), ("property_type_en", "Parking")],
)
def test_unexpected_domain_value_raises(write_dld_csv, column, value):
    with pytest.raises(SchemaDriftError, match=value):
        read_raw(write_dld_csv([{}, {column: value}]))


def test_empty_domain_value_raises(write_dld_csv):
    with pytest.raises(SchemaDriftError, match="trans_group_en"):
        read_raw(write_dld_csv([{"trans_group_en": ""}]))


def test_invalid_utf8_raises(write_dld_csv):
    path = write_dld_csv([{}])
    path.write_bytes(path.read_bytes().replace(b"Marina Mall", b"Marina \xff\xfe Mall"))
    with pytest.raises(SourceFileError):
        read_raw(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.csv"):
        read_raw(tmp_path / "nope.csv")
