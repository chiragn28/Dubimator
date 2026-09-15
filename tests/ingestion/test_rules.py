from datetime import date

import pytest

from ingestion.rules import REASONS, NoPeerGroupError, classify


def normal_group(n=40, **overrides):
    """n clean sales around AED 10,000/m² (prices spread 900k-1.1M on 100 m²)."""
    return [{"price_aed": 900_000.0 + i * (200_000.0 / (n - 1)), **overrides} for i in range(n)]


def reasons_by_id(result):
    return dict(zip(result["transaction_id"].to_list(), result["exclusion_reason"].to_list()))


def test_reason_codes_are_in_rule_order():
    assert REASONS == (
        "duplicate_transaction_id",
        "mortgage",
        "gift",
        "non_market_procedure",
        "missing_date",
        "missing_price",
        "invalid_area",
        "price_below_floor",
        "suspected_sqft_entry",
        "price_outlier_low",
        "price_outlier_high",
    )


def test_deterministic_reasons(make_typed):
    rows = [
        {
            "transaction_id": "mort",
            "trans_group": "mortgages",
            "procedure_name": "Mortgage Registration",
        },
        {"transaction_id": "gift", "trans_group": "gifts", "procedure_name": "Grant"},
        {"transaction_id": "lto", "procedure_name": "Lease to Own Registration"},
        {"transaction_id": "noproc", "procedure_name": None},
        {"transaction_id": "nodate", "instance_date": None},
        {"transaction_id": "noprice", "price_aed": None},
        {"transaction_id": "zeroarea", "area_sqm": 0.0},
        {"transaction_id": "negarea", "area_sqm": -5.0},
        {"transaction_id": "noarea", "area_sqm": None},
        {"transaction_id": "cheap", "price_aed": 9_999.0},
    ]
    result = reasons_by_id(classify(make_typed(rows)))
    assert result == {
        "mort": "mortgage",
        "gift": "gift",
        "lto": "non_market_procedure",
        "noproc": "non_market_procedure",
        "nodate": "missing_date",
        "noprice": "missing_price",
        "zeroarea": "invalid_area",
        "negarea": "invalid_area",
        "noarea": "invalid_area",
        "cheap": "price_below_floor",
    }


def test_first_matching_rule_wins(make_typed):
    rows = [
        {
            "transaction_id": "a",
            "trans_group": "mortgages",
            "price_aed": None,
            "instance_date": None,
        },
        {"transaction_id": "b", "procedure_name": "Grant", "price_aed": 5.0},
        {"transaction_id": "c", "instance_date": None, "price_aed": None},
    ]
    assert reasons_by_id(classify(make_typed(rows))) == {
        "a": "mortgage",
        "b": "non_market_procedure",
        "c": "missing_date",
    }


def test_duplicate_transaction_id_keeps_first(make_typed):
    rows = [
        {"transaction_id": "dup", "trans_group": "gifts", "procedure_name": "Grant"},
        {"transaction_id": "dup", "trans_group": "gifts", "procedure_name": "Grant"},
    ]
    result = classify(make_typed(rows))
    assert result["exclusion_reason"].to_list() == ["gift", "duplicate_transaction_id"]


def test_market_procedures_are_clean(make_typed):
    procedures = ["Sell", "Sell - Pre registration", "Delayed Sell", "Sale On Payment Plan"]
    rows = normal_group(40)
    for i, procedure in enumerate(procedures):
        rows[i]["procedure_name"] = procedure
    result = classify(make_typed(rows))
    assert result["exclusion_reason"].null_count() == 40
    assert result["peer_tier"].to_list() == [1] * 40


def test_outliers_and_sqft_entry(make_typed):
    rows = normal_group(40) + [
        {"transaction_id": "sqft", "price_aed": 100_000.0},
        {"transaction_id": "low", "price_aed": 300_000.0},
        {"transaction_id": "high", "price_aed": 5_000_000.0},
    ]
    result = classify(make_typed(rows))
    reasons = reasons_by_id(result)
    assert reasons["sqft"] == "suspected_sqft_entry"
    assert reasons["low"] == "price_outlier_low"
    assert reasons["high"] == "price_outlier_high"
    assert sum(r is None for r in reasons.values()) == 40
    z = dict(zip(result["transaction_id"].to_list(), result["price_robust_z"].to_list()))
    assert z["high"] > 3.5 and z["low"] < -3.5 and abs(z["T-1"]) <= 3.5


def test_small_group_falls_back_to_wider_tier(make_typed):
    rows = normal_group(40) + normal_group(5, area_id=2)
    result = classify(make_typed(rows))
    tiers = result["peer_tier"].to_list()
    assert tiers[:40] == [1] * 40
    assert tiers[40:] == [3] * 5


def test_zero_spread_group_falls_back(make_typed):
    identical = [{"price_aed": 1_000_000.0, "instance_date": date(2021, 3, 1)} for _ in range(35)]
    rows = normal_group(40) + identical
    tiers = classify(make_typed(rows))["peer_tier"].to_list()
    assert tiers[40:] == [5] * 35


def test_no_qualifying_tier_raises(make_typed):
    with pytest.raises(NoPeerGroupError):
        classify(make_typed([{"price_aed": 1_000_000.0} for _ in range(35)]))


def test_excluded_rows_have_no_peer_stats(make_typed):
    result = classify(
        make_typed(normal_group(40) + [{"trans_group": "gifts", "procedure_name": "Grant"}])
    )
    last = result.row(-1, named=True)
    assert last["exclusion_reason"] == "gift"
    assert last["peer_tier"] is None and last["price_robust_z"] is None


def test_tight_group_does_not_flag_modest_deviations(make_typed):
    tight = [{"price_aed": 999_000.0 + i * 50.0} for i in range(40)]
    rows = tight + [
        {"transaction_id": "x1_3", "price_aed": 1_300_000.0},
        {"transaction_id": "x2", "price_aed": 2_000_000.0},
    ]
    reasons = reasons_by_id(classify(make_typed(rows)))
    assert reasons["x1_3"] is None
    assert reasons["x2"] == "price_outlier_high"


def test_output_columns_and_order(make_typed):
    from ingestion.normalize import TYPED_SCHEMA

    rows = [{**row, "source_row": 40 - i} for i, row in enumerate(normal_group(40))]
    result = classify(make_typed(rows))
    assert result.columns == [*TYPED_SCHEMA, "exclusion_reason", "peer_tier", "price_robust_z"]
    assert result["source_row"].to_list() == list(range(1, 41))
