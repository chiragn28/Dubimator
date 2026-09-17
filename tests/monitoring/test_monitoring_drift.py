"""Pure drift statistics: PSI (numeric and categorical), KS, feature_drift and log stats."""

import json
import math

import numpy as np
import polars as pl
import pytest

from monitoring import drift
from monitoring.drift import feature_drift, ks_stat, psi, psi_categorical, search_log_stats


def test_psi_identical_distributions_is_zero():
    values = np.arange(1000, dtype=float)
    assert psi(values, values) == pytest.approx(0.0, abs=1e-12)


def test_psi_hand_computed_two_bins():
    # Reference 0..99 with 2 bins splits at the median 49.5 -> 50% / 50%.
    # Current has 75% below and 25% above:
    # (0.75-0.5)*ln(0.75/0.5) + (0.25-0.5)*ln(0.25/0.5) = 0.25*ln1.5 + 0.25*ln2
    reference = np.arange(100, dtype=float)
    current = np.array([10.0] * 75 + [90.0] * 25)
    expected = 0.25 * math.log(1.5) + 0.25 * math.log(2.0)
    assert psi(reference, current, bins=2) == pytest.approx(expected, rel=1e-9)


def test_psi_shifted_distribution_is_flagged_size():
    rng = np.random.default_rng(0)
    reference = rng.normal(0.0, 1.0, 5000)
    current = rng.normal(1.0, 1.0, 5000)
    assert psi(reference, current) > 0.2


def test_psi_empty_bin_is_smoothed_not_infinite():
    reference = np.arange(100, dtype=float)
    current = np.full(50, 5.0)  # every current value falls in the first bin
    value = psi(reference, current, bins=4)
    assert math.isfinite(value) and value > 1.0


def test_psi_ignores_non_finite_and_handles_empty():
    assert psi([1.0, 2.0, np.nan, np.inf], [1.0, 2.0]) == pytest.approx(0.0, abs=1e-12)
    assert math.isnan(psi([], [1.0]))
    assert math.isnan(psi([1.0], [np.nan]))


def test_psi_with_heavy_ties_uses_unique_edges():
    reference = np.array([1.0] * 50 + [2.0] * 50)
    assert psi(reference, reference, bins=10) == pytest.approx(0.0, abs=1e-12)
    shifted = np.array([1.0] * 90 + [2.0] * 10)
    expected = (0.9 - 0.5) * math.log(0.9 / 0.5) + (0.1 - 0.5) * math.log(0.1 / 0.5)
    assert psi(reference, shifted, bins=10) == pytest.approx(expected, rel=1e-9)


def test_psi_constant_reference_detects_a_moved_value():
    assert psi([5] * 100, [100] * 100) > 0.2
    assert psi([5] * 100, [1] * 100) > 0.2


def test_psi_constant_reference_uses_below_equal_above_bins():
    assert psi([5.0] * 100, [5.0] * 40) == pytest.approx(0.0, abs=1e-12)
    # 50% below, 50% equal vs the reference's 100% equal (below/above floored at EPSILON).
    current = [1.0] * 50 + [5.0] * 50
    eps = drift.EPSILON
    expected = (0.5 - eps) * math.log(0.5 / eps) + (0.5 - 1.0) * math.log(0.5 / 1.0)
    assert psi([5.0] * 100, current) == pytest.approx(expected, rel=1e-9)


def test_psi_categorical_hand_computed():
    reference = ["a"] * 50 + ["b"] * 50
    current = ["a"] * 80 + ["b"] * 20
    expected = 0.3 * math.log(0.8 / 0.5) + (-0.3) * math.log(0.2 / 0.5)
    assert psi_categorical(reference, current) == pytest.approx(expected, rel=1e-9)
    assert psi_categorical(reference, reference) == pytest.approx(0.0, abs=1e-12)


def test_psi_categorical_new_category_is_smoothed():
    value = psi_categorical(["a"] * 10, ["a"] * 5 + ["c"] * 5)
    assert math.isfinite(value) and value > 0.2


def test_psi_categorical_top_groups_the_rest_as_other():
    reference = ["a"] * 6 + ["b"] * 3 + ["c"] * 1
    current = ["a"] * 6 + ["b"] * 3 + ["d"] * 1  # c and d both land in "other"
    assert psi_categorical(reference, current, top=2) == pytest.approx(0.0, abs=1e-12)
    assert psi_categorical(reference, current) > 0.0


def test_ks_stat_hand_computed():
    assert ks_stat([1, 2, 3, 4], [3, 4, 5, 6]) == pytest.approx(0.5)
    assert ks_stat([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)
    assert ks_stat([1, 2], [10, 20]) == pytest.approx(1.0)
    assert math.isnan(ks_stat([], [1.0]))


def test_feature_drift_columns_and_flags():
    rng = np.random.default_rng(1)
    n = 2000
    reference = pl.DataFrame(
        {
            "size": rng.normal(100, 10, n),
            "beds": rng.integers(0, 4, n).astype(float),
            "kind": ["x"] * (n // 2) + ["y"] * (n // 2),
        }
    )
    current = pl.DataFrame(
        {
            "size": rng.normal(130, 10, n),  # shifted
            "beds": rng.integers(0, 4, n).astype(float),  # same
            "kind": ["x"] * (n // 2) + ["y"] * (n // 2),  # same
        }
    )
    table = feature_drift(reference, current, numeric=["size", "beds"], categorical=["kind"])
    assert table.columns == ["feature", "kind", "psi", "ks", "flagged"]
    by_feature = {row["feature"]: row for row in table.to_dicts()}
    assert by_feature["size"]["flagged"] is True
    assert by_feature["size"]["kind"] == "numeric"
    assert by_feature["size"]["ks"] > 0.5
    assert by_feature["beds"]["flagged"] is False
    assert by_feature["kind"]["kind"] == "categorical"
    assert by_feature["kind"]["ks"] is None
    assert by_feature["kind"]["flagged"] is False


def test_feature_drift_flags_a_constant_feature_that_moved():
    reference = pl.DataFrame({"beds": [2.0] * 100})
    current = pl.DataFrame({"beds": [3.0] * 100})
    row = feature_drift(reference, current, ["beds"], []).to_dicts()[0]
    assert row["flagged"] is True


def test_feature_drift_flags_on_ks_alone():
    assert drift.KS_THRESHOLD == 0.3
    reference = pl.DataFrame({"x": np.arange(1000, dtype=float)})
    shifted = pl.DataFrame({"x": np.arange(1000, dtype=float) + 400})  # KS = 0.4
    nudged = pl.DataFrame({"x": np.arange(1000, dtype=float) + 200})  # KS = 0.2
    # A PSI threshold nothing can reach isolates the KS rule.
    flagged = feature_drift(reference, shifted, ["x"], [], threshold=1e9).to_dicts()[0]
    assert flagged["ks"] == pytest.approx(0.4)
    assert flagged["flagged"] is True
    calm = feature_drift(reference, nudged, ["x"], [], threshold=1e9).to_dicts()[0]
    assert calm["ks"] == pytest.approx(0.2)
    assert calm["flagged"] is False


def test_feature_drift_top_categories_applies_per_feature():
    reference = pl.DataFrame({"area": [1] * 5 + [2] * 3 + [3] * 2})
    current = pl.DataFrame({"area": [1] * 5 + [2] * 3 + [4] * 2})
    table = feature_drift(reference, current, [], ["area"], top_categories={"area": 2})
    assert table["psi"][0] == pytest.approx(0.0, abs=1e-12)


def _line(**fields) -> str:
    return json.dumps(fields)


def test_search_log_stats_counts_requests_queries_and_zero_results():
    lines = [
        _line(logger="api.access", message="request", path="/v1/search", status=200),
        _line(logger="api.access", message="request", path="/v1/search", status=200),
        _line(logger="api.access", message="request", path="/v1/search", status=500),
        _line(logger="api.access", message="request", path="/v1/price", status=200),
        _line(logger="api.search", message="search", q="two bed marina"),
        _line(logger="api.search", message="search", q="villa", results=0),
        _line(logger="api.search", message="search", q="studio jlt", results=4),
        "not json at all",
        "",
    ]
    stats = search_log_stats(lines)
    assert stats["search_requests"] == 3
    assert stats["search_errors"] == 1
    assert stats["search_error_rate"] == pytest.approx(1 / 3)
    assert stats["queries_logged"] == 3
    assert stats["query_words_mean"] == pytest.approx((3 + 1 + 2) / 3)
    assert stats["query_chars_mean"] == pytest.approx((14 + 5 + 10) / 3)
    assert stats["zero_result_rate"] == pytest.approx(0.5)
    assert stats["skipped_lines"] == 1


def test_search_log_stats_without_result_counts_reports_none():
    stats = search_log_stats([_line(logger="api.search", message="search", q="a b")])
    assert stats["search_requests"] == 0
    assert stats["search_error_rate"] is None
    assert stats["zero_result_rate"] is None
    assert stats["query_words_mean"] == pytest.approx(2.0)


def test_search_log_stats_empty():
    stats = search_log_stats([])
    assert stats["queries_logged"] == 0
    assert stats["query_words_mean"] is None
