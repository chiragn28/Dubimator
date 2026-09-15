import math

import numpy as np
import polars as pl
import pytest

from models.price.evaluate import (
    conformal_quantile,
    coverage,
    fit_conformal,
    price_metrics,
    quantiles_for,
    sliced_metrics,
)


def test_price_metrics_match_hand_computed_values():
    metrics = price_metrics([100.0, 100.0, 100.0, 100.0], [110.0, 90.0, 130.0, 100.0])
    assert metrics["mdape"] == pytest.approx(0.10)
    assert metrics["ppe10"] == pytest.approx(0.75)
    assert metrics["ppe20"] == pytest.approx(0.75)
    expected_rmse = math.sqrt(
        (math.log(1.1) ** 2 + math.log(0.9) ** 2 + math.log(1.3) ** 2 + 0.0) / 4
    )
    assert metrics["rmse_log"] == pytest.approx(expected_rmse)
    assert metrics["n"] == 4.0


def test_sliced_metrics_cover_segments_levels_and_dedup():
    frame = pl.DataFrame(
        {
            "actual": [100.0, 100.0, 200.0, 200.0],
            "predicted": [110.0, 110.0, 200.0, 260.0],
            "segment": ["a", "a", "b", "b"],
            "loc_level": [3, 3, 1, 0],
            "bulk_group": [7, 7, 8, 9],
        }
    )
    metrics = sliced_metrics(frame, "test_clean")
    assert metrics["test_clean.all.n"] == 4.0
    assert metrics["test_clean.segment.a.mdape"] == pytest.approx(0.10)
    assert metrics["test_clean.segment.b.ppe20"] == pytest.approx(0.5)
    assert metrics["test_clean.loc_level.3.n"] == 2.0
    assert metrics["test_clean.loc_level.0.mdape"] == pytest.approx(0.30)
    assert metrics["test_clean.dedup.n"] == 3.0


def test_conformal_quantile_uses_the_finite_sample_rank():
    errors = [i / 10 for i in range(1, 11)]  # 0.1 .. 1.0
    assert conformal_quantile(errors, 0.20) == pytest.approx(0.9)  # rank ceil(11*0.8) = 9
    assert conformal_quantile(errors, 0.05) == math.inf  # rank 11 > n
    assert conformal_quantile(list(range(1, 20)), 0.05) == 19  # rank ceil(20*0.95) = 19


def test_small_segments_use_the_pooled_quantiles():
    segments = ["big"] * 30 + ["small"] * 5
    errors = [0.1] * 30 + [0.9] * 5
    conformal = fit_conformal(segments, errors, min_rows=20)
    assert conformal["big"] == {"q80": pytest.approx(0.1), "q95": pytest.approx(0.1)}
    assert conformal["small"] == conformal["_pooled"]
    assert quantiles_for(conformal, "never_seen") == conformal["_pooled"]


def test_fit_conformal_refuses_too_few_rows():
    with pytest.raises(ValueError, match="too few validation rows"):
        fit_conformal(["a"] * 5, [0.1] * 5, min_rows=1)


def test_coverage_is_inclusive_at_the_edges():
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    assert coverage(actual, np.array([1.0] * 4), np.array([3.0] * 4)) == pytest.approx(0.75)
