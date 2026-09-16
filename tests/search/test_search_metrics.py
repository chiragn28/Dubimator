import math

import numpy as np
import polars as pl
import pytest

from search.metrics import (
    bootstrap_ci,
    mean_top_grade,
    ndcg_at_k,
    per_query_metrics,
    precision_at_k,
    reciprocal_rank,
    summarize,
)


def test_ndcg_by_hand():
    assert ndcg_at_k([3, 2, 0]) == pytest.approx(1.0)
    dcg = 3 / math.log2(3) + 7 / 2
    ideal = 7 + 3 / math.log2(3)
    assert ndcg_at_k([0, 2, 3]) == pytest.approx(dcg / ideal)
    assert ndcg_at_k([0] * 10 + [3], k=10) == 0.0
    assert math.isnan(ndcg_at_k([0, 0, 0]))
    assert math.isnan(ndcg_at_k([]))


def test_reciprocal_rank_precision_and_mean_grade():
    assert reciprocal_rank([0, 1, 2]) == pytest.approx(1 / 3)
    assert reciprocal_rank([0, 1]) == 0.0
    assert reciprocal_rank([1, 0], min_grade=1) == 1.0
    assert precision_at_k([3, 3, 0]) == pytest.approx(0.4)
    assert precision_at_k([3, 2, 3, 3, 3, 3], k=5) == pytest.approx(0.8)
    assert mean_top_grade([3, 1]) == 2.0
    assert mean_top_grade([1] * 12 + [3], k=10) == 1.0
    assert math.isnan(mean_top_grade([]))
    assert math.isnan(precision_at_k([3, 3, 0], k=0))


FRAME = pl.DataFrame(
    {
        "query_id": [1, 1, 1, 2, 2, 3, 3, 4],
        "kind": ["specified"] * 3 + ["vague"] * 2 + ["no_match"] * 2 + ["specified"],
        "listing_id": [10, 11, 12, 20, 21, 30, 31, 40],
        "grade": [0, 3, 2, 2, 2, 1, 0, 0],
        "fused_pos": [1, 2, 3, 1, 2, 1, 2, 1],
        "score": [0.1, 0.9, 0.9, 0.5, 0.5, 0.2, 0.3, 0.0],
    }
)


def test_per_query_metrics_rank_by_score_then_fused_position():
    out = per_query_metrics(FRAME, "score", k=10).sort("query_id")
    rows = {row["query_id"]: row for row in out.to_dicts()}
    assert rows[1]["top_ids"] == [11, 12, 10]  # tie at 0.9 broken by fused_pos
    assert rows[1]["ndcg"] == pytest.approx(1.0) and rows[1]["rr"] == 1.0
    assert rows[1]["p5"] == pytest.approx(0.2)
    assert rows[2]["top_ids"] == [20, 21] and rows[2]["kind"] == "vague"
    assert rows[3]["top_ids"] == [31, 30] and rows[3]["mean_grade_top"] == 0.5
    assert [rows[q]["answerable"] for q in (1, 2, 3, 4)] == [True, True, False, False]
    assert math.isnan(rows[4]["ndcg"])


def test_per_query_metrics_nan_scores_rank_last():
    frame = pl.DataFrame(
        {
            "query_id": [1, 1, 1],
            "kind": ["specified"] * 3,
            "listing_id": [100, 101, 102],
            "grade": [3, 0, 0],
            "fused_pos": [1, 2, 3],
            "score": [float("nan"), 0.5, 0.9],
        }
    )
    out = per_query_metrics(frame, "score", k=10)
    assert out.to_dicts()[0]["top_ids"] == [102, 101, 100]


def test_summarize_uses_answerable_queries_only():
    summary = summarize(per_query_metrics(FRAME, "score", k=10))
    assert summary["queries"] == 2.0
    assert summary["ndcg_at_10"] == pytest.approx(1.0)
    assert summary["mrr"] == pytest.approx(1.0)
    assert summary["precision_at_5"] == pytest.approx(0.1)
    assert summary["no_match.queries"] == 1.0
    assert summary["no_match.mean_grade_top10"] == pytest.approx(0.5)
    assert summary["kind.specified.ndcg_at_10"] == pytest.approx(1.0)
    assert summary["kind.vague.ndcg_at_10"] == pytest.approx(1.0)
    assert "kind.no_match.ndcg_at_10" not in summary


def test_summarize_on_nothing_answerable_is_nan_not_an_error():
    only_no_match = FRAME.filter(pl.col("kind") == "no_match")
    summary = summarize(per_query_metrics(only_no_match, "score"))
    assert summary["queries"] == 0.0 and math.isnan(summary["ndcg_at_10"])


def test_bootstrap_ci():
    values = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    low, high = bootstrap_ci(values, n=500, seed=3)
    assert low <= values.mean() <= high
    assert (low, high) == bootstrap_ci(values, n=500, seed=3)
    assert bootstrap_ci(np.array([0.5, 0.5]), n=50) == (0.5, 0.5)
    assert all(math.isnan(bound) for bound in bootstrap_ci(np.array([])))
