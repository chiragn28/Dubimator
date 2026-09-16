"""Ranking metrics over per-query result lists. Pure: no database, no MLflow."""

import math

import numpy as np
import polars as pl

NAN = float("nan")
PER_QUERY_SCHEMA = {
    "query_id": pl.Int64,
    "kind": pl.Utf8,
    "top_ids": pl.List(pl.Int64),
    "ndcg": pl.Float64,
    "rr": pl.Float64,
    "p5": pl.Float64,
    "mean_grade_top": pl.Float64,
    "answerable": pl.Boolean,
}


def _dcg(grades, k: int) -> float:
    return sum((2.0**grade - 1.0) / math.log2(index + 2) for index, grade in enumerate(grades[:k]))


def ndcg_at_k(ranked_grades, k: int = 10) -> float:
    ideal = _dcg(sorted(ranked_grades, reverse=True), k)
    return NAN if ideal == 0 else _dcg(list(ranked_grades), k) / ideal


def reciprocal_rank(ranked_grades, min_grade: int = 2) -> float:
    for index, grade in enumerate(ranked_grades):
        if grade >= min_grade:
            return 1.0 / (index + 1)
    return 0.0


def precision_at_k(ranked_grades, k: int = 5, grade: int = 3) -> float:
    return sum(value == grade for value in list(ranked_grades)[:k]) / k


def mean_top_grade(ranked_grades, k: int = 10) -> float:
    top = list(ranked_grades)[:k]
    return float(np.mean(top)) if top else NAN


def per_query_metrics(frame: pl.DataFrame, score: str, k: int = 10) -> pl.DataFrame:
    grouped = (
        frame.sort(["query_id", score, "fused_pos"], descending=[False, True, False])
        .group_by("query_id", maintain_order=True)
        .agg(pl.col("kind").first(), pl.col("grade"), pl.col("listing_id"))
    )
    rows = []
    for query_id, kind, grades, listing_ids in grouped.iter_rows():
        rows.append(
            {
                "query_id": query_id,
                "kind": kind,
                "top_ids": listing_ids[:k],
                "ndcg": ndcg_at_k(grades, k),
                "rr": reciprocal_rank(grades),
                "p5": precision_at_k(grades, 5),
                "mean_grade_top": mean_top_grade(grades, k),
                "answerable": kind != "no_match" and max(grades) >= 1,
            }
        )
    return pl.DataFrame(rows, schema=PER_QUERY_SCHEMA)


def _mean(series: pl.Series) -> float:
    return float(series.mean()) if series.len() else NAN


def summarize(per_query: pl.DataFrame) -> dict[str, float]:
    answerable = per_query.filter(pl.col("answerable"))
    no_match = per_query.filter(pl.col("kind") == "no_match")
    summary = {
        "ndcg_at_10": _mean(answerable["ndcg"]),
        "mrr": _mean(answerable["rr"]),
        "precision_at_5": _mean(answerable["p5"]),
        "queries": float(answerable.height),
        "no_match.mean_grade_top10": _mean(no_match["mean_grade_top"]),
        "no_match.queries": float(no_match.height),
    }
    for (kind,), group in answerable.group_by("kind", maintain_order=True):
        summary[f"kind.{kind}.ndcg_at_10"] = _mean(group["ndcg"])
    return summary


def bootstrap_ci(
    values: np.ndarray, n: int = 1_000, seed: int = 7, alpha: float = 0.05
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return NAN, NAN
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(n, values.size))].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)
