import dataclasses
import json
import math

import mlflow
import numpy as np
import polars as pl
import pytest
from search_fixtures import ALIASES, AREAS, BAIT_IDS, BUILDINGS

from search.config import FEATURES, SLOTS, SearchConfig
from search.evaluate import (
    CONTENDERS,
    Evaluation,
    baseline_scores,
    closest_note_rates,
    evaluate_contenders,
    evaluate_report,
    load_fraud_listing_ids,
    load_true_slots,
    log_results,
    parser_accuracy,
    retrieval_recall,
    search_run,
    slot_checks,
    top_k_effects,
    write_artifacts,
)
from search.features import feature_table
from search.lexicon import Lexicon, load_lexicon
from search.parse import parse
from search.queries import build_query_set
from search.store import replace_query_set
from search.train import fit_ablation, train_rankers

CONFIG = dataclasses.replace(
    SearchConfig(),
    device="cpu",
    max_rounds=30,
    early_stopping_rounds=5,
    n_bootstrap=200,
    ef_search=100,
)
LEXICON = Lexicon.from_rows(
    areas=AREAS,
    aliases=ALIASES,
    buildings=[(name, area) for area, names in BUILDINGS.items() for name in names if name],
    projects=[(f"Project {area}", area) for area, _ in AREAS],
)


def _frame() -> pl.DataFrame:
    base = {name: 0.0 for name in FEATURES}
    rows = [
        # query 1: two listings, the fresher one is worse
        {**base, "query_id": 1, "kind": "specified", "listing_id": 1, "grade": 3, "fused_pos": 2,
         "days_since_posted": 30.0, "semantic_pos": float("nan"), "cluster_size": 3.0},
        {**base, "query_id": 1, "kind": "specified", "listing_id": 11, "grade": 0, "fused_pos": 1,
         "days_since_posted": 1.0, "semantic_pos": 1.0, "cluster_size": 1.0},
        # query 2: no_match
        {**base, "query_id": 2, "kind": "no_match", "listing_id": 12, "grade": 1, "fused_pos": 1,
         "days_since_posted": 5.0, "semantic_pos": 1.0, "cluster_size": 1.0},
    ]  # fmt: skip
    return pl.DataFrame(rows)


class FixedRanker:
    def __init__(self, scores):
        self._scores = np.asarray(scores, dtype=float)

    def score(self, frame):
        return self._scores


def test_baselines_order_by_freshness_semantic_rank_and_fused_rank():
    scores = baseline_scores(_frame())
    assert list(scores) == [
        "baseline_newest", "baseline_semantic", "baseline_fused", "baseline_rules",
    ]  # fmt: skip
    assert scores["baseline_newest"][1] > scores["baseline_newest"][0]
    assert scores["baseline_semantic"][0] < scores["baseline_semantic"][1]  # missing goes last
    assert list(scores["baseline_fused"]) == [-2.0, -1.0, -1.0]


def _rules_frame() -> pl.DataFrame:
    """One query asking for 2 bedrooms: an exact match, a near miss and a wrong area."""
    base = {name: float("nan") for name in FEATURES}
    common = {**base, "query_id": 1, "kind": "specified", "beds_stated": 1.0,
              "amenity_hits": 0.0, "amenity_asked": 0.0, "area_match": 1.0}  # fmt: skip
    rows = [
        {**common, "listing_id": 1, "grade": 1, "fused_pos": 1, "rrf_score": 0.030,
         "beds_diff": 0.0, "area_match": 0.0},
        {**common, "listing_id": 2, "grade": 2, "fused_pos": 2, "rrf_score": 0.020,
         "beds_diff": 1.0},
        {**common, "listing_id": 3, "grade": 3, "fused_pos": 3, "rrf_score": 0.010,
         "beds_diff": 0.0},
    ]  # fmt: skip
    return pl.DataFrame(rows)


def test_the_rules_baseline_orders_by_parsed_slot_matches_then_fusion():
    frame = _rules_frame()
    rules = baseline_scores(frame)["baseline_rules"]
    np.testing.assert_allclose(rules, [1.030, 2.020, 3.010])
    metrics, _ = evaluate_contenders(frame, {"baseline_rules": rules}, CONFIG)
    assert metrics["baseline_rules.ndcg_at_10"] == pytest.approx(1.0)


def test_closest_note_rates_split_no_match_from_the_rest():
    frame = pl.concat(
        [
            _rules_frame(),
            _rules_frame().with_columns(
                pl.lit(2, dtype=pl.Int64).alias("query_id"),
                pl.lit("no_match").alias("kind"),
                pl.lit(2.0).alias("beds_diff"),
            ),
        ]
    )
    rates = closest_note_rates(frame, np.arange(6, dtype=float), k=10)
    assert rates == {"note.closest.rate.no_match": 1.0, "note.closest.rate.other": 0.0}
    top_one_wrong = closest_note_rates(_rules_frame(), np.array([3.0, 2.0, 1.0]), k=2)
    assert top_one_wrong["note.closest.rate.other"] == 1.0  # the exact match is third


def test_evaluate_contenders_names_every_metric():
    frame = _frame()
    scores = {"xgboost": np.array([1.0, 0.0, 0.0]), **baseline_scores(frame)}
    metrics, per_query = evaluate_contenders(frame, scores, CONFIG)
    assert metrics["xgboost.ndcg_at_10"] == pytest.approx(1.0)
    assert metrics["baseline_fused.ndcg_at_10"] == pytest.approx((2**0 - 1 + 7 / math.log2(3)) / 7)
    assert metrics["baseline_newest.mrr"] == pytest.approx(0.5)
    assert metrics["xgboost.ci_low"] == metrics["xgboost.ci_high"] == pytest.approx(1.0)
    assert metrics["kind.specified.xgboost.ndcg_at_10"] == pytest.approx(1.0)
    assert metrics["kind.no_match.xgboost.mean_grade_top10"] == pytest.approx(1.0)
    assert metrics["kind.no_match.xgboost.queries"] == 1.0
    assert metrics["xgboost.queries"] == 1.0
    assert set(per_query) == set(scores)


def test_top_k_effects_count_hidden_duplicates_and_fraud_slots():
    frame = _frame()
    effects = top_k_effects(frame, np.array([1.0, 0.0, 0.0]), fraud_ids={12}, k=10)
    assert effects["dup.top10_removed"] == pytest.approx((2 + 0) / 2)
    assert effects["fraud.top10_share"] == pytest.approx(1 / 3)
    prefixed = top_k_effects(frame, np.zeros(3), fraud_ids=set(), k=1, prefix="ablation.x.")
    assert prefixed["ablation.x.fraud.top10_share"] == 0.0


def _truth(**overrides):
    truth = {
        "area_id": 1, "area_name": "Dubai Marina", "building": None, "bedrooms": 2,
        "property_type": "flat", "budget_min": None, "budget_max": 1_500_000.0,
        "min_size_sqm": None, "amenities": ["balcony"],
    }  # fmt: skip
    return {**truth, **overrides}


def test_slot_checks():
    parsed = parse("2BR apartment in Dubai Marina under 1.5M with balcony", LEXICON)
    checks = slot_checks(parsed, _truth())
    assert tuple(checks) == SLOTS
    assert all(ok for ok, _, _ in checks.values())
    wrong = slot_checks(parsed, _truth(bedrooms=3, budget_max=1_600_000.0, amenities=[]))
    assert [slot for slot, (ok, _, _) in wrong.items() if not ok] == [
        "bedrooms", "budget_max", "amenities",
    ]  # fmt: skip
    near = slot_checks(parsed, _truth(budget_max=1_510_000.0))
    assert near["budget_max"][0]


def test_parser_accuracy_and_error_table():
    queries = pl.DataFrame(
        {"query_id": [1, 2], "text": ["2BR flat in Dubai Marina under 1.5M with balcony",
                                      "3BR flat in Dubai Marina under 1.5M with balcony"]}
    )  # fmt: skip
    accuracy, errors = parser_accuracy(queries, {1: _truth(), 2: _truth()}, LEXICON)
    assert accuracy["parse.bedrooms.accuracy"] == 0.5
    assert accuracy["parse.area.accuracy"] == 1.0
    assert set(accuracy) == {f"parse.{slot}.accuracy" for slot in SLOTS}
    assert errors.rows() == [(2, queries["text"][1], "bedrooms", "2", "3")]


def test_retrieval_recall():
    queries = pl.DataFrame({"query_id": [1, 2, 3], "n_grade3": [2, 0, 4]})
    rows = pl.DataFrame({"query_id": [1, 1, 3, 3], "grade": [3, 1, 3, 0]})
    assert retrieval_recall(queries, rows) == pytest.approx((1 / 2 + 1 / 4) / 2)
    assert math.isnan(retrieval_recall(queries.filter(pl.col("n_grade3") == 0), rows))


def test_the_database_evaluation_end_to_end(search_db, fake_embedder, tmp_path):
    settings, _, _ = search_db
    config = dataclasses.replace(CONFIG, n_queries=150)
    conn = settings.connect()
    try:
        lexicon = load_lexicon(conn)
        queries, judgments = build_query_set(conn, fake_embedder, lexicon, config)
        replace_query_set(conn, queries, judgments)
        conn.commit()
        assert load_fraud_listing_ids(conn) == set(BAIT_IDS)
        assert set(load_true_slots(conn)) == set(
            queries.filter(pl.col("split") == "report")["query_id"]
        )
        table = feature_table(conn, lexicon)
        result = train_rankers(table, config, n_trials=1)
        ablation = fit_ablation(table, result, config)
        evaluation = evaluate_report(
            conn, table, result.rankers, lexicon, config, champion=result.winner, ablation=ablation
        )
    finally:
        conn.close()
    metrics = evaluation.metrics
    for contender in CONTENDERS:
        assert 0.0 <= metrics[f"{contender}.ndcg_at_10"] <= 1.0
    assert 0.0 <= metrics["retrieval.recall_at_200"] <= 1.0
    assert metrics["parse.bedrooms.accuracy"] >= 0.95
    assert {"dup.top10_removed", "fraud.top10_share", "ablation.no_trust.ndcg_at_10"} <= set(
        metrics
    )
    for contender in [*CONTENDERS, "ablation.no_trust"]:
        assert 0.0 <= metrics[f"{contender}.fraud.top10_share"] <= 1.0
        assert metrics[f"{contender}.dup.top10_removed"] >= 0.0
    assert metrics["fraud.top10_share"] == metrics[f"{result.winner}.fraud.top10_share"]
    assert 0.0 < metrics["candidates.fraud_share"] < 1.0
    assert 0.0 <= metrics["note.closest.rate.no_match"] <= 1.0
    assert metrics["note.closest.rate.no_match"] >= metrics["note.closest.rate.other"]
    assert metrics["report.queries"] == queries.filter(pl.col("split") == "report").height

    importance = result.rankers[result.winner].importance()
    paths = write_artifacts(tmp_path, evaluation, importance, {"queries": {"seconds": 1.5}})
    assert sorted(path.name for path in paths) == [
        "feature_importance.csv", "ndcg_comparison.png", "parse_errors.csv",
        "per_query_report.csv", "timings.json",
    ]  # fmt: skip
    report = pl.read_csv(tmp_path / "per_query_report.csv")
    assert {"query_id", "text", "kind", "parsed", *(f"ndcg.{c}" for c in CONTENDERS)} <= set(
        report.columns
    )
    assert json.loads((tmp_path / "timings.json").read_text()) == {"queries": {"seconds": 1.5}}
    assert isinstance(evaluation, Evaluation)


def test_search_run_logs_finite_metrics_and_artifacts(temp_mlflow, tmp_path):
    artifact = tmp_path / "note.txt"
    artifact.write_text("hello", encoding="utf-8")
    config = dataclasses.replace(CONFIG, experiment="search-ranking-test")
    with search_run(
        config, "unit", temp_mlflow["tracking_uri"], temp_mlflow["artifact_location"]
    ) as run:
        log_results({"a": 1.0, "b": float("nan")}, {"seed": 7}, [artifact])
    stored = mlflow.get_run(run.info.run_id)
    assert stored.data.metrics == {"a": 1.0}
    assert stored.data.params == {"seed": "7"}
    names = [item.path for item in mlflow.MlflowClient().list_artifacts(run.info.run_id)]
    assert names == ["note.txt"]
