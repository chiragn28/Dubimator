import math

import polars as pl
import pytest

from listings.config import DetectConfig
from listings.evaluate import (
    evaluate_detection,
    evaluate_fraud,
    pair_metrics,
    save_pr_curve,
    stock_photo_control_pairs,
)

CONFIG = DetectConfig()


def test_pair_metrics_are_hand_checkable():
    labels = [True, True, True, False, False]
    decisions = [True, True, False, True, False]
    metrics = pair_metrics(labels, decisions)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert (metrics["true_positives"], metrics["false_positives"]) == (2.0, 1.0)
    assert (metrics["false_negatives"], metrics["pairs"]) == (1.0, 5.0)


def test_pair_metrics_handle_nothing_flagged():
    metrics = pair_metrics([True, False], [False, False])
    assert metrics["precision"] == 0.0 and metrics["recall"] == 0.0 and metrics["f1"] == 0.0


def test_stock_photo_controls_are_same_set_different_area_and_not_duplicates():
    attributes = pl.DataFrame(
        {"listing_id": [1, 2, 3, 4], "photo_set_id": [5, 5, 5, 6], "area_id": [10, 20, 10, 30]}
    )
    pairs = pl.DataFrame(
        {
            "listing_a": [1, 1, 1],
            "listing_b": [2, 3, 4],
            "is_duplicate": [False, False, False],
        }
    )
    mask = stock_photo_control_pairs(pairs, attributes).to_list()
    assert mask == [True, False, False]  # same set + different area; same area; different set


def test_evaluate_detection_reports_every_slice():
    pairs = pl.DataFrame(
        {
            "listing_a": [1, 1, 2, 3],
            "listing_b": [2, 3, 3, 4],
            "split": ["report"] * 4,
            "score": [0.9, 0.8, 0.2, 0.1],
            "decision": [True, True, False, False],
            "baseline_decision": [True, True, True, False],
            "is_duplicate": [True, False, False, False],
            "same_building_control": [False, True, True, False],
        }
    )
    attributes = pl.DataFrame(
        {
            "listing_id": [1, 2, 3, 4],
            "photo_set_id": [5, 5, 6, 6],
            "area_id": [10, 20, 30, 30],
        }
    )
    truth = pl.DataFrame(
        {"listing_a": [1, 5], "listing_b": [2, 6], "pattern": ["exact_repost", "reworded"]}
    )
    metrics = evaluate_detection(
        type("R", (), {"pairs": pairs, "threshold": 0.5, "stats": {"detect_seconds": 1.0}})(),
        truth,
        attributes,
        CONFIG,
    )
    assert metrics["report.precision"] == pytest.approx(0.5)  # 1 of 2 flagged pairs is a duplicate
    assert metrics["report.recall"] == pytest.approx(
        1.0
    )  # the one report-split duplicate is caught
    assert metrics["retrieval.recall"] == pytest.approx(0.5)  # pair (5,6) was never retrieved
    assert metrics["control.same_building.model_fp_rate"] == pytest.approx(0.5)
    assert metrics["control.same_building.baseline_fp_rate"] == pytest.approx(1.0)
    assert metrics["pattern.exact_repost.recall"] == pytest.approx(1.0)
    assert metrics["pattern.reworded.recall"] == pytest.approx(0.0)
    assert metrics["threshold"] == 0.5
    assert "report.pr_auc" in metrics


def test_evaluate_fraud_scores_bait_against_its_label():
    flags = pl.DataFrame(
        {
            "listing_id": [1, 2],
            "flag": ["bait_price", "bait_price"],
            "detail": ["{}", "{}"],
        }
    )
    truth = pl.DataFrame(
        {
            "listing_id": [1, 2, 3, 4],
            "dup_group_id": [None, None, None, None],
            "control_group_id": [None, None, None, None],
            "fraud_label": ["bait_price", None, "bait_price", None],
        },
        schema={
            "listing_id": pl.Int64, "dup_group_id": pl.Int64,
            "control_group_id": pl.Utf8, "fraud_label": pl.Utf8,
        },
    )  # fmt: skip
    metrics = evaluate_fraud(
        type("F", (), {"flags": flags, "stats": {"bait_price_skipped": 0.0}})(), truth, CONFIG
    )
    assert metrics["fraud.bait_price.precision"] == pytest.approx(0.5)
    assert metrics["fraud.bait_price.recall"] == pytest.approx(0.5)
    assert metrics["fraud.bait_price.flagged"] == 2.0


def test_pr_curve_is_written(tmp_path):
    path = save_pr_curve([0.9, 0.2, 0.7], [True, False, True], tmp_path / "pr.png")
    assert path.is_file() and path.stat().st_size > 0


def test_duplicate_truth_reads_the_planted_patterns(loaded_corpus):
    from listings.truth import load_duplicate_truth

    settings, corpus, _ = loaded_corpus
    conn = settings.connect()
    try:
        truth = load_duplicate_truth(conn)
    finally:
        conn.close()
    counts = dict(truth.group_by("pattern").len().iter_rows())
    assert counts == {
        "exact_repost": corpus.counts["exact_repost"],
        "reworded": corpus.counts["reworded"],
        "edited_photo": corpus.counts["edited_photo"],
    }
    assert (truth["listing_a"] < truth["listing_b"]).all()


def test_log_run_writes_to_the_temporary_store(temp_mlflow, tmp_path):
    import mlflow

    from listings.evaluate import log_run

    artifact = tmp_path / "pr.png"
    artifact.write_bytes(b"png")
    run_id = log_run(
        {"report.precision": 0.99, "report.recall": 0.8},
        {"threshold": 0.5, "corpus_run_id": 1},
        [artifact],
        CONFIG,
        tracking_uri=temp_mlflow["tracking_uri"],
        artifact_location=temp_mlflow["artifact_location"],
    )
    run = mlflow.get_run(run_id)
    assert run.data.metrics["report.precision"] == pytest.approx(0.99)
    assert run.data.params["threshold"] == "0.5"
    assert "pr.png" in {item.path for item in mlflow.MlflowClient().list_artifacts(run_id)}


# --- Additional coverage for the phase-4 rulings: per-split (not pooled) precision/recall,
# the price-shift recall gap, and every fraud/stats metric evaluate_fraud/evaluate_detection
# emit — a metric with no assertion behind it is a gap. ---


def _pairs_row(listing_a, listing_b, split, score, decision, is_duplicate, ratio=None):
    row = {
        "listing_a": listing_a,
        "listing_b": listing_b,
        "split": split,
        "score": score,
        "decision": decision,
        "baseline_decision": decision,
        "is_duplicate": is_duplicate,
        "same_building_control": False,
    }
    if ratio is not None:
        row["abs_log_price_ratio"] = ratio
    return row


def test_evaluate_detection_keeps_train_and_threshold_metrics_separate_from_report():
    # train: 4 pairs, 2 positive, 1 TP/1 FP/1 FN -> precision=recall=f1=0.5
    # threshold: 2 pairs, 1 positive, 1 TP/1 FP/0 FN -> precision=0.5, recall=1.0
    # report: 2 pairs, 1 positive, 1 TP/0 FP/0 FN -> precision=recall=f1=1.0
    rows = [
        _pairs_row(1, 2, "train", 0.9, True, True),
        _pairs_row(1, 3, "train", 0.9, True, False),
        _pairs_row(2, 3, "train", 0.1, False, True),
        _pairs_row(3, 4, "train", 0.1, False, False),
        _pairs_row(5, 6, "threshold", 0.9, True, True),
        _pairs_row(6, 7, "threshold", 0.9, True, False),
        _pairs_row(8, 9, "report", 0.9, True, True),
        _pairs_row(9, 10, "report", 0.1, False, False),
    ]
    pairs = pl.DataFrame(rows)
    attributes = pl.DataFrame(
        {
            "listing_id": list(range(1, 11)),
            "photo_set_id": list(range(1, 11)),
            "area_id": list(range(1, 11)),
        }
    )
    truth = pl.DataFrame(schema={"listing_a": pl.Int64, "listing_b": pl.Int64, "pattern": pl.Utf8})
    metrics = evaluate_detection(
        type("R", (), {"pairs": pairs, "threshold": 0.5, "stats": {"detect_seconds": 2.5}})(),
        truth,
        attributes,
        CONFIG,
    )

    assert metrics["train.pairs"] == pytest.approx(4.0)
    assert metrics["train.positives"] == pytest.approx(2.0)
    assert metrics["train.precision"] == pytest.approx(0.5)
    assert metrics["train.recall"] == pytest.approx(0.5)
    assert metrics["train.f1"] == pytest.approx(0.5)

    assert metrics["threshold.pairs"] == pytest.approx(2.0)
    assert metrics["threshold.positives"] == pytest.approx(1.0)
    assert metrics["threshold.precision"] == pytest.approx(0.5)
    assert metrics["threshold.recall"] == pytest.approx(1.0)

    assert metrics["report.pairs"] == pytest.approx(2.0)
    assert metrics["report.positives"] == pytest.approx(1.0)
    assert metrics["report.precision"] == pytest.approx(1.0)
    assert metrics["report.recall"] == pytest.approx(1.0)

    # a reader scanning only for "precision"/"recall" must not land on the in-sample train
    # numbers by mistake — the headline keys are the report-split ones.
    assert metrics["report.precision"] != metrics["train.precision"] or (
        metrics["report.recall"] != metrics["train.recall"]
    )
    assert metrics["stats.detect_seconds"] == pytest.approx(2.5)


def test_evaluate_detection_price_shift_recall_gap_isolates_shifted_clones():
    # Four report-split duplicates: two with a small price ratio (both caught), two with a
    # large one (only one caught) — the split the corpus's price-shifted reposts create.
    rows = [
        _pairs_row(1, 2, "report", 0.95, True, True, ratio=0.01),
        _pairs_row(3, 4, "report", 0.95, True, True, ratio=0.02),
        _pairs_row(5, 6, "report", 0.60, True, True, ratio=0.30),
        _pairs_row(7, 8, "report", 0.10, False, True, ratio=0.35),
    ]
    pairs = pl.DataFrame(rows)
    attributes = pl.DataFrame(
        {
            "listing_id": list(range(1, 9)),
            "photo_set_id": list(range(1, 9)),
            "area_id": list(range(1, 9)),
        }
    )
    truth = pl.DataFrame(
        {
            "listing_a": [1, 3, 5, 7],
            "listing_b": [2, 4, 6, 8],
            "pattern": ["exact_repost"] * 4,
        }
    )
    metrics = evaluate_detection(
        type("R", (), {"pairs": pairs, "threshold": 0.5, "stats": {}})(), truth, attributes, CONFIG
    )

    assert metrics["price_shift.below_median.pairs"] == pytest.approx(2.0)
    assert metrics["price_shift.below_median.recall"] == pytest.approx(1.0)
    assert metrics["price_shift.above_median.pairs"] == pytest.approx(2.0)
    assert metrics["price_shift.above_median.recall"] == pytest.approx(0.5)
    # the gap this metric exists to surface
    assert metrics["price_shift.above_median.recall"] < metrics["price_shift.below_median.recall"]


def test_evaluate_detection_pr_auc_is_nan_without_both_classes():
    pairs = pl.DataFrame(
        [
            _pairs_row(1, 2, "report", 0.9, True, False),
            _pairs_row(2, 3, "report", 0.1, False, False),
        ]
    )
    attributes = pl.DataFrame(
        {"listing_id": [1, 2, 3], "photo_set_id": [1, 2, 3], "area_id": [1, 2, 3]}
    )
    truth = pl.DataFrame(schema={"listing_a": pl.Int64, "listing_b": pl.Int64, "pattern": pl.Utf8})
    metrics = evaluate_detection(
        type("R", (), {"pairs": pairs, "threshold": 0.5, "stats": {}})(), truth, attributes, CONFIG
    )
    assert math.isnan(metrics["report.pr_auc"])


def test_evaluate_fraud_reports_photo_reuse_and_relist_counts_and_stats():
    flags = pl.DataFrame(
        {
            "listing_id": [1, 2, 3, 4, 5],
            "flag": [
                "bait_price",
                "bait_price",
                "photo_reuse",
                "inconsistent_relist",
                "inconsistent_relist",
            ],
            "detail": ["{}"] * 5,
        }
    )
    truth = pl.DataFrame(
        {
            "listing_id": [1, 2, 3, 4, 5],
            "dup_group_id": [None] * 5,
            "control_group_id": [None] * 5,
            "fraud_label": ["bait_price", None, None, None, None],
        },
        schema={
            "listing_id": pl.Int64, "dup_group_id": pl.Int64,
            "control_group_id": pl.Utf8, "fraud_label": pl.Utf8,
        },
    )  # fmt: skip
    metrics = evaluate_fraud(
        type("F", (), {"flags": flags, "stats": {"bait_price_skipped": 0.0, "listings": 5.0}})(),
        truth,
        CONFIG,
    )
    assert metrics["fraud.photo_reuse.flagged"] == pytest.approx(1.0)
    assert metrics["fraud.inconsistent_relist.flagged"] == pytest.approx(2.0)
    assert metrics["stats.bait_price_skipped"] == pytest.approx(0.0)
    assert metrics["stats.listings"] == pytest.approx(5.0)
