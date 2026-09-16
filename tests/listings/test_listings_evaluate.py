import math
from datetime import date, timedelta

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
    # posted_at is dense enough (6 listings, config's default train_share=0.60 /
    # threshold_share=0.20) that the report-split cutoff (index 3 of 6 sorted dates) falls
    # between listing 4 and listing 5: pair (1, 2)'s later date (listing 2, day 5) and pair
    # (5, 6)'s later date (listing 6, day 6) both land in the report split by date — matching
    # what pattern.*.recall's denominator is now computed from, independent of the `pairs`
    # frame's own (hardcoded, retrieval-derived) split column below.
    attributes = pl.DataFrame(
        {
            "listing_id": [1, 2, 3, 4, 5, 6],
            "photo_set_id": [5, 5, 6, 6, 7, 7],
            "area_id": [10, 20, 30, 30, 40, 40],
            "posted_at": [
                date(2023, 1, 3),
                date(2023, 1, 5),
                date(2023, 1, 1),
                date(2023, 1, 2),
                date(2023, 1, 4),
                date(2023, 1, 6),
            ],
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
    # both planted pairs are report-split by date, so each pattern's denominator is 1.
    assert metrics["pattern.exact_repost.pairs"] == pytest.approx(1.0)
    assert metrics["pattern.reworded.pairs"] == pytest.approx(1.0)
    assert metrics["pattern.exact_repost.recall"] == pytest.approx(1.0)  # (1,2) was flagged
    assert metrics["pattern.reworded.recall"] == pytest.approx(0.0)  # (5,6) was never retrieved
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


def _pairs_row(
    listing_a,
    listing_b,
    split,
    score,
    decision,
    is_duplicate,
    ratio=None,
    baseline=None,
    same_building=False,
):
    row = {
        "listing_a": listing_a,
        "listing_b": listing_b,
        "split": split,
        "score": score,
        "decision": decision,
        "baseline_decision": decision if baseline is None else baseline,
        "is_duplicate": is_duplicate,
        "same_building_control": same_building,
    }
    if ratio is not None:
        row["abs_log_price_ratio"] = ratio
    return row


def test_evaluate_detection_keeps_train_and_tune_metrics_separate_from_report():
    # train: 4 pairs, 2 positive, 1 TP/1 FP/1 FN -> precision=recall=f1=0.5
    # tune (the data's split value is still "threshold"): 2 pairs, 1 positive,
    #   1 TP/1 FP/0 FN -> precision=0.5, recall=1.0
    # report: 3 pairs, 2 positive; the model misses one duplicate (13,14) that the cruder
    #   baseline rule catches -> model precision=1.0/recall=0.5, baseline precision=1.0/recall=1.0.
    #   baseline_decision genuinely differs from decision on (13,14), so this is not baseline
    #   silently aliasing the model's own decision.
    rows = [
        _pairs_row(1, 2, "train", 0.9, True, True),
        _pairs_row(1, 3, "train", 0.9, True, False),
        _pairs_row(2, 3, "train", 0.1, False, True),
        _pairs_row(3, 4, "train", 0.1, False, False),
        _pairs_row(5, 6, "threshold", 0.9, True, True),
        _pairs_row(6, 7, "threshold", 0.9, True, False),
        _pairs_row(8, 9, "report", 0.9, True, True),
        _pairs_row(9, 10, "report", 0.1, False, False),
        _pairs_row(13, 14, "report", 0.2, False, True, baseline=True),
    ]
    pairs = pl.DataFrame(rows)
    attributes = pl.DataFrame(
        {
            "listing_id": list(range(1, 11)) + [13, 14],
            "photo_set_id": list(range(1, 11)) + [13, 14],
            "area_id": list(range(1, 11)) + [13, 14],
        }
    )
    truth = pl.DataFrame(schema={"listing_a": pl.Int64, "listing_b": pl.Int64, "pattern": pl.Utf8})
    metrics = evaluate_detection(
        type(
            "R",
            (),
            {
                "pairs": pairs,
                "threshold": 0.5,
                "stats": {
                    "detect_seconds": 2.5,
                    "train_pairs": 4.0,
                    "train_positives": 2.0,
                    "threshold_pairs": 2.0,
                    "threshold_positives": 1.0,
                },
            },
        )(),
        truth,
        attributes,
        CONFIG,
    )

    assert metrics["train.pairs"] == pytest.approx(4.0)
    assert metrics["train.positives"] == pytest.approx(2.0)
    assert metrics["train.precision"] == pytest.approx(0.5)
    assert metrics["train.recall"] == pytest.approx(0.5)
    assert metrics["train.f1"] == pytest.approx(0.5)

    assert metrics["tune.pairs"] == pytest.approx(2.0)
    assert metrics["tune.positives"] == pytest.approx(1.0)
    assert metrics["tune.precision"] == pytest.approx(0.5)
    assert metrics["tune.recall"] == pytest.approx(1.0)

    assert metrics["report.pairs"] == pytest.approx(3.0)
    assert metrics["report.positives"] == pytest.approx(2.0)
    assert metrics["report.precision"] == pytest.approx(1.0)
    assert metrics["report.recall"] == pytest.approx(0.5)

    assert metrics["report.baseline_precision"] == pytest.approx(1.0)
    assert metrics["report.baseline_recall"] == pytest.approx(1.0)

    assert metrics["stats.detect_seconds"] == pytest.approx(2.5)
    # train_pairs/train_positives/threshold_pairs/threshold_positives are already carried by
    # train.pairs/train.positives/tune.pairs/tune.positives — must not pass through twice.
    for key in (
        "stats.train_pairs",
        "stats.train_positives",
        "stats.threshold_pairs",
        "stats.threshold_positives",
    ):
        assert key not in metrics


def test_evaluate_detection_scopes_stock_photo_control_to_report_not_pooled():
    # (11, 12) and (15, 16) share an agency photo set (100) across two different areas and are
    # not duplicates. (11, 12) sits in the report split, (15, 16) in train — so the report-scoped
    # stock-photo control count must be 1 while the pooled count is 2, and the model/baseline
    # false-positive rates over those two scopes must differ too.
    rows = [
        _pairs_row(8, 9, "report", 0.9, True, True),
        _pairs_row(11, 12, "report", 0.05, False, False, baseline=True),
        _pairs_row(15, 16, "train", 0.3, False, False),
    ]
    pairs = pl.DataFrame(rows)
    attributes = pl.DataFrame(
        {
            "listing_id": [8, 9, 11, 12, 15, 16],
            "photo_set_id": [8, 9, 100, 100, 100, 100],
            "area_id": [8, 9, 50, 51, 50, 51],
        }
    )
    truth = pl.DataFrame(schema={"listing_a": pl.Int64, "listing_b": pl.Int64, "pattern": pl.Utf8})
    metrics = evaluate_detection(
        type("R", (), {"pairs": pairs, "threshold": 0.5, "stats": {}})(), truth, attributes, CONFIG
    )

    assert metrics["control.stock_photo.pairs"] == pytest.approx(1.0)
    assert metrics["control.stock_photo.model_fp_rate"] == pytest.approx(0.0)
    assert metrics["control.stock_photo.baseline_fp_rate"] == pytest.approx(1.0)

    assert metrics["pooled.control.stock_photo.pairs"] == pytest.approx(2.0)
    assert metrics["pooled.control.stock_photo.model_fp_rate"] == pytest.approx(0.0)
    assert metrics["pooled.control.stock_photo.baseline_fp_rate"] == pytest.approx(0.5)

    # no planted pairs at all here: retrieval.recall's own empty-denominator convention is
    # 0.0, not NaN (see the module docstring on where NaN does and doesn't apply).
    assert metrics["retrieval.recall"] == pytest.approx(0.0)


def test_evaluate_detection_scopes_same_building_control_to_report_not_pooled():
    # (1, 2) is a same-building control pair in the report split; (5, 6) is another one, but
    # in the train split. The report-scoped control.same_building.* must see only (1, 2); the
    # pooled counterpart must see both, with different fp rates on each side — the same
    # report-vs-pooled treatment as the stock-photo control above, previously untested for
    # same_building since the brief's own fixture is entirely report-split.
    rows = [
        _pairs_row(1, 2, "report", 0.9, False, False, same_building=True, baseline=True),
        _pairs_row(3, 4, "report", 0.9, True, True),
        _pairs_row(5, 6, "train", 0.9, True, False, same_building=True, baseline=False),
    ]
    pairs = pl.DataFrame(rows)
    attributes = pl.DataFrame(
        {
            "listing_id": [1, 2, 3, 4, 5, 6],
            "photo_set_id": [1, 2, 3, 4, 5, 6],
            "area_id": [1, 2, 3, 4, 5, 6],
        }
    )
    truth = pl.DataFrame(schema={"listing_a": pl.Int64, "listing_b": pl.Int64, "pattern": pl.Utf8})
    metrics = evaluate_detection(
        type("R", (), {"pairs": pairs, "threshold": 0.5, "stats": {}})(), truth, attributes, CONFIG
    )

    assert metrics["control.same_building.pairs"] == pytest.approx(1.0)
    assert metrics["control.same_building.model_fp_rate"] == pytest.approx(0.0)
    assert metrics["control.same_building.baseline_fp_rate"] == pytest.approx(1.0)

    assert metrics["pooled.control.same_building.pairs"] == pytest.approx(2.0)
    assert metrics["pooled.control.same_building.model_fp_rate"] == pytest.approx(0.5)
    assert metrics["pooled.control.same_building.baseline_fp_rate"] == pytest.approx(0.5)


def test_evaluate_detection_pattern_recall_denominator_is_date_derived_not_retrieval():
    # Reproduces the exact bug the reviewer demonstrated against the installed code: pattern
    # recall must not mix a report-scoped numerator with a pooled-across-splits denominator.
    #
    # Two planted pairs: (1, 2)'s later posting date (listing 2, day 2) falls inside the train
    # block, (3, 4)'s later date (listing 4, day 4) falls in the report block — per
    # listings.detect.assign_pair_split's own date logic, not by whether/where retrieval
    # flagged them. Both are flagged wherever they were retrieved. Labeling them with two
    # different patterns also exercises "a pattern with zero report-split members" in the
    # same fixture: exact_repost's only planted pair is train-split by date, so its
    # report-scoped denominator is a real, computed zero — not "we couldn't tell" — and its
    # recall is undefined (NaN), never the misleading 0.5 a pooled denominator used to
    # produce here when nothing that belonged in the report split was actually missed.
    rows = [
        _pairs_row(1, 2, "train", 0.9, True, True),
        _pairs_row(3, 4, "report", 0.9, True, True),
    ]
    pairs = pl.DataFrame(rows)
    attributes = pl.DataFrame(
        {
            "listing_id": [1, 2, 3, 4],
            "photo_set_id": [1, 2, 3, 4],
            "area_id": [1, 2, 3, 4],
            "posted_at": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4)],
        }
    )
    truth = pl.DataFrame(
        {"listing_a": [1, 3], "listing_b": [2, 4], "pattern": ["exact_repost", "reworded"]}
    )
    metrics = evaluate_detection(
        type("R", (), {"pairs": pairs, "threshold": 0.5, "stats": {}})(), truth, attributes, CONFIG
    )

    # exact_repost's only planted pair is train-split by date: a real, computed zero-sized
    # report-split population (not "couldn't compute"), so pairs is 0.0 and recall is NaN.
    assert metrics["pattern.exact_repost.pairs"] == pytest.approx(0.0)
    assert math.isnan(metrics["pattern.exact_repost.recall"])
    # the pooled counterpart still sees the train-split hit.
    assert metrics["pooled.pattern.exact_repost.recall"] == pytest.approx(1.0)

    # reworded's planted pair is report-split by date and was flagged there: recall is a
    # clean 1.0 — not the 0.5 a shared, unscoped denominator would have produced by also
    # counting exact_repost's train-split pair against a pooled population of 2.
    assert metrics["pattern.reworded.pairs"] == pytest.approx(1.0)
    assert metrics["pattern.reworded.recall"] == pytest.approx(1.0)


def test_evaluate_detection_price_shift_recall_gap_isolates_shifted_clones():
    # Five report-split duplicates: two listed at the same price (both caught), three with a
    # price gap (one caught) -- the split the corpus's price-shifted reposts create. With
    # three of five shifted the median gap is non-zero, so a median split would put one
    # shifted pair on the "unshifted" side; the metric must split on "any gap" instead.
    rows = [
        _pairs_row(1, 2, "report", 0.95, True, True, ratio=0.0),
        _pairs_row(3, 4, "report", 0.95, True, True, ratio=0.0),
        _pairs_row(5, 6, "report", 0.60, True, True, ratio=0.30),
        _pairs_row(7, 8, "report", 0.10, False, True, ratio=0.35),
        _pairs_row(9, 10, "report", 0.10, False, True, ratio=0.02),
    ]
    pairs = pl.DataFrame(rows)
    attributes = pl.DataFrame(
        {
            "listing_id": list(range(1, 11)),
            "photo_set_id": list(range(1, 11)),
            "area_id": list(range(1, 11)),
        }
    )
    truth = pl.DataFrame(
        {
            "listing_a": [1, 3, 5, 7, 9],
            "listing_b": [2, 4, 6, 8, 10],
            "pattern": ["exact_repost"] * 5,
        }
    )
    metrics = evaluate_detection(
        type("R", (), {"pairs": pairs, "threshold": 0.5, "stats": {}})(), truth, attributes, CONFIG
    )

    assert metrics["price_shift.unshifted.pairs"] == pytest.approx(2.0)
    assert metrics["price_shift.unshifted.recall"] == pytest.approx(1.0)
    assert metrics["price_shift.shifted.pairs"] == pytest.approx(3.0)
    assert metrics["price_shift.shifted.recall"] == pytest.approx(1 / 3)
    # the gap this metric exists to surface
    assert metrics["price_shift.shifted.recall"] < metrics["price_shift.unshifted.recall"]
    assert not [key for key in metrics if "median" in key]


def test_evaluate_detection_price_shift_gap_is_nan_without_report_duplicates():
    # No report-split duplicate at all: both buckets are undefined (NaN), not zero -- there is
    # no computed "zero pairs" fact here, only an uncomputed one. A bucket that is merely
    # empty while the other is not gets 0 pairs and a NaN recall.
    attributes = pl.DataFrame({"listing_id": [1, 2], "photo_set_id": [1, 2], "area_id": [1, 2]})
    truth = pl.DataFrame(schema={"listing_a": pl.Int64, "listing_b": pl.Int64, "pattern": pl.Utf8})
    none = pl.DataFrame([_pairs_row(1, 2, "report", 0.9, True, False, ratio=0.05)])
    metrics = evaluate_detection(
        type("R", (), {"pairs": none, "threshold": 0.5, "stats": {}})(), truth, attributes, CONFIG
    )
    for key in ("unshifted.pairs", "unshifted.recall", "shifted.pairs", "shifted.recall"):
        assert math.isnan(metrics[f"price_shift.{key}"])

    one = pl.DataFrame([_pairs_row(1, 2, "report", 0.9, True, True, ratio=0.05)])
    metrics = evaluate_detection(
        type("R", (), {"pairs": one, "threshold": 0.5, "stats": {}})(), truth, attributes, CONFIG
    )
    assert metrics["price_shift.shifted.pairs"] == pytest.approx(1.0)
    assert metrics["price_shift.unshifted.pairs"] == pytest.approx(0.0)
    assert math.isnan(metrics["price_shift.unshifted.recall"])


def test_evaluate_detection_pr_auc_matches_hand_computed_average_precision():
    # report-split points sorted by score descending: (0.9, dup), (0.7, not), (0.5, dup), (0.2, not).
    # Average precision (sklearn's definition): AP = sum_n (R_n - R_{n-1}) * P_n over ranks n,
    # where R_n/P_n are recall/precision after including the n-th highest-scored sample:
    #   rank1 score=0.9 label=1: TP=1 FP=0 -> P=1.0    R=1/2=0.500  dR=0.500  contributes 0.500000
    #   rank2 score=0.7 label=0: TP=1 FP=1 -> P=0.5    R=0.500      dR=0.000  contributes 0
    #   rank3 score=0.5 label=1: TP=2 FP=1 -> P=2/3    R=2/2=1.000  dR=0.500  contributes 0.333333
    #   rank4 score=0.2 label=0: TP=2 FP=2 -> P=0.5    R=1.000      dR=0.000  contributes 0
    # AP = 0.500000 + 0.333333... = 5/6 = 0.8333333... (confirmed against
    # sklearn.metrics.average_precision_score directly, not just by hand).
    #
    # A train row (score=1.0, not a duplicate) and a threshold row (score=0.05, a duplicate)
    # are included with values that would change this number if pr_auc were computed from
    # `decision` instead of `score` (decisions are all correct here, so an accidental swap
    # would read as a perfect 1.0 instead of 5/6), or pooled across all six rows instead of
    # report-only (independently checked against sklearn: that pools to 0.5, not 5/6).
    rows = [
        _pairs_row(1, 2, "report", 0.9, True, True),
        _pairs_row(3, 4, "report", 0.7, False, False),
        _pairs_row(5, 6, "report", 0.5, True, True),
        _pairs_row(7, 8, "report", 0.2, False, False),
        _pairs_row(9, 10, "train", 1.0, False, False),
        _pairs_row(11, 12, "threshold", 0.05, True, True),
    ]
    pairs = pl.DataFrame(rows)
    attributes = pl.DataFrame(
        {
            "listing_id": list(range(1, 13)),
            "photo_set_id": list(range(1, 13)),
            "area_id": list(range(1, 13)),
        }
    )
    truth = pl.DataFrame(schema={"listing_a": pl.Int64, "listing_b": pl.Int64, "pattern": pl.Utf8})
    metrics = evaluate_detection(
        type("R", (), {"pairs": pairs, "threshold": 0.5, "stats": {}})(), truth, attributes, CONFIG
    )
    assert metrics["report.pr_auc"] == pytest.approx(5 / 6)


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
            "listing_id": [1, 2, 3, 3, 4, 5],
            "flag": [
                "bait_price",
                "bait_price",
                "photo_reuse",
                "photo_reuse",  # listing 3 appears twice under the same flag
                "inconsistent_relist",
                "inconsistent_relist",
            ],
            "detail": ["{}"] * 6,
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
    # listing 3's duplicate photo_reuse row must not double-count it.
    assert metrics["fraud.photo_reuse.flagged"] == pytest.approx(1.0)
    assert metrics["fraud.inconsistent_relist.flagged"] == pytest.approx(2.0)
    assert metrics["stats.bait_price_skipped"] == pytest.approx(0.0)
    assert metrics["stats.listings"] == pytest.approx(5.0)


def _dated_attributes(n: int) -> pl.DataFrame:
    """n listings posted one per day, so the 60/20/20 blocks are easy to count."""
    return pl.DataFrame(
        {
            "listing_id": list(range(1, n + 1)),
            "posted_at": [date(2023, 1, 1) + timedelta(days=i) for i in range(n)],
            "photo_set_id": list(range(1, n + 1)),
            "area_id": list(range(1, n + 1)),
        }
    )


def _relist_flags(ids):
    return pl.DataFrame(
        {
            "listing_id": ids,
            "flag": ["inconsistent_relist"] * len(ids),
            "detail": ["{}"] * len(ids),
        },
        schema={"listing_id": pl.Int64, "flag": pl.Utf8, "detail": pl.Utf8},
    )


EMPTY_TRUTH = pl.DataFrame(
    schema={
        "listing_id": pl.Int64, "dup_group_id": pl.Int64,
        "control_group_id": pl.Utf8, "fraud_label": pl.Utf8,
    }
)  # fmt: skip


def test_evaluate_fraud_scores_relist_on_report_listings_only():
    # 10 listings, one per day: 1-6 train, 7-8 threshold, 9-10 report.
    attributes = _dated_attributes(10)
    # flagged: 1, 2 (train, both true), 9 (report, true), 10 (report, false)
    # truth:   1, 2, 9 and 8 (threshold block)
    flags = _relist_flags([1, 2, 9, 10])
    relist_truth = pl.DataFrame({"listing_id": [1, 2, 8, 9]})
    metrics = evaluate_fraud(
        type("F", (), {"flags": flags, "stats": {}})(),
        EMPTY_TRUTH,
        CONFIG,
        relist_truth,
        attributes,
    )
    assert metrics["fraud.inconsistent_relist.flagged"] == pytest.approx(4.0)  # all listings
    assert metrics["fraud.inconsistent_relist.report_flagged"] == pytest.approx(2.0)
    assert metrics["fraud.inconsistent_relist.report_positives"] == pytest.approx(1.0)
    assert metrics["fraud.inconsistent_relist.precision"] == pytest.approx(0.5)
    assert metrics["fraud.inconsistent_relist.recall"] == pytest.approx(1.0)
    assert metrics["pooled.fraud.inconsistent_relist.precision"] == pytest.approx(3 / 4)
    assert metrics["pooled.fraud.inconsistent_relist.recall"] == pytest.approx(3 / 4)


def test_relist_rates_are_nan_when_undefined():
    attributes = _dated_attributes(10)
    metrics = evaluate_fraud(
        type("F", (), {"flags": _relist_flags([]), "stats": {}})(),
        EMPTY_TRUTH,
        CONFIG,
        pl.DataFrame({"listing_id": []}, schema={"listing_id": pl.Int64}),
        attributes,
    )
    assert metrics["fraud.inconsistent_relist.flagged"] == 0.0
    assert math.isnan(metrics["fraud.inconsistent_relist.precision"])
    assert math.isnan(metrics["fraud.inconsistent_relist.recall"])
    # Without relist truth the rates are simply not computed.
    metrics = evaluate_fraud(
        type("F", (), {"flags": _relist_flags([1]), "stats": {}})(), EMPTY_TRUTH, CONFIG
    )
    assert "fraud.inconsistent_relist.precision" not in metrics


def test_end_to_end_recall_divides_by_every_planted_report_pair():
    # 10 listings one per day; pairs (2,9) and (3,10) are planted and report-split by date.
    # Only (2,9) was retrieved (and flagged), so report.recall is 1.0 but end-to-end is 0.5.
    attributes = _dated_attributes(10)
    pairs = pl.DataFrame([_pairs_row(2, 9, "report", 0.9, True, True)])
    truth = pl.DataFrame(
        {"listing_a": [2, 3], "listing_b": [9, 10], "pattern": ["exact_repost", "reworded"]}
    )
    metrics = evaluate_detection(
        type("R", (), {"pairs": pairs, "threshold": 0.5, "stats": {}})(), truth, attributes, CONFIG
    )
    assert metrics["report.recall"] == pytest.approx(1.0)
    assert metrics["report.planted_pairs"] == pytest.approx(2.0)
    assert metrics["report.end_to_end_recall"] == pytest.approx(0.5)


def test_threshold_table_scores_the_tuning_split_only():
    from listings.evaluate import threshold_table

    pairs = pl.DataFrame(
        [
            _pairs_row(1, 2, "threshold", 0.999, True, True),
            _pairs_row(3, 4, "threshold", 0.97, True, False),
            _pairs_row(5, 6, "threshold", 0.6, False, True),
            _pairs_row(7, 8, "report", 0.9999, True, False),  # must be ignored
        ]
    )
    table = threshold_table(pairs, chosen=0.97)
    rows = {row["threshold"]: row for row in table.iter_rows(named=True)}
    assert rows[0.97]["chosen"] and sum(table["chosen"]) == 1
    assert (rows[0.97]["flagged"], rows[0.97]["precision"]) == (2, 0.5)
    assert rows[0.97]["recall"] == pytest.approx(0.5)
    assert (rows[0.5]["flagged"], rows[0.5]["recall"]) == (3, 1.0)
    assert rows[0.999]["precision"] == 1.0
    assert rows[0.9999]["flagged"] == 0 and math.isnan(rows[0.9999]["precision"])


def test_confusion_matrix_is_report_split_for_model_and_baseline():
    from listings.evaluate import confusion_matrix

    pairs = pl.DataFrame(
        [
            _pairs_row(1, 2, "report", 0.9, True, True, baseline=True),
            _pairs_row(3, 4, "report", 0.9, True, False, baseline=True),
            _pairs_row(5, 6, "report", 0.1, False, True, baseline=False),
            _pairs_row(7, 8, "report", 0.1, False, False, baseline=True),
            _pairs_row(9, 10, "train", 0.9, True, False),  # must be ignored
        ]
    )
    matrix = confusion_matrix(pairs)
    assert matrix["model"] == {
        "true_positives": 1, "false_positives": 1, "false_negatives": 1, "true_negatives": 1,
    }  # fmt: skip
    assert matrix["baseline"] == {
        "true_positives": 1, "false_positives": 2, "false_negatives": 1, "true_negatives": 0,
    }  # fmt: skip


def test_run_artifacts_are_written_as_strict_json_and_csv(tmp_path):
    import json

    from listings.evaluate import write_run_artifacts

    pairs = pl.DataFrame(
        [
            _pairs_row(1, 2, "threshold", 0.9, True, True),
            _pairs_row(3, 4, "report", 0.9, True, True),
        ]
    )
    result = type("R", (), {"pairs": pairs, "threshold": 0.9})()
    timings = {"build": {"seconds": 1.5, "finished_at": "2026-09-16T00:00:00+00:00"}}
    paths = write_run_artifacts(result, timings, tmp_path)
    assert [path.name for path in paths] == [
        "threshold_table.csv", "confusion_matrix.json", "timings.json",
    ]  # fmt: skip
    assert json.loads(paths[1].read_text())["model"]["true_positives"] == 1
    assert json.loads(paths[2].read_text()) == timings
    assert pl.read_csv(paths[0]).height >= 1
    with pytest.raises(ValueError):
        write_run_artifacts(result, {"evaluate": {"seconds": float("nan")}}, tmp_path)
