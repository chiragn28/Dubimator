import json
from datetime import date

import numpy as np
import polars as pl
import pytest

from listings.config import PAIR_FEATURES, DetectConfig
from listings.detect import (
    assign_pair_split,
    baseline_decisions,
    choose_threshold,
    fit_pair_model,
    run_detection,
    write_detection,
)

CONFIG = DetectConfig()


def test_pairs_are_split_by_the_later_listing():
    attributes = pl.DataFrame(
        {
            "listing_id": [1, 2, 3, 4, 5],
            "posted_at": [date(2023, 1, d) for d in (1, 2, 3, 4, 5)],
        }
    )
    pairs = pl.DataFrame({"listing_a": [1, 1, 1], "listing_b": [2, 4, 5]})
    split = assign_pair_split(pairs, attributes, CONFIG)
    assert split["split"].to_list() == ["train", "threshold", "report"]
    assert split["pair_date"].to_list() == [date(2023, 1, 2), date(2023, 1, 4), date(2023, 1, 5)]


def test_threshold_is_the_deepest_cut_that_still_meets_precision():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    labels = np.array([True, True, True, False, True])
    # top 3 are all duplicates (precision 1.0); adding the 4th drops precision to 0.75
    assert choose_threshold(scores, labels, 0.98) == pytest.approx(0.7)
    assert choose_threshold(scores, labels, 0.75) == pytest.approx(0.5)


def test_threshold_flags_nothing_when_precision_is_unreachable():
    scores = np.array([0.9, 0.8])
    labels = np.array([False, False])
    assert choose_threshold(scores, labels, 0.98) > 0.9


def test_baseline_uses_image_similarity_alone():
    features = pl.DataFrame({"image_max_cosine": [0.96, 0.94]})
    assert baseline_decisions(features, CONFIG).tolist() == [True, False]


def test_training_without_positive_pairs_fails_loudly():
    features = pl.DataFrame({name: [0.0, 1.0] for name in PAIR_FEATURES})
    with pytest.raises(ValueError, match="no duplicate pairs"):
        fit_pair_model(features, np.array([False, False]), CONFIG)


def test_detection_scores_duplicates_above_controls(loaded_corpus):
    settings, _, _ = loaded_corpus
    conn = settings.connect()
    try:
        result = run_detection(conn, CONFIG)
    finally:
        conn.close()

    assert set(PAIR_FEATURES) <= set(result.pairs.columns)
    assert {"score", "decision", "split", "is_duplicate", "baseline_decision"} <= set(
        result.pairs.columns
    )
    assert 0.0 <= result.threshold <= 1.0
    duplicates = result.pairs.filter(pl.col("is_duplicate"))
    controls = result.pairs.filter(pl.col("same_building_control"))
    assert duplicates.height > 0 and controls.height > 0
    assert duplicates["score"].mean() > controls["score"].mean()
    assert result.stats["candidate_pairs"] == result.pairs.height


def test_detection_beats_the_single_signal_baseline_on_controls(loaded_corpus):
    settings, _, _ = loaded_corpus
    conn = settings.connect()
    try:
        result = run_detection(conn, CONFIG)
    finally:
        conn.close()
    controls = result.pairs.filter(pl.col("same_building_control"))
    model_flags = controls["decision"].sum()
    baseline_flags = controls["baseline_decision"].sum()
    assert model_flags <= baseline_flags, (
        f"the model flagged {model_flags} same-building controls, the image-only baseline "
        f"{baseline_flags} — multi-signal agreement is supposed to help here"
    )


def test_write_detection_stores_flagged_pairs_with_their_signals(loaded_corpus):
    settings, _, corpus_run_id = loaded_corpus
    conn = settings.connect()
    try:
        result = run_detection(conn, CONFIG)
        detect_run_id = write_detection(conn, result, corpus_run_id)
        conn.commit()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT listing_a, listing_b, score, decision, signals FROM listings.duplicate_pairs "
                "WHERE detect_run_id = %s",
                (detect_run_id,),
            )
            rows = cur.fetchall()
            cur.execute(
                "SELECT corpus_run_id, threshold FROM listings.detect_runs WHERE detect_run_id = %s",
                (detect_run_id,),
            )
            run = cur.fetchone()
    finally:
        conn.close()

    flagged = result.pairs.filter(pl.col("decision"))
    assert flagged.height > 0, "the fixture must flag something, or this test proves nothing"
    assert len(rows) == flagged.height
    assert all(row[0] < row[1] for row in rows)
    assert all(row[3] is True for row in rows)
    signals = rows[0][4]
    signals = json.loads(signals) if isinstance(signals, str) else signals
    assert set(signals) == set(PAIR_FEATURES)
    assert run[0] == corpus_run_id
    assert run[1] == pytest.approx(result.threshold)
