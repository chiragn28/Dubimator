"""Split pairs by time, fit the pair model, pick a threshold, write decisions."""

import json
import time
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ingestion.load import copy_frame
from listings.candidates import fetch_candidates
from listings.config import PAIR_FEATURES, DetectConfig
from listings.features import (
    build_features,
    load_listing_attributes,
    load_listing_vectors,
    load_photo_sets,
)
from listings.truth import label_pairs, load_truth

PRICE_FEATURE = "abs_log_price_ratio"
RELIST_PAIR_SCHEMA = {"listing_a": pl.Int64, "listing_b": pl.Int64, "price_blind_score": pl.Float64}


@dataclass(frozen=True)
class DetectionResult:
    pairs: pl.DataFrame
    threshold: float
    model: Pipeline
    feature_names: tuple[str, ...]
    stats: dict[str, float]
    # Pairs the SAME fitted model flags at the SAME threshold once the price gap is zeroed.
    # inconsistent_relist looks for duplicate clusters whose prices disagree, which the
    # headline decision (it penalises a price gap heavily) can by construction never supply.
    relist_pairs: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(schema=RELIST_PAIR_SCHEMA)
    )


def split_cutoffs(attributes: pl.DataFrame, config: DetectConfig):
    """The last posting dates of the train block and of the threshold block."""
    dates = attributes["posted_at"].sort()

    def block_end(share: float):
        """The last date inside the first `share` of listings, not the first date after it."""
        return dates[min(max(int(dates.len() * share) - 1, 0), dates.len() - 1)]

    return block_end(config.train_share), block_end(config.train_share + config.threshold_share)


def assign_pair_split(
    pairs: pl.DataFrame, attributes: pl.DataFrame, config: DetectConfig
) -> pl.DataFrame:
    """A pair belongs to the split of its later listing: 'is this new post a repost?'"""
    posted = attributes.select("listing_id", "posted_at")
    joined = pairs.join(
        posted, left_on="listing_a", right_on="listing_id", how="inner", maintain_order="left"
    ).join(
        posted,
        left_on="listing_b",
        right_on="listing_id",
        how="inner",
        suffix="_b",
        maintain_order="left",
    )
    train_end, threshold_end = split_cutoffs(attributes, config)
    later = pl.max_horizontal("posted_at", "posted_at_b")
    split = (
        pl.when(later <= train_end)
        .then(pl.lit("train"))
        .when(later <= threshold_end)
        .then(pl.lit("threshold"))
        .otherwise(pl.lit("report"))
    )
    return joined.with_columns(later.alias("pair_date"), split.alias("split")).drop(
        "posted_at", "posted_at_b"
    )


def fit_pair_model(features: pl.DataFrame, labels: np.ndarray, config: DetectConfig) -> Pipeline:
    labels = np.asarray(labels, dtype=bool)
    if labels.sum() == 0:
        raise ValueError("training split has no duplicate pairs to learn from")
    if (~labels).sum() == 0:
        raise ValueError("training split has no non-duplicate pairs to learn from")
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=config.seed
                ),
            ),
        ]
    )
    model.fit(features.select(PAIR_FEATURES).to_numpy(), labels)
    return model


def choose_threshold(scores: np.ndarray, labels: np.ndarray, target_precision: float) -> float:
    """The lowest score at which running precision still meets the target.

    A threshold always stays inside [0, 1], the range of a probability: it is stored in a
    double column and compared against predict_proba output. When nothing reaches the target
    the cut sits just above the best score so that nothing is flagged, clamped to 1.0 — a
    score of exactly 1.0 does then flag, which is the one degenerate case worth the tidy range.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    if scores.size == 0:
        return 1.0
    order = np.argsort(-scores)
    ranked_scores, ranked_labels = scores[order], labels[order]
    precision = np.cumsum(ranked_labels) / np.arange(1, scores.size + 1)
    acceptable = np.flatnonzero(precision >= target_precision)
    if acceptable.size == 0:
        return min(float(ranked_scores[0]) + 1e-9, 1.0)
    return float(ranked_scores[acceptable[-1]])


def baseline_decisions(features: pl.DataFrame, config: DetectConfig) -> np.ndarray:
    """The single-signal rule the multi-signal model is measured against."""
    return features["image_max_cosine"].to_numpy() >= config.baseline_image_cosine


def price_blind_scores(model: Pipeline, features: pl.DataFrame) -> np.ndarray:
    """The fitted pair model's scores with the price gap set to 0 — nothing else changes."""
    blind = features.select(PAIR_FEATURES).with_columns(pl.lit(0.0).alias(PRICE_FEATURE))
    return model.predict_proba(blind.select(PAIR_FEATURES).to_numpy())[:, 1]


def run_detection(conn, config: DetectConfig) -> DetectionResult:
    started = time.perf_counter()
    pairs, candidate_stats = fetch_candidates(conn, config)
    attributes = load_listing_attributes(conn)
    text_vectors, image_vectors = load_listing_vectors(conn)
    listing_photo_ids, photo_vectors = load_photo_sets(conn)

    features = build_features(
        pairs.select("listing_a", "listing_b"),
        attributes,
        text_vectors,
        image_vectors,
        listing_photo_ids,
        photo_vectors,
    )
    features = assign_pair_split(features, attributes, config)
    labelled = features.join(
        label_pairs(features.select("listing_a", "listing_b"), load_truth(conn)),
        on=["listing_a", "listing_b"],
        how="left",
        maintain_order="left",
    ).with_columns(
        pl.col("is_duplicate").fill_null(False),
        pl.col("same_building_control").fill_null(False),
    )

    train = labelled.filter(pl.col("split") == "train")
    model = fit_pair_model(train, train["is_duplicate"].to_numpy(), config)
    scores = model.predict_proba(labelled.select(PAIR_FEATURES).to_numpy())[:, 1]
    labelled = labelled.with_columns(pl.Series("score", scores))

    validation = labelled.filter(pl.col("split") == "threshold")
    threshold = choose_threshold(
        validation["score"].to_numpy(),
        validation["is_duplicate"].to_numpy(),
        config.target_precision,
    )
    labelled = labelled.with_columns(
        (pl.col("score") >= threshold).alias("decision"),
        pl.Series("baseline_decision", baseline_decisions(labelled, config)),
    )
    blind = price_blind_scores(model, labelled)
    relist_pairs = (
        labelled.select("listing_a", "listing_b")
        .with_columns(pl.Series("price_blind_score", blind, dtype=pl.Float64))
        .filter(pl.col("price_blind_score") >= threshold)
    )
    stats = {
        "candidate_pairs": float(labelled.height),
        "text_pairs": float(candidate_stats.text_pairs),
        "image_pairs": float(candidate_stats.image_pairs),
        "shared_photo_pairs": float(candidate_stats.shared_photo_pairs),
        "retrieval_seconds": candidate_stats.seconds,
        "detect_seconds": time.perf_counter() - started,
        "train_pairs": float(train.height),
        # Positives per split: a threshold block with no duplicates in it flags nothing, and
        # without these counts that run looks exactly like a broken one.
        "train_positives": float(train["is_duplicate"].sum()),
        "threshold_pairs": float(validation.height),
        "threshold_positives": float(validation["is_duplicate"].sum()),
        "flagged": float(labelled["decision"].sum()),
        "price_blind_flagged": float(relist_pairs.height),
        "text_seconds": candidate_stats.text_seconds,
        "image_seconds": candidate_stats.image_seconds,
        "shared_photo_seconds": candidate_stats.shared_photo_seconds,
    }
    return DetectionResult(labelled, threshold, model, PAIR_FEATURES, stats, relist_pairs)


def write_detection(conn, result: DetectionResult, corpus_run_id: int) -> int:
    """Record the run and store the flagged pairs with the signals behind each decision."""
    flagged = result.pairs.filter(pl.col("decision"))
    # allow_nan=False: a non-finite feature fails here, with the row in hand, rather than
    # producing NaN/Infinity literals that jsonb rejects halfway through the COPY.
    signals = [
        json.dumps({name: row[name] for name in PAIR_FEATURES}, allow_nan=False)
        for row in flagged.iter_rows(named=True)
    ]
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO listings.detect_runs (corpus_run_id, threshold, metrics) "
            "VALUES (%s, %s, %s::jsonb) RETURNING detect_run_id",
            (corpus_run_id, result.threshold, json.dumps(result.stats)),
        )
        detect_run_id = cur.fetchone()[0]
        if flagged.height:
            frame = flagged.select(
                pl.lit(detect_run_id, dtype=pl.Int64).alias("detect_run_id"),
                "listing_a",
                "listing_b",
                "score",
                "decision",
                pl.Series("signals", signals),
            )
            copy_frame(cur, "listings.duplicate_pairs", frame)
    return detect_run_id
