"""Measure the detector against ground truth and log the run to MLflow.

Precision, recall and f1 are reported per split — `train.*`, `tune.*` (the data's
`split` column still says "threshold"; the metric prefix says "tune" so it can
never be confused with the `threshold` decision-cutoff metric), and `report.*` —
so nobody mistakes an in-sample number for a held-out one. `report.*` is the only
headline.

`control.*` rates are scored against report-split decisions only: a control pair
only exists in this metric because it was retrieved and evaluated, and its split
comes straight from the same `pairs` frame the model scored, so filtering that
frame to `split == "report"` is exact. Pooling across splits would readmit the
in-sample contamination that made an earlier run's headline precision read 92.5%
instead of the held-out number the threshold was actually tuned to hit.
`pooled.control.*` is logged alongside it, explicitly named, for debugging —
never under the bare name a reader would take as the held-out figure.

`pattern.{pattern}.recall` is the fraction of PLANTED `{pattern}` clone -> source
pairs — the full truth population, not just the ones retrieval happened to find —
whose later listing's posting date falls in the report split (via
`listings.detect.assign_pair_split`, applied to the truth pairs directly, not to
`pairs`) that the detector actually flagged; `pattern.{pattern}.pairs` is that
denominator's size. A planted pair retrieval never found still belongs in the
denominator and still scores as a miss, so this stays a true end-to-end recall on
held-out data rather than a number conditional on retrieval succeeding — that
conditional question is what `retrieval.recall` measures separately, below.
`pooled.pattern.{pattern}.recall` runs the identical numerator/denominator logic
over every split's planted pairs at once, for debugging only, never as the
headline.

`retrieval.recall` is the one metric that stays pooled across all splits on
purpose: candidate generation runs before any model is fit or any threshold is
picked, so it carries no in-sample leakage, and splitting it would only shrink
an already-small denominator (the planted pairs).

A rate whose denominator cannot be determined at all — a pattern's report-split
population when posting dates aren't available to compute it, or too few
report-split duplicates to price-shift-bucket — is logged as `float("nan")`,
never `0.0`: "caught none of these" and "there were none of these to catch" must
not share a representation. This governs `pattern.*.recall`, `pattern.*.pairs`
and `price_shift.*`. It does NOT extend to the per-split precision/recall/f1 from
`pair_metrics` (`train.*`/`tune.*`/`report.*`) or to `retrieval.recall` with no
planted pairs at all — both of those fall back to `0.0` on an empty/zero
denominator, inherited from `pair_metrics`' own convention.
"""

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

from listings.config import DetectConfig
from listings.detect import assign_pair_split
from listings.truth import load_duplicate_truth  # noqa: F401  (re-exported for the CLI)

# The `split` column's data values (unchanged — assign_pair_split in detect.py still writes
# "threshold") mapped to the metric-name prefix each one is logged under.
SPLIT_METRIC_PREFIX = {"train": "train", "threshold": "tune", "report": "report"}
PATTERNS = ("exact_repost", "reworded", "edited_photo")
# Already carried by train.pairs/train.positives and tune.pairs/tune.positives — do not also
# pass these four through under stats.*.
_STATS_EXCLUDE = {"train_pairs", "train_positives", "threshold_pairs", "threshold_positives"}
_NAN = float("nan")


def pair_metrics(labels, decisions) -> dict[str, float]:
    labels = np.asarray(labels, dtype=bool)
    decisions = np.asarray(decisions, dtype=bool)
    true_positives = float(np.sum(labels & decisions))
    false_positives = float(np.sum(~labels & decisions))
    false_negatives = float(np.sum(labels & ~decisions))
    precision = true_positives / (true_positives + false_positives) if decisions.any() else 0.0
    recall = true_positives / (true_positives + false_negatives) if labels.any() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "pairs": float(labels.size),
    }


def stock_photo_control_pairs(pairs: pl.DataFrame, attributes: pl.DataFrame) -> pl.Series:
    """Unrelated listings that happen to share an agency photo set: must not be flagged."""
    lookup = attributes.select("listing_id", "photo_set_id", "area_id")
    # maintain_order="left": the caller filters the ORIGINAL `pairs` frame with this mask, so
    # it must stay row-aligned with `pairs` — detect.py's own join of this shape pins order
    # the same way, and an unordered join has bitten a previous task in this phase.
    joined = pairs.join(
        lookup, left_on="listing_a", right_on="listing_id", how="left", maintain_order="left"
    ).join(
        lookup,
        left_on="listing_b",
        right_on="listing_id",
        how="left",
        suffix="_b",
        maintain_order="left",
    )
    mask = (
        (pl.col("photo_set_id") == pl.col("photo_set_id_b"))
        & (pl.col("area_id") != pl.col("area_id_b"))
        & ~pl.col("is_duplicate")
    ).fill_null(False)
    return joined.select(mask.alias("stock_photo_control"))["stock_photo_control"]


def _control_rates(pairs: pl.DataFrame, mask: pl.Series, prefix: str) -> dict[str, float]:
    """`prefix` carries both the scope (`control.` or `pooled.control.`) and the control name."""
    controls = pairs.filter(mask)
    if controls.is_empty():
        return {f"{prefix}.pairs": 0.0}
    return {
        f"{prefix}.pairs": float(controls.height),
        f"{prefix}.model_fp_rate": float(controls["decision"].mean()),
        f"{prefix}.baseline_fp_rate": float(controls["baseline_decision"].mean()),
    }


def _price_shift_recall_gap(report: pl.DataFrame) -> dict[str, float]:
    """Recall on report-split duplicates split by abs_log_price_ratio vs. their median.

    abs_log_price_ratio carries the largest coefficient in the pair model (-4.31), and
    ~300 of the 1,200 planted exact reposts carry a shifted asking price. If those
    price-shifted clones score below threshold disproportionately, "above median"
    recall reads visibly lower than "below median" recall here.
    """
    # There is no meaningful "0 pairs" to report when the computation itself could not run
    # (no column, or too few report-split duplicates to split into two groups) — that would
    # claim a fact ("zero pairs") we never actually checked. NaN throughout.
    insufficient_data = {
        "price_shift.below_median.pairs": _NAN,
        "price_shift.below_median.recall": _NAN,
        "price_shift.above_median.pairs": _NAN,
        "price_shift.above_median.recall": _NAN,
    }
    if "abs_log_price_ratio" not in report.columns:
        return insufficient_data
    duplicates = report.filter(pl.col("is_duplicate"))
    if duplicates.height < 2:
        return insufficient_data
    ratios = duplicates["abs_log_price_ratio"].to_numpy()
    median = float(np.median(ratios))
    below = duplicates.filter(pl.Series(ratios <= median))
    above = duplicates.filter(pl.Series(ratios > median))
    return {
        "price_shift.below_median.pairs": float(below.height),
        "price_shift.below_median.recall": float(below["decision"].mean())
        if below.height
        else _NAN,
        "price_shift.above_median.pairs": float(above.height),
        "price_shift.above_median.recall": float(above["decision"].mean())
        if above.height
        else _NAN,
    }


def _flagged_pairs(frame: pl.DataFrame) -> set[tuple[int, int]]:
    return {
        (a, b)
        for a, b, decision in zip(
            frame["listing_a"].to_list(), frame["listing_b"].to_list(), frame["decision"].to_list()
        )
        if decision
    }


def _pattern_recall(subset: list[tuple[int, int]], flagged: set[tuple[int, int]]) -> float:
    if not subset:
        return _NAN
    return float(sum(pair in flagged for pair in subset) / len(subset))


def _report_split_truth(
    truth_pairs: pl.DataFrame, attributes: pl.DataFrame, config: DetectConfig
) -> pl.DataFrame | None:
    """Planted pairs restricted to the report split, with the split derived from posting
    dates via `listings.detect.assign_pair_split` applied to the truth pairs directly —
    never from the retrieved `pairs` frame. A planted pair retrieval never found still
    needs a split assigned so it still counts in the report-split denominator; scoping by
    what was retrieved instead would silently drop retrieval misses from the denominator
    and turn `pattern.*.recall` into a number conditional on retrieval succeeding.

    Returns None when the split cannot be determined at all: `assign_pair_split` indexes
    into attributes' sorted `posted_at` column and raises `IndexError` on an empty frame,
    and needs that column to exist in the first place.
    """
    if truth_pairs.is_empty() or attributes.is_empty() or "posted_at" not in attributes.columns:
        return None
    return assign_pair_split(truth_pairs, attributes, config).filter(pl.col("split") == "report")


def evaluate_detection(
    result, truth_pairs: pl.DataFrame, attributes: pl.DataFrame, config: DetectConfig
) -> dict[str, float]:
    pairs = result.pairs
    metrics: dict[str, float] = {}

    # Per-split precision/recall/f1, split sizes and positive counts — train and tune are
    # in-sample/tuning splits and must never be read as the headline number.
    for split_value, prefix in SPLIT_METRIC_PREFIX.items():
        split_pairs = pairs.filter(pl.col("split") == split_value)
        metrics |= {
            f"{prefix}.{key}": value
            for key, value in pair_metrics(
                split_pairs["is_duplicate"], split_pairs["decision"]
            ).items()
        }
        metrics[f"{prefix}.positives"] = (
            float(split_pairs["is_duplicate"].sum()) if split_pairs.height else 0.0
        )

    report = pairs.filter(pl.col("split") == "report")
    metrics |= {
        f"report.baseline_{key}": value
        for key, value in pair_metrics(report["is_duplicate"], report["baseline_decision"]).items()
    }
    if report["is_duplicate"].any() and not report["is_duplicate"].all():
        metrics["report.pr_auc"] = float(
            average_precision_score(report["is_duplicate"].to_numpy(), report["score"].to_numpy())
        )
    else:
        metrics["report.pr_auc"] = _NAN

    metrics |= _price_shift_recall_gap(report)

    # Pooled on purpose (see module docstring): candidate generation runs before any model
    # is fit, so it carries no in-sample leakage.
    retrieved = set(zip(pairs["listing_a"].to_list(), pairs["listing_b"].to_list()))
    planted = list(zip(truth_pairs["listing_a"].to_list(), truth_pairs["listing_b"].to_list()))
    metrics["retrieval.recall"] = (
        float(sum(pair in retrieved for pair in planted) / len(planted)) if planted else 0.0
    )

    patterns = truth_pairs["pattern"].to_list() if truth_pairs.height else []
    report_flagged = _flagged_pairs(report)
    pooled_flagged = _flagged_pairs(pairs)

    report_truth = _report_split_truth(truth_pairs, attributes, config)
    if report_truth is not None:
        report_planted = list(
            zip(report_truth["listing_a"].to_list(), report_truth["listing_b"].to_list())
        )
        report_patterns = report_truth["pattern"].to_list()
    else:
        report_planted = report_patterns = None

    for pattern in PATTERNS:
        pooled_subset = [pair for pair, kind in zip(planted, patterns) if kind == pattern]
        metrics[f"pooled.pattern.{pattern}.recall"] = _pattern_recall(pooled_subset, pooled_flagged)
        if report_planted is None:
            metrics[f"pattern.{pattern}.pairs"] = _NAN
            metrics[f"pattern.{pattern}.recall"] = _NAN
            continue
        report_subset = [
            pair for pair, kind in zip(report_planted, report_patterns) if kind == pattern
        ]
        metrics[f"pattern.{pattern}.pairs"] = float(len(report_subset))
        metrics[f"pattern.{pattern}.recall"] = _pattern_recall(report_subset, report_flagged)

    metrics |= _control_rates(report, report["same_building_control"], "control.same_building")
    metrics |= _control_rates(pairs, pairs["same_building_control"], "pooled.control.same_building")
    metrics |= _control_rates(
        report, stock_photo_control_pairs(report, attributes), "control.stock_photo"
    )
    metrics |= _control_rates(
        pairs, stock_photo_control_pairs(pairs, attributes), "pooled.control.stock_photo"
    )
    metrics["threshold"] = float(result.threshold)
    metrics |= {
        f"stats.{key}": float(value)
        for key, value in result.stats.items()
        if key not in _STATS_EXCLUDE
    }
    return metrics


def evaluate_fraud(fraud_result, truth: pl.DataFrame, config: DetectConfig) -> dict[str, float]:
    metrics = {f"stats.{key}": float(value) for key, value in fraud_result.stats.items()}
    flagged = set(fraud_result.flags.filter(pl.col("flag") == "bait_price")["listing_id"].to_list())
    actual = set(truth.filter(pl.col("fraud_label") == "bait_price")["listing_id"].to_list())
    hits = len(flagged & actual)
    metrics["fraud.bait_price.flagged"] = float(len(flagged))
    metrics["fraud.bait_price.precision"] = float(hits / len(flagged)) if flagged else 0.0
    metrics["fraud.bait_price.recall"] = float(hits / len(actual)) if actual else 0.0
    for name in ("photo_reuse", "inconsistent_relist"):
        # Distinct listings, not row count: a listing must never be double-counted just
        # because it happened to accumulate more than one row for the same flag.
        metrics[f"fraud.{name}.flagged"] = float(
            fraud_result.flags.filter(pl.col("flag") == name)["listing_id"].n_unique()
        )
    return metrics


def save_pr_curve(scores, labels, path: Path) -> Path:
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    plt.switch_backend("Agg")
    precision, recall, _ = precision_recall_curve(
        np.asarray(labels, dtype=bool), np.asarray(scores, dtype=float)
    )
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(recall, precision)
    axis.set_xlabel("recall")
    axis.set_ylabel("precision")
    axis.set_title("Duplicate detection: precision vs recall")
    axis.grid(alpha=0.3)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return path


def log_run(
    metrics: dict[str, float],
    params: dict,
    artifacts,
    config: DetectConfig,
    tracking_uri: str | None = None,
    artifact_location: str | None = None,
) -> str:
    import math

    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if artifact_location and mlflow.get_experiment_by_name(config.experiment) is None:
        mlflow.create_experiment(config.experiment, artifact_location=artifact_location)
    mlflow.set_experiment(config.experiment)
    with mlflow.start_run(run_name="listing-dedup") as run:
        mlflow.log_params({key: str(value) for key, value in params.items()})
        mlflow.log_metrics(
            {key: float(value) for key, value in metrics.items() if math.isfinite(float(value))}
        )
        for artifact in artifacts:
            mlflow.log_artifact(str(artifact))
        return run.info.run_id
