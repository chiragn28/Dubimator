"""Measure the detector against ground truth and log the run to MLflow.

Precision and recall are reported per split (train / threshold / report), never
pooled: the pooled number in an earlier run mixed the in-sample train block into
the headline figure, which is how a threshold tuned for 0.98 precision on held-out
data read as 92.5% overall. `report.*` is the only held-out headline; `train.*` and
`threshold.*` are logged alongside it, distinctly prefixed, so nobody mistakes an
in-sample number for one.
"""

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

from listings.config import DetectConfig
from listings.truth import load_duplicate_truth  # noqa: F401  (re-exported for the CLI)

SPLITS = ("train", "threshold", "report")
PATTERNS = ("exact_repost", "reworded", "edited_photo")


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
    joined = pairs.join(lookup, left_on="listing_a", right_on="listing_id", how="left").join(
        lookup, left_on="listing_b", right_on="listing_id", how="left", suffix="_b"
    )
    mask = (
        (pl.col("photo_set_id") == pl.col("photo_set_id_b"))
        & (pl.col("area_id") != pl.col("area_id_b"))
        & ~pl.col("is_duplicate")
    ).fill_null(False)
    return joined.select(mask.alias("stock_photo_control"))["stock_photo_control"]


def _control_rates(pairs: pl.DataFrame, mask: pl.Series, name: str) -> dict[str, float]:
    controls = pairs.filter(mask)
    if controls.is_empty():
        return {f"control.{name}.pairs": 0.0}
    return {
        f"control.{name}.pairs": float(controls.height),
        f"control.{name}.model_fp_rate": float(controls["decision"].mean()),
        f"control.{name}.baseline_fp_rate": float(controls["baseline_decision"].mean()),
    }


def _price_shift_recall_gap(report: pl.DataFrame) -> dict[str, float]:
    """Recall on report-split duplicates split by abs_log_price_ratio vs. their median.

    abs_log_price_ratio carries the largest coefficient in the pair model (-4.31), and
    ~300 of the 1,200 planted exact reposts carry a shifted asking price. If those
    price-shifted clones score below threshold disproportionately, "above median"
    recall reads visibly lower than "below median" recall here.
    """
    defaults = {
        "price_shift.below_median.pairs": 0.0,
        "price_shift.below_median.recall": 0.0,
        "price_shift.above_median.pairs": 0.0,
        "price_shift.above_median.recall": 0.0,
    }
    if "abs_log_price_ratio" not in report.columns:
        return defaults
    duplicates = report.filter(pl.col("is_duplicate"))
    if duplicates.height < 2:
        return defaults
    ratios = duplicates["abs_log_price_ratio"].to_numpy()
    median = float(np.median(ratios))
    below = duplicates.filter(pl.Series(ratios <= median))
    above = duplicates.filter(pl.Series(ratios > median))
    return {
        "price_shift.below_median.pairs": float(below.height),
        "price_shift.below_median.recall": float(below["decision"].mean()) if below.height else 0.0,
        "price_shift.above_median.pairs": float(above.height),
        "price_shift.above_median.recall": float(above["decision"].mean()) if above.height else 0.0,
    }


def evaluate_detection(
    result, truth_pairs: pl.DataFrame, attributes: pl.DataFrame, config: DetectConfig
) -> dict[str, float]:
    pairs = result.pairs
    metrics: dict[str, float] = {}

    # Per-split precision/recall/f1, split sizes and positive counts — train and threshold
    # are in-sample/tuning splits and must never be read as the headline number.
    for split_name in SPLITS:
        split_pairs = pairs.filter(pl.col("split") == split_name)
        metrics |= {
            f"{split_name}.{key}": value
            for key, value in pair_metrics(
                split_pairs["is_duplicate"], split_pairs["decision"]
            ).items()
        }
        metrics[f"{split_name}.positives"] = (
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
        metrics["report.pr_auc"] = float("nan")

    metrics |= _price_shift_recall_gap(report)

    retrieved = set(zip(pairs["listing_a"].to_list(), pairs["listing_b"].to_list()))
    planted = list(zip(truth_pairs["listing_a"].to_list(), truth_pairs["listing_b"].to_list()))
    metrics["retrieval.recall"] = (
        float(sum(pair in retrieved for pair in planted) / len(planted)) if planted else 0.0
    )

    flagged = {
        (a, b)
        for a, b, decision in zip(
            pairs["listing_a"].to_list(), pairs["listing_b"].to_list(), pairs["decision"].to_list()
        )
        if decision
    }
    for pattern in PATTERNS:
        subset = [
            (a, b)
            for (a, b), kind in zip(planted, truth_pairs["pattern"].to_list())
            if kind == pattern
        ]
        metrics[f"pattern.{pattern}.recall"] = (
            float(sum(pair in flagged for pair in subset) / len(subset)) if subset else 0.0
        )

    metrics |= _control_rates(pairs, pairs["same_building_control"], "same_building")
    metrics |= _control_rates(pairs, stock_photo_control_pairs(pairs, attributes), "stock_photo")
    metrics["threshold"] = float(result.threshold)
    metrics |= {f"stats.{key}": float(value) for key, value in result.stats.items()}
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
        metrics[f"fraud.{name}.flagged"] = float(
            fraud_result.flags.filter(pl.col("flag") == name).height
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
