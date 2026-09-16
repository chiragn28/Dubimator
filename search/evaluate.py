"""Report-split evaluation: every contender, the Phase 4 effects, parser accuracy and retrieval
recall, plus artifacts and MLflow logging.

This module reads ground truth (fraud labels, true slots) on purpose: it is where answers are
compared with predictions.
"""

import json
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import polars as pl

from ingestion.normalize import match_key
from models.price.registry import configure, log_metrics
from search.config import SLOTS, SearchConfig
from search.features import meets_every_slot, rule_grade
from search.lexicon import Lexicon
from search.metrics import bootstrap_ci, per_query_metrics, summarize
from search.parse import ParsedQuery, parse
from search.store import read_queries, read_query_labels

CONTENDERS = (
    "xgboost", "lightgbm", "baseline_newest", "baseline_semantic", "baseline_fused",
    "baseline_rules",
)  # fmt: skip
MISSING_POSITION = 1e9
MAX_PARSE_ERRORS = 200
MONEY_TOLERANCE = 0.01


def baseline_scores(frame: pl.DataFrame, near_margin: float = 0.10) -> dict[str, np.ndarray]:
    """Label-free orderings. `baseline_rules` scores 3 / 2 / 1 from the PARSED slots with the
    grading rules' shape (see features.rule_grade) and breaks ties on the fusion score, which
    is always below 1."""
    semantic = frame["semantic_pos"].fill_nan(None).fill_null(MISSING_POSITION).to_numpy()
    rules = frame.select(rule_grade(near_margin) + pl.col("rrf_score")).to_series()
    return {
        "baseline_newest": -frame["days_since_posted"].to_numpy().astype(np.float64),
        "baseline_semantic": -semantic.astype(np.float64),
        "baseline_fused": -frame["fused_pos"].to_numpy().astype(np.float64),
        "baseline_rules": rules.to_numpy().astype(np.float64),
    }


def contender_scores(
    frame: pl.DataFrame, rankers: dict, near_margin: float = 0.10
) -> dict[str, np.ndarray]:
    scores = {name: np.asarray(ranker.score(frame)) for name, ranker in rankers.items()}
    return {**scores, **baseline_scores(frame, near_margin)}


def _scored(frame: pl.DataFrame, values: np.ndarray) -> pl.DataFrame:
    return frame.with_columns(pl.Series("score", np.asarray(values, dtype=np.float64)))


def evaluate_contenders(frame, scores, config: SearchConfig):
    metrics: dict[str, float] = {}
    per_query: dict[str, pl.DataFrame] = {}
    for name, values in scores.items():
        table = per_query_metrics(_scored(frame, values), "score", config.ndcg_k)
        per_query[name] = table
        for key, value in summarize(table).items():
            if key.startswith("kind."):
                kind = key.split(".")[1]
                metrics[f"kind.{kind}.{name}.ndcg_at_10"] = value
            elif key.startswith("no_match."):
                metrics[f"kind.no_match.{name}.{key.split('.', 1)[1]}"] = value
            else:
                metrics[f"{name}.{key}"] = value
        answerable = table.filter(pl.col("answerable"))["ndcg"].to_numpy()
        low, high = bootstrap_ci(answerable, config.n_bootstrap, config.seed)
        metrics[f"{name}.ci_low"], metrics[f"{name}.ci_high"] = low, high
    return metrics, per_query


def top_k_effects(frame, scores, fraud_ids: set[int], k: int, prefix: str = "") -> dict[str, float]:
    table = per_query_metrics(_scored(frame, scores), "score", k)
    sizes = dict(zip(frame["listing_id"].to_list(), frame["cluster_size"].to_list()))
    tops = table["top_ids"].to_list()
    hidden = [sum(int(sizes[listing]) - 1 for listing in top) for top in tops]
    slots = sum(len(top) for top in tops)
    flagged = sum(listing in fraud_ids for top in tops for listing in top)
    return {
        f"{prefix}dup.top10_removed": float(np.mean(hidden)) if hidden else math.nan,
        f"{prefix}fraud.top10_share": flagged / slots if slots else math.nan,
    }


def closest_note_rates(frame, scores, k: int, near_margin: float = 0.10) -> dict[str, float]:
    """How often the engine would say "nothing meets every requirement": no top-k candidate
    meets every parsed slot exactly. Split by `no_match` versus answerable queries."""
    table = _scored(frame, scores).with_columns(
        meets_every_slot(near_margin).alias("meets"),
        pl.col("score").fill_nan(float("-inf")).alias("rank_score"),
    )
    top = (
        table.sort(["query_id", "rank_score", "fused_pos"], descending=[False, True, False])
        .group_by("query_id", maintain_order=True)
        .agg(pl.col("kind").first(), pl.col("meets").head(k).any().alias("any_meets"))
        .with_columns((~pl.col("any_meets")).alias("noted"))
    )
    no_match = top.filter(pl.col("kind") == "no_match")["noted"]
    others = top.filter(pl.col("kind") != "no_match")["noted"]
    return {
        "note.closest.rate.no_match": float(no_match.mean()) if no_match.len() else math.nan,
        "note.closest.rate.other": float(others.mean()) if others.len() else math.nan,
    }


def load_fraud_listing_ids(conn) -> set[int]:
    with conn.cursor() as cur:
        cur.execute("SELECT listing_id FROM listings.listings WHERE fraud_label IS NOT NULL")
        return {row[0] for row in cur.fetchall()}


def load_true_slots(conn, split: str = "report") -> dict[int, dict]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT query_id, true_slots FROM search.queries WHERE split = %s ORDER BY query_id",
            (split,),
        )
        return {query_id: dict(slots) for query_id, slots in cur.fetchall()}


def _close(parsed: float | None, truth: float | None) -> bool:
    if parsed is None or truth is None:
        return parsed is None and truth is None
    return abs(parsed - truth) <= MONEY_TOLERANCE * max(abs(truth), 1.0)


def _key(name: str | None) -> str | None:
    return match_key(name) if name else None


def slot_checks(parsed: ParsedQuery, truth: dict) -> dict[str, tuple[bool, object, object]]:
    area_ok = (
        truth["area_id"] in parsed.area_ids
        if truth["area_id"] is not None
        else parsed.area_name is None
    )
    return {
        "area": (area_ok, truth["area_id"], list(parsed.area_ids)),
        "building": (
            _key(parsed.building) == _key(truth["building"]),
            truth["building"],
            parsed.building,
        ),
        "bedrooms": (parsed.bedrooms == truth["bedrooms"], truth["bedrooms"], parsed.bedrooms),
        "property_type": (
            parsed.property_type == truth["property_type"],
            truth["property_type"],
            parsed.property_type,
        ),
        "budget_min": (
            _close(parsed.budget_min, truth["budget_min"]),
            truth["budget_min"],
            parsed.budget_min,
        ),
        "budget_max": (
            _close(parsed.budget_max, truth["budget_max"]),
            truth["budget_max"],
            parsed.budget_max,
        ),
        "min_size_sqm": (
            _close(parsed.min_size_sqm, truth["min_size_sqm"]),
            truth["min_size_sqm"],
            parsed.min_size_sqm,
        ),
        "amenities": (
            set(parsed.amenities) == set(truth["amenities"]),
            sorted(truth["amenities"]),
            sorted(parsed.amenities),
        ),
    }


def parser_accuracy(queries: pl.DataFrame, slots: dict[int, dict], lexicon: Lexicon):
    right = dict.fromkeys(SLOTS, 0)
    errors = []
    rows = queries.select("query_id", "text").iter_rows()
    for query_id, text in rows:
        for slot, (ok, expected, got) in slot_checks(parse(text, lexicon), slots[query_id]).items():
            right[slot] += ok
            if not ok and len(errors) < MAX_PARSE_ERRORS:
                errors.append((query_id, text, slot, json.dumps(expected), json.dumps(got)))
    total = queries.height
    accuracy = {
        f"parse.{slot}.accuracy": (right[slot] / total if total else math.nan) for slot in SLOTS
    }
    schema = {
        "query_id": pl.Int64,
        "text": pl.Utf8,
        "slot": pl.Utf8,
        "expected": pl.Utf8,
        "parsed": pl.Utf8,
    }
    return accuracy, pl.DataFrame(errors, schema=schema, orient="row")


def retrieval_recall(queries: pl.DataFrame, report_rows: pl.DataFrame) -> float:
    """queries: query_id and n_grade3 (from store.read_query_labels)."""
    found = report_rows.group_by("query_id").agg((pl.col("grade") == 3).sum().alias("found"))
    answerable = (
        queries.filter(pl.col("n_grade3") > 0)
        .join(found, on="query_id", how="left")
        .with_columns(pl.col("found").fill_null(0))
    )
    if answerable.height == 0:
        return math.nan
    return float((answerable["found"] / answerable["n_grade3"]).mean())


@dataclass(frozen=True)
class Evaluation:
    metrics: dict[str, float]
    per_query: dict[str, pl.DataFrame]
    parse_errors: pl.DataFrame
    report_queries: pl.DataFrame  # query_id, text, kind, parsed (JSON)


def evaluate_report(
    conn,
    table: pl.DataFrame,
    rankers: dict,
    lexicon: Lexicon,
    config: SearchConfig,
    champion: str | None = None,
    ablation=None,
) -> Evaluation:
    report = table.filter(pl.col("split") == "report").sort("query_id", "fused_pos")
    scores = contender_scores(report, rankers, config.near_margin)
    metrics, per_query = evaluate_contenders(report, scores, config)
    fraud_ids = load_fraud_listing_ids(conn)
    for name, values in scores.items():
        metrics |= top_k_effects(report, values, fraud_ids, config.ndcg_k, f"{name}.")
    metrics["candidates.fraud_share"] = (
        float(report["listing_id"].is_in(list(fraud_ids)).mean()) if report.height else math.nan
    )
    if champion is not None:
        metrics |= top_k_effects(report, scores[champion], fraud_ids, config.ndcg_k)
        metrics |= closest_note_rates(report, scores[champion], config.ndcg_k, config.near_margin)
    if ablation is not None:
        ablation_scores = ablation.score(report)
        extra, _ = evaluate_contenders(report, {"ablation.no_trust": ablation_scores}, config)
        metrics |= extra
        metrics |= top_k_effects(
            report, ablation_scores, fraud_ids, config.ndcg_k, "ablation.no_trust."
        )
    queries = read_queries(conn, splits=("report",))
    accuracy, errors = parser_accuracy(queries, load_true_slots(conn), lexicon)
    metrics |= accuracy
    metrics["retrieval.recall_at_200"] = retrieval_recall(
        read_query_labels(conn, splits=("report",)), report
    )
    metrics["report.queries"] = float(queries.height)
    parsed = [json.dumps(parse(text, lexicon).to_dict()) for text in queries["text"].to_list()]
    report_queries = queries.select("query_id", "text", "kind").with_columns(
        pl.Series("parsed", parsed, dtype=pl.Utf8)
    )
    return Evaluation(metrics, per_query, errors, report_queries)


def _comparison_chart(metrics: dict[str, float], path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [name for name in CONTENDERS if f"{name}.ndcg_at_10" in metrics]
    values = [metrics[f"{name}.ndcg_at_10"] for name in names]
    errors = np.array(
        [
            [value - metrics[f"{name}.ci_low"], metrics[f"{name}.ci_high"] - value]
            for name, value in zip(names, values)
        ]
    ).T
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(names, values, yerr=np.nan_to_num(errors), capsize=4, color="#0e6e7e")
    axis.set_ylabel("NDCG@10 (report split, 95% CI)")
    axis.set_ylim(0, 1)
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def write_artifacts(
    directory: Path, evaluation: Evaluation, importance: pl.DataFrame, timings: dict
) -> list[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report = evaluation.report_queries
    for name, table in evaluation.per_query.items():
        report = report.join(
            table.select("query_id", pl.col("ndcg").alias(f"ndcg.{name}")),
            on="query_id",
            how="left",
        )
    paths = [
        _comparison_chart(evaluation.metrics, directory / "ndcg_comparison.png"),
        directory / "feature_importance.csv",
        directory / "per_query_report.csv",
        directory / "parse_errors.csv",
        directory / "timings.json",
    ]
    importance.sort("gain", descending=True).write_csv(paths[1])
    report.sort("query_id").write_csv(paths[2])
    evaluation.parse_errors.write_csv(paths[3])
    paths[4].write_text(json.dumps(timings, indent=2, allow_nan=False), encoding="utf-8")
    return paths


@contextmanager
def search_run(
    config: SearchConfig,
    run_name: str,
    tracking_uri: str | None = None,
    artifact_location: str | None = None,
) -> Iterator:
    configure(config.experiment, tracking_uri, artifact_location)
    with mlflow.start_run(run_name=run_name) as run:
        yield run


def log_results(metrics: dict[str, float], params: dict, artifacts: list[Path]) -> None:
    mlflow.log_params({key: str(value) for key, value in params.items()})
    log_metrics(metrics)
    for artifact in artifacts:
        mlflow.log_artifact(str(artifact))
