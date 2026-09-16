"""Command line: python -m search queries|train|evaluate|query."""

import argparse
import dataclasses
import json
import logging
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from ingestion.config import DbSettings
from listings.embed import FakeEmbedder, SentenceTransformerEmbedder, resolve_device
from listings.fraud import resolve_price_model_version
from search.config import DATA_DIR, SearchConfig
from search.engine import SearchEngine
from search.evaluate import (
    CONTENDERS,
    evaluate_report,
    log_results,
    search_run,
    write_artifacts,
)
from search.features import feature_table, refresh_estimates
from search.lexicon import load_lexicon
from search.queries import build_query_set
from search.ranker import load_pinned
from search.store import (
    apply_schema,
    latest_corpus_run,
    queries_corpus_run,
    replace_query_set,
)
from search.train import (
    fit_ablation,
    gate_baseline,
    gate_passes,
    register_ranker,
    train_rankers,
)

TIMINGS_FILE = "stage_timings.json"
STATS_FILE = "queries_stats.json"
DEVICES = ("auto", "cuda", "cpu")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m search", description="Synthetic queries, ranker training and search."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    queries = commands.add_parser("queries", help="generate, retrieve and grade the query set")
    queries.add_argument("--n", type=int, default=SearchConfig.n_queries)
    queries.add_argument("--seed", type=int, default=SearchConfig.seed)
    queries.add_argument("--device", choices=DEVICES, default="auto")
    queries.add_argument("--fake", action="store_true", help="use the deterministic test embedder")
    queries.add_argument("--data-dir", type=Path, default=DATA_DIR)

    train = commands.add_parser("train", help="fit, evaluate, gate and register the ranker")
    train.add_argument("--trials", type=int, default=SearchConfig.n_trials)
    train.add_argument("--device", choices=DEVICES, default="auto")
    train.add_argument("--data-dir", type=Path, default=DATA_DIR)

    evaluate = commands.add_parser("evaluate", help="re-score the registered champion")
    evaluate.add_argument("--data-dir", type=Path, default=DATA_DIR)

    query = commands.add_parser("query", help="search the corpus and explain the results")
    query.add_argument("text")
    query.add_argument("--k", type=int, default=10)
    query.add_argument("--fake", action="store_true")
    query.add_argument("--device", choices=DEVICES, default="auto")
    query.add_argument("--data-dir", type=Path, default=DATA_DIR)
    return parser


def _utf8_console() -> None:
    """Windows pipes default to cp1252, which cannot encode the ✓ in the reasons."""
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        if encoding != "utf8" and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _utf8_console()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parser().parse_args(argv)
    load_dotenv()  # before any settings are read
    commands = {"queries": _queries, "train": _train, "evaluate": _evaluate, "query": _query}
    args.started = time.perf_counter()
    try:
        status = commands[args.command](args)
    except Exception as exc:  # noqa: BLE001 — CLI boundary: report and exit 1
        print(f"{args.command} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if status == 0 and args.command != "query":
        seconds = time.perf_counter() - args.started
        _record_timing(args.data_dir, args.command, seconds)
        print(f"{args.command} took {seconds:.1f}s")
    return status


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _record_timing(data_dir: Path, stage: str, seconds: float) -> None:
    path = Path(data_dir) / TIMINGS_FILE
    timings = {} if stage == "queries" else _read_json(path)
    timings[stage] = {
        "seconds": round(seconds, 3),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(timings, indent=2, allow_nan=False), encoding="utf-8")


def _embedder(args):
    if args.fake:
        return FakeEmbedder(), "cpu"
    device = resolve_device(args.device)
    return SentenceTransformerEmbedder(device=device), device


def _latest_detect_run(conn) -> int | None:
    with conn.cursor() as cur:
        cur.execute("SELECT max(detect_run_id) FROM listings.detect_runs")
        return cur.fetchone()[0]


def _queries(args) -> int:
    config = dataclasses.replace(SearchConfig(), n_queries=args.n, seed=args.seed)
    embedder, device = _embedder(args)
    conn = DbSettings.from_env().connect()
    try:
        apply_schema(conn)
        conn.commit()
        stats = refresh_estimates(conn, config)
        conn.commit()
        print(f"building {config.n_queries} queries (embedding on {device})")
        queries, judgments = build_query_set(conn, embedder, load_lexicon(conn), config)
        replace_query_set(conn, queries, judgments)
        conn.commit()
    finally:
        conn.close()
    if stats["value_features_skipped"]:
        print(
            "WARNING: value features were SKIPPED — the price model "
            f"({config.price_model_uri}) could not be loaded; price_to_estimate and "
            "within_interval will be missing for every listing.",
            file=sys.stderr,
        )
    counts = queries.group_by("split", "kind").len().sort("split", "kind")
    summary = {
        **stats,
        "queries": queries.height,
        "judgments": judgments.height,
        "counts": {f"{split}.{kind}": n for split, kind, n in counts.iter_rows()},
    }
    path = Path(args.data_dir) / STATS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"{queries.height} queries, {judgments.height:,} graded candidates")
    for key, value in summary["counts"].items():
        print(f"  {key:<24}{value:>6}")
    return 0


def _require_fresh_queries(conn) -> int:
    corpus_run_id = latest_corpus_run(conn)
    if queries_corpus_run(conn) != corpus_run_id:
        raise RuntimeError(
            "the stored query set was built on an older corpus — run `python -m search queries`"
        )
    return corpus_run_id


def _print_contenders(metrics: dict[str, float], names) -> None:
    print(f"{'contender':<22}{'NDCG@10':>9}{'95% CI':>18}{'MRR':>8}{'P@5':>8}")
    for name in names:
        if f"{name}.ndcg_at_10" not in metrics:
            continue
        ci = f"{metrics[f'{name}.ci_low']:.3f}-{metrics[f'{name}.ci_high']:.3f}"
        print(
            f"{name:<22}{metrics[f'{name}.ndcg_at_10']:>9.3f}{ci:>18}"
            f"{metrics[f'{name}.mrr']:>8.3f}{metrics[f'{name}.precision_at_5']:>8.3f}"
        )


def _train(args) -> int:
    config = dataclasses.replace(
        SearchConfig(), device=resolve_device(args.device), n_trials=args.trials
    )
    conn = DbSettings.from_env().connect()
    try:
        corpus_run_id = _require_fresh_queries(conn)
        detect_run_id = _latest_detect_run(conn)
        lexicon = load_lexicon(conn)
        table = feature_table(conn, lexicon)
        result = train_rankers(table, config, config.n_trials)
        ablation = fit_ablation(table, result, config)
        evaluation = evaluate_report(
            conn, table, result.rankers, lexicon, config, champion=result.winner, ablation=ablation
        )
    finally:
        conn.close()
    stats = _read_json(Path(args.data_dir) / STATS_FILE)
    metrics = dict(evaluation.metrics)
    for kind, value in result.tune_ndcg.items():
        metrics[f"tune.{kind}.ndcg_at_10"] = value
    winner = result.winner
    baseline, baseline_ndcg = gate_baseline(metrics)
    passed = gate_passes(
        metrics[f"{winner}.ndcg_at_10"], metrics[f"{winner}.ci_low"], baseline_ndcg
    )
    metrics["gate.passed"] = float(passed)
    metrics["gate.baseline_ndcg"] = baseline_ndcg
    # only this query set's own `queries` timing: never an older train or evaluate entry
    stored = _read_json(Path(args.data_dir) / TIMINGS_FILE)
    timings = {"queries": stored["queries"]} if "queries" in stored else {}
    timings["train_before_logging"] = {"seconds": round(time.perf_counter() - args.started, 3)}
    for stage, entry in timings.items():
        metrics[f"timing.{stage}_seconds"] = float(entry["seconds"])

    with tempfile.TemporaryDirectory() as tmp, search_run(config, "search-train"):
        version = (
            register_ranker(result.rankers[winner], config, Path(tmp) / "ranker")
            if passed
            else None
        )
        artifacts = write_artifacts(
            Path(tmp) / "artifacts", evaluation, result.rankers[winner].importance(), timings
        )
        params = {
            "seed": config.seed,
            "n_queries": stats.get("queries", "unknown"),
            "trials": config.n_trials,
            "device": config.device,
            "winner": winner,
            "winner_params": json.dumps(result.best_params[winner], sort_keys=True),
            "price_model_uri": config.price_model_uri,
            "price_model_version": resolve_price_model_version(config.price_model_uri)
            or "unavailable",
            "value_features_skipped": stats.get("value_features_skipped", "unknown"),
            "corpus_run_id": corpus_run_id,
            "detect_run_id": detect_run_id,
            "registered_version": version or "none",
            "gate.baseline": baseline,
        }
        log_results(metrics, params, artifacts)

    _print_contenders(metrics, [*CONTENDERS, "ablation.no_trust"])
    print("tune NDCG@10: " + ", ".join(f"{k} {v:.3f}" for k, v in result.tune_ndcg.items()))
    for name in [*CONTENDERS, "ablation.no_trust"]:
        if f"{name}.fraud.top10_share" in metrics:
            print(f"fraud top-10 share {name}: {metrics[f'{name}.fraud.top10_share']:.4f}")
    print(f"fraud share among report candidates: {metrics['candidates.fraud_share']:.4f}")
    verdict = f"registered {config.ranker_name} v{version}" if passed else "not registered"
    print(
        f"gate: winner {winner} {'PASSED' if passed else 'FAILED'} against {baseline} "
        f"({baseline_ndcg:.3f}) — {verdict}"
    )
    return 0


def _evaluate(args) -> int:
    config = SearchConfig()
    champion, ranker_version = load_pinned(config.ranker_uri)
    if champion is None:
        raise RuntimeError("no champion registered — run python -m search train first")
    conn = DbSettings.from_env().connect()
    try:
        corpus_run_id = _require_fresh_queries(conn)
        lexicon = load_lexicon(conn)
        table = feature_table(conn, lexicon)
        evaluation = evaluate_report(
            conn, table, {champion.kind: champion}, lexicon, config, champion=champion.kind
        )
    finally:
        conn.close()
    with tempfile.TemporaryDirectory() as tmp, search_run(config, "search-evaluate"):
        artifacts = write_artifacts(
            Path(tmp),
            evaluation,
            champion.importance(),
            _read_json(Path(args.data_dir) / TIMINGS_FILE),
        )
        log_results(
            evaluation.metrics,
            {
                "ranker_uri": config.ranker_uri,
                "ranker_version": ranker_version,
                "ranker_kind": champion.kind,
                "corpus_run_id": corpus_run_id,
            },
            artifacts,
        )
    _print_contenders(evaluation.metrics, [champion.kind, *CONTENDERS[2:]])
    return 0


def _query(args) -> int:
    embedder, _ = _embedder(args)
    conn = DbSettings.from_env().connect()
    try:
        step = time.perf_counter()
        engine = SearchEngine(conn, embedder)
        ready_ms = (time.perf_counter() - step) * 1000
        result = engine.search(args.text, args.k)
    finally:
        conn.close()
    understood = {k: v for k, v in result.parsed.to_dict().items() if v not in (None, [], "")}
    print(f"understood: {json.dumps(understood)}")
    print(f"ranker: {result.ranker}")
    for note in result.notes:
        print(f"note: {note}")
    for position, hit in enumerate(result.results, start=1):
        beds = "?" if hit.bedrooms is None else hit.bedrooms
        print(
            f"{position:>2}. #{hit.listing_id} {hit.title} | {hit.area_name} | {beds} bed | "
            f"{hit.size_sqm:.0f} sqm | AED {hit.asking_price_aed:,.0f}"
        )
        hidden = f" (+{hit.duplicates_hidden} duplicate)" if hit.duplicates_hidden else ""
        print(f"    {', '.join(hit.reasons)}{hidden}")
    print(f"engine ready in {ready_ms:.0f} ms (data, ranker and embedder warm-up)")
    print(f"took {result.timings_ms['total']:.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
