"""Command line: python -m listings build|embed|detect|evaluate."""

import argparse
import dataclasses
import logging
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from ingestion.config import DbSettings
from listings.candidates import brute_force_pairs
from listings.config import CorpusConfig, DetectConfig
from listings.detect import run_detection, write_detection
from listings.embed import (
    FakeEmbedder,
    SentenceTransformerEmbedder,
    embed_listings,
    embed_photos,
    resolve_device,
)
from listings.evaluate import evaluate_detection, evaluate_fraud, log_run, save_pr_curve
from listings.fraud import run_fraud_checks, write_fraud_flags
from listings.generate import generate_corpus, load_areas, load_sales, render_variants
from listings.load import (
    create_vector_indexes,
    latest_corpus_run,
    load_corpus,
    read_corpus_for_embedding,
    write_listing_embeddings,
    write_photo_embeddings,
)
from listings.photos import DATA_DIR, ensure_pool
from listings.truth import load_duplicate_truth, load_truth


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m listings",
        description="Build the synthetic listings corpus and detect duplicates and fraud.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="generate the corpus and load it into Postgres")
    build.add_argument("--listings", type=int, default=CorpusConfig.n_listings)
    build.add_argument("--seed", type=int, default=CorpusConfig.seed)
    build.add_argument("--data-dir", type=Path, default=DATA_DIR)

    embed = commands.add_parser(
        "embed", help="embed photos and text, then build the vector indexes"
    )
    embed.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    embed.add_argument("--data-dir", type=Path, default=DATA_DIR)
    embed.add_argument("--fake", action="store_true", help="use the deterministic test embedder")

    commands.add_parser("detect", help="score candidates and write duplicates and fraud flags")
    evaluate = commands.add_parser(
        "evaluate", help="measure against ground truth and log to MLflow"
    )
    evaluate.add_argument(
        "--brute-force",
        action="store_true",
        help=(
            "also measure retrieval vs. a brute-force (index-disabled) scan; off by default "
            "because it disables index scans over the whole corpus and takes minutes at full "
            "scale — Task 11's own script measures this comparison"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parser().parse_args(argv)
    load_dotenv()  # before any settings are read
    commands = {"build": _build, "embed": _embed, "detect": _detect, "evaluate": _evaluate}
    try:
        return commands[args.command](args)
    except Exception as exc:  # noqa: BLE001 — CLI boundary: report and exit 1
        print(f"{args.command} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _build(args) -> int:
    reserved = CorpusConfig.n_exact_repost + CorpusConfig.n_reworded + CorpusConfig.n_edited_photo
    n_base = args.listings - reserved
    if n_base <= 0:
        raise ValueError(
            f"--listings {args.listings} is too small: {reserved:,} listings are reserved for "
            "planted duplicates (exact reposts + reworded copies + edited-photo copies), "
            f"which leaves n_base={n_base} <= 0. Pass a larger --listings."
        )
    config = dataclasses.replace(
        CorpusConfig(), seed=args.seed, n_listings=args.listings, n_base=n_base
    )
    settings = DbSettings.from_env()
    pool = ensure_pool(config, data_dir=args.data_dir)
    print(f"photo pool: {len(pool.set_ids)} sets under {pool.root}")
    corpus = generate_corpus(
        load_sales(settings, config), load_areas(settings), pool.set_ids, config
    )
    written = render_variants(corpus, pool, config, Path(args.data_dir) / "variants")
    corpus_run_id = load_corpus(settings, corpus, config.seed, pool.archive_sha256)
    print(f"corpus run {corpus_run_id}: {corpus.counts}, {written} edited photos rendered")
    return 0


def _embed(args) -> int:
    settings = DbSettings.from_env()
    device = "cpu" if args.fake else resolve_device(args.device)
    embedder = FakeEmbedder() if args.fake else SentenceTransformerEmbedder(device=device)
    conn = settings.connect()
    try:
        photos, listings, listing_photos = read_corpus_for_embedding(conn)
        if listings.height == 0:
            raise RuntimeError(
                "no corpus has been loaded yet — run `python -m listings build` first"
            )
        print(f"embedding {photos.height} photos and {listings.height} listings on {device}")
        photo_frame, photo_vectors = embed_photos(photos, embedder, args.data_dir)
        listing_frame = embed_listings(listings, listing_photos, photo_vectors, embedder)
        write_photo_embeddings(conn, photo_frame)
        write_listing_embeddings(conn, listing_frame)
        create_vector_indexes(conn)
        conn.commit()
    finally:
        conn.close()
    print("vectors written and HNSW indexes built")
    return 0


def _detect(args) -> int:
    settings, config = DbSettings.from_env(), DetectConfig()
    conn = settings.connect()
    try:
        corpus_run_id = latest_corpus_run(conn)
        result = run_detection(conn, config)
        detect_run_id = write_detection(conn, result, corpus_run_id)
        flagged = result.pairs.filter(result.pairs["decision"])
        fraud = run_fraud_checks(conn, flagged, config)
        write_fraud_flags(conn, fraud, detect_run_id)
        conn.commit()
    finally:
        conn.close()
    if fraud.stats.get("bait_price_skipped"):
        print(
            "WARNING: bait_price check was SKIPPED — the champion price model "
            f"({config.price_model_uri}) could not be loaded (see the 'price model ... "
            "unavailable' warning above for the resolved tracking URI and exception). "
            "Fraud results do not include any bait_price flags for this run.",
            file=sys.stderr,
        )
    print(
        f"detect run {detect_run_id}: {result.pairs.height:,} candidate pairs, "
        f"{flagged.height:,} flagged at threshold {result.threshold:.4f}, "
        f"{fraud.flags.height:,} fraud flags"
    )
    return 0


def _evaluate(args) -> int:
    settings, config = DbSettings.from_env(), DetectConfig()
    conn = settings.connect()
    try:
        corpus_run_id = latest_corpus_run(conn)
        result = run_detection(conn, config)
        flagged = result.pairs.filter(result.pairs["decision"])
        fraud = run_fraud_checks(conn, flagged, config)
        truth_pairs, truth = load_duplicate_truth(conn), load_truth(conn)
        from listings.features import load_listing_attributes

        attributes = load_listing_attributes(conn)
        exact = brute_force_pairs(conn, "text", config.text_top_k) if args.brute_force else None
    finally:
        conn.close()

    metrics = evaluate_detection(result, truth_pairs, attributes, config)
    metrics |= evaluate_fraud(fraud, truth, config)
    if exact is not None:
        metrics["retrieval.exact_pairs"] = float(exact.height)
    if fraud.stats.get("bait_price_skipped"):
        print(
            "WARNING: bait_price check was SKIPPED — the champion price model "
            f"({config.price_model_uri}) could not be loaded (see the 'price model ... "
            "unavailable' warning above for the resolved tracking URI and exception). "
            "fraud.bait_price.* metrics below are computed against zero flags.",
            file=sys.stderr,
        )
    for name, value in sorted(metrics.items()):
        print(f"{name:<45}{value:>12.4f}")

    with tempfile.TemporaryDirectory() as tmp:
        curve = save_pr_curve(
            result.pairs["score"], result.pairs["is_duplicate"], Path(tmp) / "pr.png"
        )
        run_id = log_run(
            metrics,
            {
                "corpus_run_id": corpus_run_id,
                "threshold": result.threshold,
                "text_top_k": config.text_top_k,
                "photo_top_k": config.photo_top_k,
                "max_photo_fanout": config.max_photo_fanout,
                "target_precision": config.target_precision,
            },
            [curve],
            config,
        )
    print(f"logged MLflow run {run_id} in experiment {config.experiment}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
