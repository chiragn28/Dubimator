import dataclasses
from pathlib import Path

import mlflow
import polars as pl

from ingestion.pipeline import run_pipeline
from listings.config import CorpusConfig, DetectConfig
from listings.detect import run_detection, write_detection
from listings.embed import FakeEmbedder, embed_listings, embed_photos
from listings.evaluate import evaluate_detection, evaluate_fraud, log_run, save_pr_curve
from listings.fraud import run_fraud_checks, write_fraud_flags
from listings.generate import generate_corpus, load_areas, load_sales, render_variants
from listings.load import (
    create_vector_indexes,
    load_corpus,
    read_corpus_for_embedding,
    write_listing_embeddings,
    write_photo_embeddings,
)
from listings.truth import load_duplicate_truth, load_truth

DLD_FIXTURE = Path("tests/fixtures/price_sample.csv")
SMALL_CORPUS = CorpusConfig(
    n_listings=400,
    n_base=320,
    n_exact_repost=40,
    n_reworded=20,
    n_edited_photo=20,
    n_bait_price=20,
    n_price_shifted_reposts=10,
    n_from_busy_buildings=40,
    n_stock_sets=2,
    n_agents=30,
    # NOT 2 as elsewhere in this phase's SMALL configs: generate._plain_sets_by_area assigns
    # every non-stock set to 1 OR 2 areas, so stock_min_areas must leave room for a
    # legitimately-local set to touch 2 areas without tripping the "stays under
    # stock_min_areas" citywide check. 3 is what small_corpus_config (tests/listings/conftest.py)
    # already uses at this scale; verified empirically that 2 fails deterministically against
    # the real, ~49-68-area-diverse DLD fixture (tests/fixtures/price_sample.csv) no matter how
    # large the local photo pool is, while 3 passes cleanly.
    stock_min_areas=3,
)
SMALL_DETECT = dataclasses.replace(DetectConfig(), text_top_k=10, photo_top_k=10)


def test_the_whole_pipeline_runs_on_real_dld_rows(
    pg_test_db, temp_mlflow, large_photo_pool, tmp_path
):
    run_pipeline(DLD_FIXTURE, pg_test_db)  # real DLD rows -> dld.market_sales

    sales = load_sales(pg_test_db, SMALL_CORPUS)
    areas = load_areas(pg_test_db)
    assert sales.height > SMALL_CORPUS.n_base, "fixture should hold enough home sales"

    photo_pool = large_photo_pool
    data_dir = photo_pool.root.parent
    corpus = generate_corpus(sales, areas, photo_pool.set_ids, SMALL_CORPUS)
    render_variants(corpus, photo_pool, SMALL_CORPUS, data_dir / "variants")
    corpus_run_id = load_corpus(pg_test_db, corpus, SMALL_CORPUS.seed, photo_pool.archive_sha256)

    conn = pg_test_db.connect()
    try:
        photos, listings, listing_photos = read_corpus_for_embedding(conn)
        embedder = FakeEmbedder()
        photo_frame, photo_vectors = embed_photos(photos, embedder, data_dir)
        listing_frame = embed_listings(listings, listing_photos, photo_vectors, embedder)
        write_photo_embeddings(conn, photo_frame)
        write_listing_embeddings(conn, listing_frame)
        create_vector_indexes(conn)
        conn.commit()

        result = run_detection(conn, SMALL_DETECT)
        detect_run_id = write_detection(conn, result, corpus_run_id)
        fraud = run_fraud_checks(
            conn, result.pairs.filter(pl.col("decision")), SMALL_DETECT, predictor=None
        )
        write_fraud_flags(conn, fraud, detect_run_id)
        conn.commit()

        truth_pairs = load_duplicate_truth(conn)
        truth = load_truth(conn)
        from listings.features import load_listing_attributes

        attributes = load_listing_attributes(conn)
    finally:
        conn.close()

    metrics = evaluate_detection(result, truth_pairs, attributes, SMALL_DETECT)
    metrics |= evaluate_fraud(fraud, truth, SMALL_DETECT)

    assert metrics["retrieval.recall"] >= 0.9, metrics["retrieval.recall"]
    assert metrics["report.recall"] > 0.0
    assert metrics["report.precision"] >= 0.9, metrics["report.precision"]
    assert metrics["pattern.exact_repost.recall"] >= 0.9
    assert metrics["control.same_building.model_fp_rate"] <= metrics.get(
        "control.same_building.baseline_fp_rate", 1.0
    )
    assert metrics["stats.bait_price_skipped"] == 1.0  # no price model in the test environment

    curve = save_pr_curve(result.pairs["score"], result.pairs["is_duplicate"], tmp_path / "pr.png")
    run_id = log_run(
        metrics,
        {"corpus_run_id": corpus_run_id, "threshold": result.threshold},
        [curve],
        SMALL_DETECT,
        tracking_uri=temp_mlflow["tracking_uri"],
        artifact_location=temp_mlflow["artifact_location"],
    )
    assert mlflow.get_run(run_id).data.metrics["retrieval.recall"] >= 0.9

    conn = pg_test_db.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM listings.duplicate_pairs WHERE detect_run_id=%s",
                (detect_run_id,),
            )
            flagged = cur.fetchone()[0]
            cur.execute(
                "SELECT count(*) FROM listings.fraud_flags WHERE detect_run_id=%s", (detect_run_id,)
            )
            flags = cur.fetchone()[0]
    finally:
        conn.close()
    assert flagged == result.pairs.filter(pl.col("decision")).height > 0
    assert flags == fraud.flags.height


def test_only_truth_py_reads_label_columns():
    """Every other module's SQL must select listing content, never ground truth."""
    import inspect

    from listings import candidates, detect, features, fraud, load, truth

    for module in (candidates, detect, features, fraud, load):
        source = inspect.getsource(module)
        for label in ("dup_group_id", "control_group_id", "fraud_label"):
            statements = [
                line
                for line in source.splitlines()
                if label in line
                and "SELECT" in source[: source.index(line)].split("\n")[-1].upper()
            ]
            assert not statements, f"{module.__name__} selects {label}: {statements}"
        # a blunt second check: the label names must not appear in any SQL constant
        for name, value in vars(module).items():
            if name.endswith("SQL") and isinstance(value, str):
                for label in ("dup_group_id", "control_group_id", "fraud_label"):
                    assert label not in value, f"{module.__name__}.{name} mentions {label}"
    assert "dup_group_id" in truth.TRUTH_SQL  # truth.py is the one place they belong


def test_cli_reports_failures_and_exits_1(monkeypatch, capsys):
    from listings import __main__ as cli

    monkeypatch.setattr(cli, "load_dotenv", lambda: None)

    def boom(_args):
        raise RuntimeError("database unreachable")

    monkeypatch.setattr(cli, "_detect", boom)
    assert cli.main(["detect"]) == 1
    assert "detect failed: RuntimeError: database unreachable" in capsys.readouterr().err


def test_cli_rejects_an_unknown_command():
    import pytest

    from listings import __main__ as cli

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["nonsense"])
    assert exit_info.value.code == 2
