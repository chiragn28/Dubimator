import dataclasses

import polars as pl

from listings.candidates import brute_force_pairs, fetch_candidates
from listings.config import DetectConfig

CONFIG = DetectConfig()


def candidates_for(settings, config=CONFIG):
    conn = settings.connect()
    try:
        return fetch_candidates(conn, config)
    finally:
        conn.close()


def test_pairs_are_canonical_unique_and_never_self(loaded_corpus):
    settings, corpus, _ = loaded_corpus
    pairs, stats = candidates_for(settings)

    assert pairs.height > 0
    assert (pairs["listing_a"] < pairs["listing_b"]).all()
    assert pairs.select("listing_a", "listing_b").is_duplicated().sum() == 0
    known = set(corpus.listings["listing_id"].to_list())
    assert set(pairs["listing_a"].to_list()) <= known
    assert set(pairs["listing_b"].to_list()) <= known
    assert stats.total == pairs.height
    assert stats.seconds >= 0.0
    assert {"text", "image", "shared_photo"} >= {
        source for row in pairs["sources"].to_list() for source in row.split(",")
    }


def test_planted_duplicates_are_retrieved(loaded_corpus):
    settings, corpus, _ = loaded_corpus
    pairs, _ = candidates_for(settings)
    found = {(a, b) for a, b in zip(pairs["listing_a"].to_list(), pairs["listing_b"].to_list())}

    clones = corpus.listings.filter(pl.col("dup_group_id").is_not_null())
    truth = {
        (min(row["listing_id"], row["dup_group_id"]), max(row["listing_id"], row["dup_group_id"]))
        for row in clones.iter_rows(named=True)
    }
    recall = len(truth & found) / len(truth)
    assert recall >= 0.95, (
        f"retrieval recall {recall:.3f} — duplicates missed here can never be caught"
    )


def test_stock_photos_do_not_explode_the_shared_photo_channel(loaded_corpus):
    settings, _corpus, _ = loaded_corpus
    tight = dataclasses.replace(CONFIG, max_photo_fanout=3)
    pairs, stats = candidates_for(settings, tight)
    _, wide_stats = candidates_for(settings, dataclasses.replace(CONFIG, max_photo_fanout=10_000))
    assert stats.shared_photo_pairs < wide_stats.shared_photo_pairs
    assert pairs.height <= wide_stats.total


def test_index_retrieval_agrees_with_an_exact_scan(loaded_corpus):
    settings, _, _ = loaded_corpus
    conn = settings.connect()
    try:
        approximate, _ = fetch_candidates(conn, CONFIG)
        exact = brute_force_pairs(conn, "text", CONFIG.text_top_k)
    finally:
        conn.close()
    found = {(a, b) for a, b in zip(approximate["listing_a"], approximate["listing_b"])}
    truth = {(a, b) for a, b in zip(exact["listing_a"], exact["listing_b"])}
    recall = len(truth & found) / len(truth)
    assert recall >= 0.95, f"HNSW recall against an exact scan was {recall:.3f}"


def test_a_corpus_without_vectors_yields_no_pairs(
    pg_test_db, sales_frame, areas_frame, small_corpus_config
):
    from listings.generate import generate_corpus
    from listings.load import load_corpus

    corpus = generate_corpus(sales_frame(), areas_frame, tuple(range(1, 21)), small_corpus_config)
    load_corpus(pg_test_db, corpus, seed=1, photo_dataset_sha="x")  # no embeddings written
    conn = pg_test_db.connect()
    try:
        pairs, stats = fetch_candidates(conn, dataclasses.replace(CONFIG, max_photo_fanout=0))
    finally:
        conn.close()
    assert pairs.height == 0 and stats.total == 0


def test_benchmark_compares_the_text_index_with_the_exact_text_scan(loaded_corpus):
    """`evaluate --brute-force` compares like with like: the indexed text channel against the
    exact (index-disabled) text scan, with both timed and the overlap measured."""
    from listings.candidates import benchmark_text_retrieval

    settings, _, _ = loaded_corpus
    conn = settings.connect()
    try:
        pairs, _ = fetch_candidates(conn, CONFIG)
        bench = benchmark_text_retrieval(conn, CONFIG, candidates=pairs)
        exact = brute_force_pairs(conn, "text", CONFIG.text_top_k)
    finally:
        conn.close()

    assert bench["retrieval.bench.text_index_seconds"] > 0.0
    assert bench["retrieval.bench.text_exact_seconds"] > 0.0
    assert bench["retrieval.bench.text_exact_pairs"] == float(exact.height)
    assert bench["retrieval.bench.text_index_pairs"] > 0.0
    assert 0.95 <= bench["retrieval.bench.text_index_recall"] <= 1.0
    # all three channels together contain at least what the text channel alone found
    assert (
        bench["retrieval.bench.candidates_exact_text_recall"]
        >= bench["retrieval.bench.text_index_recall"]
    )
