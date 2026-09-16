import dataclasses
from datetime import date

import polars as pl
import pytest
from search_fixtures import N_CLONES

from search.config import SearchConfig, listing_kind
from search.lexicon import load_lexicon
from search.parse import ParsedQuery, parse
from search.queries import build_query_set
from search.retrieve import (
    CANDIDATE_SCHEMA,
    DuplicateClusters,
    collapse,
    configure_session,
    fulltext_terms,
    fuse,
    load_clusters,
    retrieve,
)
from search.store import read_judgments, read_queries, read_query_labels, replace_query_set

CONFIG = dataclasses.replace(SearchConfig(), ef_search=100)


def test_fuse_adds_reciprocal_ranks_and_breaks_ties_by_id():
    fused = fuse([(1, 0.9), (2, 0.8)], [(2, 0.5), (3, 0.4)], rrf_k=60)
    assert fused["listing_id"].to_list() == [2, 1, 3]
    two, one, three = fused.to_dicts()
    assert two["rrf_score"] == pytest.approx(1 / 62 + 1 / 61)
    assert (two["semantic_pos"], two["fulltext_pos"]) == (2, 1)
    assert one["fulltext_rank"] == 0.0 and one["fulltext_pos"] is None
    assert three["semantic_cos"] is None and three["semantic_pos"] is None
    tied = fuse([(9, 0.5)], [(4, 0.5)], rrf_k=60)
    assert tied["listing_id"].to_list() == [4, 9]


def _clusters():
    return DuplicateClusters(
        cluster_of={1: 1, 231: 1, 7: 7, 8: 7},
        size={1: 2, 7: 2},
        posted={
            1: date(2023, 1, 1),
            231: date(2023, 1, 6),
            7: date(2023, 2, 1),
            8: date(2023, 1, 5),
        },
    )


def test_collapse_keeps_the_earliest_retrieved_member_in_its_own_position():
    fused = fuse([(231, 0.9), (1, 0.8), (5, 0.7), (7, 0.6), (8, 0.5)], [], rrf_k=60)
    out = collapse(fused, _clusters(), k=10)
    assert out["listing_id"].to_list() == [1, 5, 8]
    assert out["fused_pos"].to_list() == [1, 2, 3]
    assert out["cluster_size"].to_list() == [2, 1, 2]
    assert out.schema == pl.Schema(CANDIDATE_SCHEMA)


def test_collapse_keeps_a_lone_retrieved_clone_and_truncates_after_collapsing():
    fused = fuse([(231, 0.9), (5, 0.8), (6, 0.7), (7, 0.6), (8, 0.5)], [], rrf_k=60)
    out = collapse(fused, _clusters(), k=3)
    assert out["listing_id"].to_list() == [231, 5, 6]
    assert _clusters().cluster_size(999) == 1


def test_fulltext_terms_prefer_free_text_and_amenities():
    parsed = ParsedQuery(free_text="quiet family", amenities=("sea view",))
    assert fulltext_terms(parsed, "ignored") == "quiet or family or sea or view"
    assert fulltext_terms(ParsedQuery(bedrooms=2), "2BR in Dubai Marina") == (
        "2br or in or dubai or marina"
    )


def test_clusters_come_from_the_latest_detect_run(search_db):
    settings, listings, _ = search_db
    conn = settings.connect()
    try:
        clusters = load_clusters(conn)
    finally:
        conn.close()
    n = listings.height
    assert len(clusters.size) == N_CLONES
    assert clusters.cluster_size(1) == 2 and clusters.cluster_size(n) == 2
    assert clusters.cluster_of[n - N_CLONES + 1] == clusters.cluster_of[1]
    assert clusters.cluster_size(100) == 1


def test_missing_detect_run_fails_loudly(search_db):
    settings, _, _ = search_db
    conn = settings.connect()
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE listings.detect_runs CASCADE")
        with pytest.raises(RuntimeError, match="python -m listings detect"):
            load_clusters(conn)
    finally:
        conn.close()


def _retrieve(settings, text, embedder, config=CONFIG):
    conn = settings.connect()
    try:
        lexicon, clusters = load_lexicon(conn), load_clusters(conn)
        configure_session(conn, config)
        parsed = parse(text, lexicon)
        vector = embedder.embed_texts([text])[0]
        return parsed, retrieve(conn, parsed, text, vector, clusters, config)
    finally:
        conn.close()


@pytest.mark.parametrize("autocommit", [False, True])
def test_configure_session_sets_the_hnsw_settings_for_the_session(search_db, autocommit):
    settings, _, _ = search_db
    conn = settings.connect()
    try:
        conn.autocommit = autocommit
        configure_session(conn, CONFIG)
        conn.commit()
        with conn.cursor() as cur:
            cur.execute("SHOW hnsw.ef_search")
            ef_search = cur.fetchone()[0]
            cur.execute("SHOW hnsw.iterative_scan")
            scan = cur.fetchone()[0]
        conn.rollback()  # a committed session-level SET outlives a later rollback
        with conn.cursor() as cur:
            cur.execute("SHOW hnsw.ef_search")
            again = cur.fetchone()[0]
    finally:
        conn.close()
    assert ef_search == again == str(CONFIG.ef_search) and scan == "relaxed_order"


def test_a_building_name_is_not_an_area_filter(search_db, fake_embedder):
    settings, listings, _ = search_db
    parsed, out = _retrieve(settings, "flat at Marina Gate", fake_embedder)
    assert parsed.building == "Marina Gate" and parsed.area_ids == ()
    assert parsed.building_area_ids == (1,)
    areas = set(out.join(listings.select("listing_id", "area_id"), on="listing_id")["area_id"])
    assert len(areas) > 1  # other areas stay candidates; building_match ranks them


def test_area_and_type_filter_both_channels(search_db, fake_embedder):
    settings, listings, _ = search_db
    _, out = _retrieve(settings, "2 bed apartment in Dubai Marina with balcony", fake_embedder)
    assert out.height > 0
    attributes = listings.select("listing_id", "area_id", "property_type", "property_sub_type")
    joined = out.join(attributes, on="listing_id")
    assert (joined["area_id"] == 1).all()
    kinds = {
        listing_kind(t, s) for t, s in joined.select("property_type", "property_sub_type").rows()
    }
    assert kinds == {"flat"}
    assert out["fused_pos"].to_list() == list(range(1, out.height + 1))
    assert out["fulltext_pos"].null_count() < out.height  # "balcony" hits the full-text channel
    assert out.schema == pl.Schema(CANDIDATE_SCHEMA)


def test_no_listing_matches_an_impossible_filter(search_db, fake_embedder):
    settings, _, _ = search_db
    _, out = _retrieve(settings, "villa in Dubai Marina", fake_embedder)  # no villas there
    assert out.height == 0 and out.schema == pl.Schema(CANDIDATE_SCHEMA)


def test_unfiltered_query_ranks_by_similarity_and_collapses_clusters(search_db, fake_embedder):
    settings, listings, _ = search_db
    wide = dataclasses.replace(CONFIG, semantic_k=500, fulltext_k=500, candidate_k=500)
    parsed, out = _retrieve(settings, "family home with sea view", fake_embedder, wide)
    assert not parsed.area_ids and parsed.property_type is None
    assert out.height >= 150  # most of the 230 distinct listings in the tiny corpus
    semantic = out.filter(pl.col("semantic_pos").is_not_null()).sort("semantic_pos")
    assert semantic["semantic_cos"].is_sorted(descending=True)
    ids = set(out["listing_id"])
    first_clone = listings.height - N_CLONES + 1
    for source in range(1, N_CLONES + 1):
        assert not {source, first_clone + source - 1} <= ids  # never both members of a cluster


def test_candidate_k_caps_the_result(search_db, fake_embedder):
    settings, _, _ = search_db
    config = dataclasses.replace(CONFIG, candidate_k=5)
    _, out = _retrieve(settings, "apartment", fake_embedder, config)
    assert out.height == 5


def test_an_empty_query_retrieves_nothing(search_db, fake_embedder):
    settings, _, _ = search_db
    parsed, out = _retrieve(settings, "under 5", fake_embedder)
    assert parsed.is_empty and out.height == 0


def test_build_query_set_grades_every_candidate(search_db, fake_embedder):
    settings, _, corpus_run_id = search_db
    config = dataclasses.replace(CONFIG, n_queries=60)
    conn = settings.connect()
    try:
        lexicon = load_lexicon(conn)
        queries, judgments = build_query_set(conn, fake_embedder, lexicon, config)
        replace_query_set(conn, queries, judgments)
        conn.commit()
        stored_queries, stored = read_queries(conn), read_judgments(conn)
        labels = read_query_labels(conn)
    finally:
        conn.close()
    assert queries.height == 60 and (queries["corpus_run_id"] == corpus_run_id).all()
    assert stored_queries.height == 60 and stored.height == judgments.height > 0
    assert set(stored["grade"].unique()) <= {0, 1, 2, 3}
    assert set(stored["query_id"]) <= set(stored_queries["query_id"])
    per_query = stored.group_by("query_id").agg(
        pl.col("fused_pos").min().alias("first"),
        pl.len().alias("n"),
        pl.col("fused_pos").max().alias("last"),
        (pl.col("grade") == 3).sum().alias("judged3"),
    )
    assert (per_query["first"] == 1).all() and (per_query["n"] == per_query["last"]).all()
    assert not {"seed_listing_id", "n_grade3"} & set(stored_queries.columns)
    checked = per_query.join(stored_queries, on="query_id").join(labels, on="query_id")
    assert (checked["judged3"] <= checked["n_grade3"]).all()
    no_match = checked.filter(pl.col("kind") == "no_match")
    assert (no_match["judged3"] == 0).all()
