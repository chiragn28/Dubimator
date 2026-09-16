import dataclasses

import numpy as np
import polars as pl
import pytest
from search_fixtures import N_CLONES

import search.engine as engine_module
from search.config import FEATURES, SearchConfig
from search.engine import FALLBACK, SearchEngine, query_notes, reasons
from search.parse import ParsedQuery
from search.ranker import XGBRanker
from search.train import register_ranker

CONFIG = dataclasses.replace(SearchConfig(), ef_search=100, device="cpu")
NAN = float("nan")


class LargestFirst:
    kind = "test"

    def score(self, frame):
        # size dominates; the fusion score (at most ~0.033) only breaks exact size ties
        return frame["size_ratio"].fill_nan(0.0).to_numpy() * 1000 + frame["rrf_score"].to_numpy()


def _row(**overrides):
    row = {name: NAN for name in FEATURES}
    row.update(
        area_match=1.0, type_match=0.0, building_match=NAN, beds_diff=1.0, bedrooms=3,
        price_over_max=0.043, price_under_min=NAN, size_ratio=0.9,
        description="Features include balcony.", price_to_estimate=0.8,
        flag_bait_price=1.0, flag_photo_reuse=0.0, flag_inconsistent_relist=0.0,
    )  # fmt: skip
    row.update(overrides)
    return row


def test_reasons_cover_every_stated_slot_in_order():
    parsed = ParsedQuery(
        area_ids=(1,), area_name="Dubai Marina", property_type="flat", bedrooms=2,
        budget_max=1e6, min_size_sqm=100.0, amenities=("balcony", "shared pool"),
    )  # fmt: skip
    assert reasons(parsed, _row()) == (
        "area ✓",
        "different type",
        "3 bedrooms (asked 2)",
        "4% over budget",
        "10% smaller than asked",
        "balcony ✓",
        "no shared pool",
        "priced 20% below estimate",
        "flagged: bait price",
    )


def test_reasons_for_exact_matches_and_quiet_values():
    parsed = ParsedQuery(bedrooms=0, budget_max=1e6, building="Marina Gate", min_size_sqm=50.0)
    row = _row(
        beds_diff=0.0, bedrooms=0, price_over_max=-0.2, size_ratio=1.5, building_match=1.0,
        price_to_estimate=1.02, flag_bait_price=0.0, flag_inconsistent_relist=1.0,
    )  # fmt: skip
    assert reasons(parsed, row) == (
        "Marina Gate ✓",
        "studio ✓",
        "within budget",
        "size ✓",
        "flagged: inconsistent relist",
    )
    # _row()'s own baseline sets price_to_estimate=0.8 and flag_bait_price=1.0 (used deliberately
    # by other assertions in this file), so a query that only asks about bedrooms still picks up
    # the unconditional Value and Flags reasons; check index 0 like the two asserts below it.
    assert reasons(ParsedQuery(bedrooms=1), _row(beds_diff=0.0, bedrooms=1))[0] == "1 bedroom ✓"
    assert reasons(ParsedQuery(bedrooms=2), _row(beds_diff=NAN, bedrooms=None))[0] == (
        "bedrooms not listed"
    )
    under = reasons(ParsedQuery(budget_min=2e6), _row(price_under_min=0.25))
    assert under[0] == "25% under your minimum"


def test_query_notes():
    parsed = ParsedQuery(
        unrecognised=(("place", "al barsha"), ("type", "land")),
        errors=("budget_min_exceeds_max",),
    )
    assert query_notes(parsed) == [
        "area not recognised: al barsha",
        "property type not listed: land",
        "the minimum budget is above the maximum, so the budget was ignored",
    ]


def _engine(settings, embedder, ranker, config=CONFIG):
    conn = settings.connect()
    return conn, SearchEngine(conn, embedder, config, ranker=ranker)


def test_fallback_keeps_retrieval_order_and_explains_itself(search_db, fake_embedder):
    settings, listings, _ = search_db
    conn, engine = _engine(settings, fake_embedder, None)
    try:
        result = engine.search("2 bed apartment in Dubai Marina with balcony", k=5)
    finally:
        conn.close()
    assert result.ranker == FALLBACK == engine.ranker_label
    assert "no ranking model is registered; showing retrieval order" in result.notes
    assert 0 < len(result.results) <= 5
    area_one = set(listings.filter(pl.col("area_id") == 1)["listing_id"])
    assert all(hit.listing_id in area_one for hit in result.results)
    assert all(hit.area_name == "Dubai Marina" and hit.title for hit in result.results)
    assert all(hit.reasons[:2] == ("area ✓", "type ✓") for hit in result.results)
    scores = [hit.score for hit in result.results]
    assert scores == sorted(scores, reverse=True)
    assert result.parsed.bedrooms == 2
    assert set(result.timings_ms) == {"parse", "retrieve", "rank", "total"}


def test_an_injected_ranker_orders_the_results(search_db, fake_embedder):
    settings, _, _ = search_db
    conn, engine = _engine(settings, fake_embedder, LargestFirst())
    try:
        result = engine.search("apartment over 60 sqm", k=8)
    finally:
        conn.close()
    assert result.ranker == "test (injected)"
    sizes = [hit.size_sqm for hit in result.results]
    assert sizes == sorted(sizes, reverse=True)


def test_duplicates_are_hidden_and_counted(search_db, fake_embedder):
    settings, listings, _ = search_db
    conn, engine = _engine(settings, fake_embedder, None)
    source = listings.filter(pl.col("listing_id") == 1).row(0, named=True)
    try:
        result = engine.search(source["title"], k=50)
    finally:
        conn.close()
    ids = [hit.listing_id for hit in result.results]
    clone_id = listings.height - N_CLONES + 1
    assert 1 in ids and clone_id not in ids
    assert next(hit for hit in result.results if hit.listing_id == 1).duplicates_hidden == 1


@pytest.mark.parametrize(
    ("text", "note"),
    [
        ("under 5", "empty query: add an area, a budget, a property type or a few words"),
        (
            "villa in Dubai Marina",
            "no listings match; try widening the budget or removing a filter",
        ),
        ("flat in Al Barsha", "area not recognised: al barsha"),
    ],
)
def test_notes_for_unhelpful_queries(search_db, fake_embedder, text, note):
    settings, _, _ = search_db
    conn, engine = _engine(settings, fake_embedder, LargestFirst())
    try:
        result = engine.search(text)
    finally:
        conn.close()
    assert note in result.notes
    if note.startswith(("empty", "no listings")):
        assert result.results == ()


def test_the_registered_champion_is_loaded_and_labelled(
    search_db, fake_embedder, temp_mlflow, tmp_path
):
    import mlflow

    rng = np.random.default_rng(0)
    rows = []
    for query_id in range(1, 31):
        for index in range(8):
            features = {name: float(rng.random()) for name in FEATURES}
            rows.append({"query_id": query_id, "grade": int(features["size_ratio"] * 3),
                         "fused_pos": index + 1, **features})  # fmt: skip
    frame = pl.DataFrame(rows)
    config = dataclasses.replace(
        CONFIG, ranker_name="search-engine-test",
        ranker_uri="models:/search-engine-test@champion", max_rounds=20, early_stopping_rounds=5,
    )  # fmt: skip
    ranker = XGBRanker.fit(frame, frame, FEATURES, {"max_depth": 2}, config)
    mlflow.create_experiment("engine-test", artifact_location=temp_mlflow["artifact_location"])
    mlflow.set_experiment("engine-test")
    with mlflow.start_run():
        register_ranker(ranker, config, tmp_path / "ranker")
    settings, _, _ = search_db
    conn = settings.connect()
    try:
        engine = SearchEngine(conn, fake_embedder, config)
        result = engine.search("2 bed apartment in Dubai Marina")
    finally:
        conn.close()
    assert engine.ranker_label == "search-engine-test/v1" == result.ranker
    assert result.results and FALLBACK not in result.ranker


def test_module_search_reuses_one_engine_per_connection(search_db, monkeypatch):
    from listings.embed import FakeEmbedder

    settings, _, _ = search_db
    built = []

    class CountingEngine(SearchEngine):
        def __init__(self, conn, embedder, config=SearchConfig(), ranker=None):  # noqa: B008
            built.append(type(embedder).__name__)
            super().__init__(conn, embedder, CONFIG, ranker=None)

    monkeypatch.setattr(engine_module, "SearchEngine", CountingEngine)
    monkeypatch.setattr(engine_module, "SentenceTransformerEmbedder", lambda device: FakeEmbedder())
    monkeypatch.setattr(engine_module, "_ENGINES", {})
    conn = settings.connect()
    try:
        first = engine_module.search(conn, "apartment in JVC")
        second = engine_module.search(conn, "villa in Arabian Ranches", k=3)
    finally:
        conn.close()
    assert built == ["FakeEmbedder"]
    assert first.results and len(second.results) <= 3
