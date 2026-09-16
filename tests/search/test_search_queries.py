import dataclasses
import json
from collections import Counter

import polars as pl
import pytest
from search_fixtures import ALIASES, AREAS, BUILDINGS, build_search_listings

from search.config import QUERY_KINDS, SPLITS, SearchConfig
from search.lexicon import Lexicon
from search.parse import parse
from search.queries import (
    N_TEMPLATES,
    TRUE_SLOT_KEYS,
    assign_template_splits,
    generate_queries,
    grading_frame,
    load_area_aliases,
    load_grading_listings,
    true_slots,
)

GRADING = grading_frame(build_search_listings())
ALIASES_BY_AREA = {}
for alias, area_id in ALIASES:
    ALIASES_BY_AREA.setdefault(area_id, []).append(alias)
LEXICON = Lexicon.from_rows(
    areas=AREAS,
    aliases=ALIASES,
    buildings=[(name, area) for area, names in BUILDINGS.items() for name in names if name],
    projects=[(f"Project {area}", area) for area, _ in AREAS],
)


def _generate(n: int, seed: int = 7) -> pl.DataFrame:
    config = dataclasses.replace(SearchConfig(), n_queries=n, seed=seed)
    return generate_queries(GRADING, ALIASES_BY_AREA, config)


@pytest.fixture(scope="module")
def queries():
    return _generate(900)


def test_templates_split_60_20_20_and_deterministically():
    splits = assign_template_splits(7, (0.6, 0.2, 0.2))
    assert sorted(splits) == list(range(N_TEMPLATES)) and N_TEMPLATES == 30
    assert Counter(splits.values()) == {"train": 18, "tune": 6, "report": 6}
    assert splits == assign_template_splits(7, (0.6, 0.2, 0.2))
    assert splits != assign_template_splits(8, (0.6, 0.2, 0.2))


def test_generation_is_deterministic_under_the_seed():
    first, again, other = _generate(120), _generate(120), _generate(120, seed=8)
    assert first.equals(again)
    assert not first.equals(other)


def test_shape_ids_and_unique_text(queries):
    assert queries.height == 900
    assert queries["query_id"].to_list() == list(range(1, 901))
    assert queries["text"].n_unique() == 900
    assert set(queries["kind"]) == set(QUERY_KINDS) and set(queries["split"]) == set(SPLITS)


def test_kind_shares_are_close_to_config(queries):
    shares = queries["kind"].value_counts(normalize=True)
    expected = dict(zip(QUERY_KINDS, SearchConfig().kind_shares))
    for kind, share in shares.iter_rows():
        assert abs(share - expected[kind]) <= 0.03, kind


def test_no_template_or_text_crosses_splits_and_every_split_has_every_kind(queries):
    per_template = queries.group_by("template_id").agg(pl.col("split").n_unique())
    assert per_template["split"].max() == 1
    per_text = queries.group_by("text").agg(pl.col("split").n_unique())
    assert per_text["split"].max() == 1
    pairs = set(queries.select("split", "kind").unique().iter_rows())
    assert pairs == {(split, kind) for split in SPLITS for kind in QUERY_KINDS}


def test_true_slots_are_json_with_the_fixed_keys(queries):
    for raw in queries["true_slots"].to_list():
        slots = true_slots(raw)
        assert tuple(slots) == TRUE_SLOT_KEYS
        assert isinstance(slots["amenities"], list)
    assert json.loads(queries["true_slots"][0]) == true_slots(queries["true_slots"][0])


def test_answerability_matches_the_kind(queries):
    by_kind = {kind: frame for (kind,), frame in queries.group_by("kind")}
    assert (by_kind["no_match"]["n_grade3"] == 0).all()
    assert (by_kind["specified"]["n_grade3"] >= 1).all()  # the seed listing always qualifies
    assert (by_kind["vague"]["n_grade3"] >= 1).all()


def test_specified_queries_state_at_least_three_slots_and_vague_at_most_two(queries):
    def stated(raw):
        slots = true_slots(raw)
        keys = ("area_id", "property_type", "bedrooms", "min_size_sqm")
        count = sum(slots[key] is not None for key in keys)
        count += slots["budget_max"] is not None
        count += bool(slots["amenities"])
        return count

    counts = queries.with_columns(
        pl.col("true_slots").map_elements(stated, return_dtype=pl.Int64).alias("stated")
    )
    assert counts.filter(pl.col("kind") == "specified")["stated"].min() >= 3
    assert counts.filter(pl.col("kind") == "vague")["stated"].max() <= 2


def test_seed_listings_are_never_fraud_labelled(queries):
    labelled = set(GRADING.filter(pl.col("fraud_label").is_not_null())["listing_id"])
    assert labelled and not labelled & set(queries["seed_listing_id"])


def test_the_parser_reads_generated_queries_back(queries):
    """Generator and parser must agree on the phrasings; the rate is measured, not assumed."""
    checks = Counter()
    for text, raw in queries.select("text", "true_slots").iter_rows():
        slots, parsed = true_slots(raw), parse(text, LEXICON)
        if slots["bedrooms"] is not None:
            checks["bedrooms", parsed.bedrooms == slots["bedrooms"]] += 1
        if slots["area_id"] is not None:
            checks["area", slots["area_id"] in parsed.area_ids] += 1
        if slots["property_type"] is not None:
            checks["type", parsed.property_type == slots["property_type"]] += 1
        if slots["budget_max"] is not None:
            same = (
                parsed.budget_max is not None and abs(parsed.budget_max - slots["budget_max"]) < 1
            )
            checks["budget_max", same] += 1
        if slots["min_size_sqm"] is not None:
            same = (
                parsed.min_size_sqm is not None
                and abs(parsed.min_size_sqm - slots["min_size_sqm"]) < 0.5
            )
            checks["size", same] += 1
        if slots["amenities"]:
            checks["amenities", set(parsed.amenities) == set(slots["amenities"])] += 1
    for slot in ("bedrooms", "area", "type", "budget_max", "size", "amenities"):
        right, wrong = checks[slot, True], checks[slot, False]
        assert right + wrong > 20, slot
        assert right / (right + wrong) >= 0.97, (slot, right, wrong)


def test_loaders_read_postgres(search_db):
    settings, listings, _ = search_db
    conn = settings.connect()
    try:
        grading = load_grading_listings(conn)
        aliases = load_area_aliases(conn)
    finally:
        conn.close()
    assert grading.sort("listing_id").equals(grading_frame(listings).sort("listing_id"))
    assert aliases == {1: ["Marina"], 2: ["JVC"], 4: ["Downtown"], 6: ["The Palm"]}
