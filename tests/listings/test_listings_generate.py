import polars as pl
import pytest
from PIL import Image

from listings.generate import generate_corpus, render_variants

POOL = tuple(range(1, 21))


def build(sales_frame, areas_frame, config):
    return generate_corpus(sales_frame(), areas_frame, POOL, config)


def test_corpus_is_deterministic(sales_frame, areas_frame, small_corpus_config):
    first = build(sales_frame, areas_frame, small_corpus_config)
    second = build(sales_frame, areas_frame, small_corpus_config)
    assert first.listings.equals(second.listings)
    assert first.listing_photos.equals(second.listing_photos)
    assert first.photos.equals(second.photos)


def test_counts_match_the_config(sales_frame, areas_frame, small_corpus_config):
    config = small_corpus_config
    corpus = build(sales_frame, areas_frame, config)
    assert corpus.listings.height == config.n_listings
    assert corpus.counts["base"] == config.n_base
    assert corpus.counts["exact_repost"] == config.n_exact_repost
    assert corpus.counts["reworded"] == config.n_reworded
    assert corpus.counts["edited_photo"] == config.n_edited_photo
    assert corpus.counts["bait_price"] == config.n_bait_price
    assert corpus.listings["listing_id"].n_unique() == config.n_listings
    assert corpus.listings["is_synthetic"].all()


def test_every_listing_has_four_photos_in_order(sales_frame, areas_frame, small_corpus_config):
    corpus = build(sales_frame, areas_frame, small_corpus_config)
    per_listing = corpus.listing_photos.group_by("listing_id").agg(
        pl.col("position").sort().alias("positions")
    )
    assert per_listing.height == corpus.listings.height
    assert per_listing["positions"].to_list() == [[0, 1, 2, 3]] * per_listing.height
    known = set(corpus.photos["photo_id"].to_list())
    assert set(corpus.listing_photos["photo_id"].to_list()) <= known


def clones_with_sources(corpus):
    listings = corpus.listings
    clones = listings.filter(pl.col("dup_group_id").is_not_null())
    sources = listings.rename({c: f"src_{c}" for c in listings.columns})
    return clones.join(sources, left_on="dup_group_id", right_on="src_listing_id", how="inner")


def photo_ids(corpus, listing_id):
    rows = corpus.listing_photos.filter(pl.col("listing_id") == listing_id).sort("position")
    return rows["photo_id"].to_list()


def test_clones_point_at_their_source_and_post_later(sales_frame, areas_frame, small_corpus_config):
    config = small_corpus_config
    corpus = build(sales_frame, areas_frame, config)
    joined = clones_with_sources(corpus)
    assert joined.height == config.n_exact_repost + config.n_reworded + config.n_edited_photo
    assert (joined["posted_at"] > joined["src_posted_at"]).all()
    assert (joined["agent_id"] != joined["src_agent_id"]).all()
    assert (joined["source_transaction_id"] == joined["src_source_transaction_id"]).all()
    assert joined["dup_group_id"].is_in(corpus.listings["listing_id"]).all()


def test_the_three_clone_patterns_are_present_and_distinguishable(
    sales_frame, areas_frame, small_corpus_config
):
    config = small_corpus_config
    corpus = build(sales_frame, areas_frame, config)
    variants = dict(zip(corpus.photos["photo_id"].to_list(), corpus.photos["variant_of"].to_list()))
    exact = reworded = edited = 0
    for clone in clones_with_sources(corpus).iter_rows(named=True):
        mine = photo_ids(corpus, clone["listing_id"])
        theirs = photo_ids(corpus, clone["dup_group_id"])
        if all(variants[p] is not None for p in mine):
            assert [variants[p] for p in mine] == theirs  # variants of the source's photos
            assert clone["description"] == clone["src_description"]
            edited += 1
        elif mine == theirs and clone["description"] == clone["src_description"]:
            exact += 1
        else:
            assert mine == theirs and clone["description"] != clone["src_description"]
            reworded += 1
    assert (exact, reworded, edited) == (
        config.n_exact_repost, config.n_reworded, config.n_edited_photo,
    )  # fmt: skip


def test_some_reposts_shift_the_price_and_the_rest_keep_it(
    sales_frame, areas_frame, small_corpus_config
):
    config = small_corpus_config
    joined = clones_with_sources(build(sales_frame, areas_frame, config))
    same = joined.filter(pl.col("asking_price_aed") == pl.col("src_asking_price_aed"))
    shifted = joined.filter(pl.col("asking_price_aed") != pl.col("src_asking_price_aed"))
    assert shifted.height == config.n_price_shifted_reposts
    assert same.height == joined.height - config.n_price_shifted_reposts
    ratio = (shifted["asking_price_aed"] / shifted["src_asking_price_aed"]).to_list()
    # The shift is drawn in +/-15-30% and only then rounded to the nearest 10,000, so the
    # realised ratio can sit up to one rounding step outside the drawn band.
    tolerance = [10_000 / price for price in shifted["src_asking_price_aed"].to_list()]
    assert all(0.70 - slack <= value <= 1.30 + slack for value, slack in zip(ratio, tolerance))
    assert all(abs(value - 1.0) >= 0.10 for value in ratio)


def test_same_building_controls_are_labelled_in_groups(
    sales_frame, areas_frame, small_corpus_config
):
    config = small_corpus_config
    listings = build(sales_frame, areas_frame, config).listings
    controls = listings.filter(pl.col("control_group_id").is_not_null())
    assert controls.height >= config.n_from_busy_buildings
    groups = controls.group_by("control_group_id").agg(
        pl.col("building_name").n_unique().alias("buildings"),
        pl.col("source_transaction_id").n_unique().alias("sales"),
        pl.len().alias("members"),
    )
    assert (groups["members"] >= 2).all()
    assert (groups["buildings"] == 1).all()
    assert (groups["sales"] == groups["members"]).all()  # different units, not the same sale


def test_stock_photo_sets_span_several_areas(sales_frame, areas_frame, small_corpus_config):
    config = small_corpus_config
    corpus = build(sales_frame, areas_frame, config)
    stock_sets = set(corpus.photos.filter(pl.col("is_stock"))["set_id"].to_list())
    assert len(stock_sets) == config.n_stock_sets
    spread = (
        corpus.listings.filter(pl.col("photo_set_id").is_in(list(stock_sets)))
        .group_by("photo_set_id")
        .agg(pl.col("area_id").n_unique().alias("areas"))
    )
    assert (spread["areas"] >= config.stock_min_areas).all()


def test_bait_listings_are_cheap_and_labelled(sales_frame, areas_frame, small_corpus_config):
    config = small_corpus_config
    corpus = build(sales_frame, areas_frame, config)
    listings = corpus.listings
    bait = listings.filter(pl.col("fraud_label") == "bait_price")
    base_bait = bait.filter(pl.col("dup_group_id").is_null())
    assert base_bait.height == config.n_bait_price  # bait is planted on base listings
    assert corpus.counts["bait_price"] == config.n_bait_price
    normal = listings.filter(pl.col("fraud_label").is_null())
    assert bait["asking_price_aed"].median() < normal["asking_price_aed"].median()

    # A repost of a bait listing is bait too: the label rides along with the cheap price.
    # Counted from the sources, so the check cannot pass by every clone simply being unlabelled.
    base_bait_ids = set(base_bait["listing_id"].to_list())
    clones = listings.filter(pl.col("dup_group_id").is_not_null())
    expected = sum(1 for source in clones["dup_group_id"].to_list() if source in base_bait_ids)
    assert expected > 0  # the fixture must exercise this path at all
    clone_bait = bait.filter(pl.col("dup_group_id").is_not_null())
    assert clone_bait.height == expected
    assert corpus.counts["bait_price_clones"] == expected
    assert set(clone_bait["dup_group_id"].to_list()) <= base_bait_ids

    # ...and it keeps the source's price unless it is one of the price-shifted reposts, in
    # which case the only permitted departure is a shift inside the configured band.
    joined = clones_with_sources(corpus).filter(pl.col("fraud_label") == "bait_price")
    moved = 0
    for row in joined.iter_rows(named=True):
        if row["asking_price_aed"] == row["src_asking_price_aed"]:
            continue
        ratio = row["asking_price_aed"] / row["src_asking_price_aed"]
        slack = 10_000 / row["src_asking_price_aed"]
        assert 0.70 - slack <= ratio <= 1.30 + slack
        assert abs(ratio - 1.0) >= 0.10
        moved += 1
    assert moved <= config.n_price_shifted_reposts


def test_asking_prices_are_rounded(sales_frame, areas_frame, small_corpus_config):
    listings = build(sales_frame, areas_frame, small_corpus_config).listings
    assert (listings["asking_price_aed"] % 10_000 == 0).all()
    assert (listings["asking_price_aed"] > 0).all()


def test_generation_fails_loudly_when_there_are_too_few_sales(
    sales_frame, areas_frame, small_corpus_config
):
    with pytest.raises(ValueError, match="not enough sales"):
        generate_corpus(sales_frame(n=100), areas_frame, POOL, small_corpus_config)


def test_render_variants_writes_one_file_per_variant_row(
    sales_frame, areas_frame, small_corpus_config, photo_pool, tmp_path
):
    config = small_corpus_config
    corpus = generate_corpus(sales_frame(), areas_frame, photo_pool.set_ids, config)
    written = render_variants(corpus, photo_pool, config, tmp_path / "variants")
    assert written == config.n_edited_photo * 4
    for path in corpus.photos.filter(pl.col("variant_kind") == "edited")["path"].to_list():
        target = tmp_path / "variants" / path.split("/")[-1]
        assert target.is_file()
        with Image.open(target) as image:
            assert image.width > 0
