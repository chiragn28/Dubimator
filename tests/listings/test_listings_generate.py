import polars as pl
import pytest
from PIL import Image

from listings.generate import LISTING_SCHEMA, generate_corpus, render_variants

POOL = tuple(range(1, 21))

# Rounding to the nearest 10,000 moves a price by at most 5,000, so a realised ratio may sit
# that much either side of the band it was drawn from -- and no further.
ROUNDING_STEP = 5_000


def build(sales_frame, areas_frame, config):
    return generate_corpus(sales_frame(), areas_frame, POOL, config)


def base_with_sale_prices(corpus, sales):
    """Base listings joined back to the sales they were generated from."""
    base = corpus.listings.filter(pl.col("dup_group_id").is_null())
    return base.join(
        sales.select("transaction_id", "price_aed"),
        left_on="source_transaction_id",
        right_on="transaction_id",
        how="inner",
    )


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
    assert joined["dup_group_id"].is_in(corpus.listings["listing_id"].to_list()).all()


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
    tolerance = [ROUNDING_STEP / price for price in shifted["src_asking_price_aed"].to_list()]
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


def _control_pairs(listings: pl.DataFrame) -> pl.DataFrame:
    """Every within-group pair of same-building control listings."""
    controls = listings.filter(pl.col("control_group_id").is_not_null()).select(
        "listing_id", "control_group_id", "asking_price_aed", "photo_set_id"
    )
    return (
        controls.join(controls, on="control_group_id", suffix="_b")
        .filter(pl.col("listing_id") < pl.col("listing_id_b"))
        .with_columns(
            (
                pl.max_horizontal("asking_price_aed", "asking_price_aed_b")
                / pl.min_horizontal("asking_price_aed", "asking_price_aed_b")
                - 1.0
            ).alias("spread"),
            (pl.col("photo_set_id") == pl.col("photo_set_id_b")).alias("shared_set"),
        )
    )


@pytest.mark.parametrize("seed", [42, 7, 1234])
def test_same_building_control_pairs_are_priced_within_the_spread(
    sales_frame, areas_frame, small_corpus_config, seed
):
    """Spec: same-building controls are priced within 20% of each other.

    Otherwise price alone separates them and the control is not the hard negative it claims.
    """
    import dataclasses

    config = dataclasses.replace(small_corpus_config, seed=seed)
    pairs = _control_pairs(build(sales_frame, areas_frame, config).listings)
    assert pairs.height > 0
    assert pairs["spread"].max() <= config.control_price_spread, pairs.sort("spread").tail(3)


@pytest.mark.parametrize("seed", [42, 7, 1234])
def test_half_of_the_control_pairs_share_a_developer_photo_set(
    sales_frame, areas_frame, small_corpus_config, seed
):
    """Spec: half of the same-building controls share the same developer photo set."""
    import dataclasses

    config = dataclasses.replace(small_corpus_config, seed=seed)
    corpus = build(sales_frame, areas_frame, config)
    pairs = _control_pairs(corpus.listings)
    share = pairs["shared_set"].mean()
    assert 0.4 <= share <= 0.6, f"{share:.2f} of {pairs.height} control pairs share a photo set"
    # A developer set is a local (non-stock) set: sharing an agency stock set is the other,
    # cross-area control and must not be what makes these pairs look alike.
    stock = set(corpus.photos.filter(pl.col("is_stock"))["set_id"].to_list())
    shared = pairs.filter(pl.col("shared_set"))
    assert not set(shared["photo_set_id"].to_list()) & stock


def test_bait_is_never_planted_on_a_same_building_control(
    sales_frame, areas_frame, small_corpus_config
):
    """A bait price inside a control group would break the controls' price band."""
    listings = build(sales_frame, areas_frame, small_corpus_config).listings
    bait_controls = listings.filter(
        (pl.col("fraud_label") == "bait_price") & pl.col("control_group_id").is_not_null()
    )
    assert bait_controls.height == 0


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
        slack = ROUNDING_STEP / row["src_asking_price_aed"]
        assert 0.70 - slack <= ratio <= 1.30 + slack
        assert abs(ratio - 1.0) >= 0.10
        moved += 1
    assert moved <= config.n_price_shifted_reposts


def assert_ratio_band(joined, low, high):
    """Every row's asking/sale ratio must sit in [low, high], give or take one rounding step."""
    assert joined.height > 0
    for row in joined.iter_rows(named=True):
        ratio = row["asking_price_aed"] / row["price_aed"]
        slack = ROUNDING_STEP / row["price_aed"]
        assert low - slack <= ratio <= high + slack, (row["listing_id"], ratio)


def test_normal_asking_prices_sit_just_above_the_sale_price(
    sales_frame, areas_frame, small_corpus_config
):
    config = small_corpus_config
    sales = sales_frame()
    corpus = generate_corpus(sales, areas_frame, POOL, config)
    joined = base_with_sale_prices(corpus, sales)
    assert joined.height == config.n_base  # every base listing traced back to its sale
    assert_ratio_band(joined.filter(pl.col("fraud_label").is_null()), *config.asking_factor)


def test_bait_prices_discount_the_sale_price_not_the_asking_price(
    sales_frame, areas_frame, small_corpus_config
):
    """The spec says bait is 0.40-0.65x the DLD SALE price.

    Discounting the asking price instead would compound the two factors into an effective
    0.40-0.70x band, so this asserts against price_aed from the sales fixture.
    """
    config = small_corpus_config
    sales = sales_frame()
    corpus = generate_corpus(sales, areas_frame, POOL, config)
    bait = base_with_sale_prices(corpus, sales).filter(pl.col("fraud_label") == "bait_price")
    assert bait.height == config.n_bait_price
    assert_ratio_band(bait, *config.bait_factor)
    # ...and the upper edge really is 0.65, not the 0.702 that compounding would allow.
    ratios = (bait["asking_price_aed"] / bait["price_aed"]).to_list()
    assert max(ratios) <= config.bait_factor[1] + ROUNDING_STEP / min(bait["price_aed"].to_list())


def test_the_corpus_stores_no_sale_price_and_no_clone_kind():
    """Two 'never stored' invariants that nothing else would catch if they regressed."""
    assert "price_aed" not in LISTING_SCHEMA  # bait must be caught by the model, not a lookup
    assert not [name for name in LISTING_SCHEMA if "sale" in name and "price" in name]
    assert "clone_kind" not in LISTING_SCHEMA  # the clone pattern is derived, never stored
    assert "asking_price_aed" in LISTING_SCHEMA


def test_control_groups_are_spread_across_the_city(sales_frame, areas_frame, small_corpus_config):
    """Hard negatives drawn from three neighbourhoods would generalise to nothing."""
    config = small_corpus_config
    listings = build(sales_frame, areas_frame, config).listings
    controls = listings.filter(pl.col("control_group_id").is_not_null())
    groups = controls["control_group_id"].n_unique()
    spanned = controls["area_id"].n_unique()
    # Expectation derived from the fixture: the groups cannot cover more areas than there are
    # groups, nor more than the fixture has. Half that ceiling is well clear of both ends --
    # seeded-hash order spans 7-11 of 12 across seeds 0-39, while taking the buildings in
    # area/building order (the regression this guards) collapses to about 3.
    ceiling = min(groups, areas_frame.height)
    assert spanned >= 0.5 * ceiling, f"{spanned} areas over {groups} groups (ceiling {ceiling})"


def test_stock_sets_go_citywide_and_plain_sets_stay_local(
    sales_frame, areas_frame, small_corpus_config
):
    """The photo_reuse rule keys on a set spanning >= stock_min_areas areas.

    That only means something if ordinary sets stay below the line; otherwise the rule fires
    on nearly every listing and unrelated listings share byte-identical photos.
    """
    config = small_corpus_config
    corpus = build(sales_frame, areas_frame, config)
    stock_sets = corpus.photos.filter(pl.col("is_stock"))["set_id"].unique().to_list()
    plain_sets = corpus.photos.filter(~pl.col("is_stock"))["set_id"].unique().to_list()
    spread = (
        corpus.listings.group_by("photo_set_id")
        .agg(pl.col("area_id").n_unique().alias("areas"))
        .to_dict(as_series=False)
    )
    areas_by_set = dict(zip(spread["photo_set_id"], spread["areas"]))
    used_plain = [s for s in plain_sets if s in areas_by_set]
    assert used_plain, "the fixture must actually use plain sets"
    assert all(areas_by_set[s] >= config.stock_min_areas for s in stock_sets)
    assert all(areas_by_set[s] < config.stock_min_areas for s in used_plain)


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
    # The 6-set pool leaves 4 non-stock sets, so the sales must span at most 4 areas for every
    # area to get a local plain set (see _plain_sets_by_area). Unrelated to what is asserted here.
    corpus = generate_corpus(sales_frame(areas=4), areas_frame, photo_pool.set_ids, config)
    written = render_variants(corpus, photo_pool, config, tmp_path / "variants")
    assert written == config.n_edited_photo * 4
    for path in corpus.photos.filter(pl.col("variant_kind") == "edited")["path"].to_list():
        target = tmp_path / "variants" / path.split("/")[-1]
        assert target.is_file()
        with Image.open(target) as image:
            assert image.width > 0
