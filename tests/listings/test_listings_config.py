from listings.config import (
    DETECTION_FORBIDDEN_COLUMNS,
    LABEL_COLUMNS,
    PAIR_FEATURES,
    PHOTO_ROOMS,
    CorpusConfig,
    DetectConfig,
)


def test_pair_features_are_exact_and_ordered():
    assert PAIR_FEATURES == (
        "text_cosine", "image_max_cosine", "image_mean_cosine", "shared_photo_count",
        "abs_log_price_ratio", "abs_log_size_ratio", "same_area", "same_building",
        "same_project", "bedrooms_equal", "days_apart", "same_agent",
    )  # fmt: skip


def test_label_columns_are_named_so_detection_can_exclude_them():
    assert LABEL_COLUMNS == ("dup_group_id", "control_group_id", "fraud_label")
    assert not set(LABEL_COLUMNS) & set(PAIR_FEATURES)


def test_forbidden_columns_cover_every_free_duplicate_oracle():
    """Labels are not the only give-aways: provenance and variant columns are oracles too.

    source_transaction_id is shared by a clone and its source and unique to every base
    listing; variant_of/variant_kind name an edited copy without looking at an image. All
    three are kept in the corpus (truth and evaluation need them) but must never be selected
    by a detection module.
    """
    assert set(LABEL_COLUMNS) <= set(DETECTION_FORBIDDEN_COLUMNS)
    assert {"source_transaction_id", "variant_of", "variant_kind"} <= set(
        DETECTION_FORBIDDEN_COLUMNS
    )
    assert set(DETECTION_FORBIDDEN_COLUMNS) == set(LABEL_COLUMNS) | {
        "source_transaction_id",
        "variant_of",
        "variant_kind",
    }
    assert not set(DETECTION_FORBIDDEN_COLUMNS) & set(PAIR_FEATURES)
    assert len(set(DETECTION_FORBIDDEN_COLUMNS)) == len(DETECTION_FORBIDDEN_COLUMNS)


def test_corpus_counts_add_up_to_the_requested_total():
    config = CorpusConfig()
    assert config.n_listings == 20_000
    assert config.n_base == 17_000
    assert (config.n_exact_repost, config.n_reworded, config.n_edited_photo) == (1_200, 900, 900)
    assert config.n_base + config.n_exact_repost + config.n_reworded + config.n_edited_photo == (
        config.n_listings
    )
    assert config.n_bait_price == 600
    assert config.n_price_shifted_reposts == 300
    assert config.n_price_shifted_reposts <= config.n_exact_repost


def test_corpus_photo_and_sampling_settings():
    config = CorpusConfig()
    assert PHOTO_ROOMS == ("bathroom", "bedroom", "frontal", "kitchen")
    assert config.n_stock_sets == 40
    assert config.stock_share == 0.25
    assert config.stock_min_areas == 5
    assert config.min_building_sales == 4
    assert config.n_from_busy_buildings == 1_500
    assert config.asking_factor == (1.00, 1.08)
    assert config.bait_factor == (0.40, 0.65)
    assert config.seed == 42


def test_detect_settings_match_the_spec():
    config = DetectConfig()
    assert (config.text_top_k, config.photo_top_k, config.ef_search) == (20, 20, 100)
    assert config.max_photo_fanout == 50
    assert (config.train_share, config.threshold_share) == (0.60, 0.20)
    assert config.target_precision == 0.98
    assert config.baseline_image_cosine == 0.95
    assert config.photo_reuse_min_areas == 5
    assert config.relist_price_spread == 0.20
    assert config.bait_margin == 0.10
    assert config.experiment == "listing-dedup"
