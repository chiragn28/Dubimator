import dataclasses
import json

import polars as pl
import pytest

from listings.config import HOME_UNIT_SUB_TYPES, DetectConfig
from listings.fraud import (
    KIND_BY_SUB_TYPE,
    bait_price_flags,
    inconsistent_relist_flags,
    photo_reuse_flags,
    price_request,
    run_fraud_checks,
    write_fraud_flags,
)
from models.price.predictor import PriceRequest

CONFIG = DetectConfig()

ATTRIBUTES = pl.DataFrame(
    {
        "listing_id": [1, 2, 3, 4],
        "asking_price_aed": [1_000_000.0, 400_000.0, 2_000_000.0, 2_100_000.0],
        "area_id": [10, 10, 20, 20],
        "area_name": ["Marsa Dubai"] * 2 + ["Hadaeq Sheikh Mohammed Bin Rashid"] * 2,
        "building_name": ["Marina Gate 1", "Marina Gate 1", None, None],
        "project_name": ["Marina Gate", "Marina Gate", "Dubai Hills", "Dubai Hills"],
        "property_type": ["unit", "unit", "villa", "villa"],
        "property_sub_type": ["Flat", "Flat", None, "Villa"],
        "reg_type": ["ready", "ready", "ready", "off_plan"],
        "size_sqm": [100.0, 100.0, 520.0, 300.0],
        "bedrooms": [2, 2, None, None],
        "photo_set_id": [7, 7, 8, 9],
    }
)


class StubPredictor:
    """Estimates 1,000,000 with an 80% range of 900,000-1,100,000, whatever it is asked."""

    def __init__(self, unsupported: set[int] | None = None):
        self.calls = []
        self.unsupported = unsupported or set()

    def predict_one(self, request):
        from listings.fraud import PriceInputError

        self.calls.append(request)
        if len(self.calls) in self.unsupported:
            raise PriceInputError("status", "unsupported combination")
        return type(
            "Estimate",
            (),
            {"estimate_aed": 1_000_000.0, "range_80": (900_000.0, 1_100_000.0)},
        )()


def test_price_request_maps_listing_shapes():
    rows = ATTRIBUTES.to_dicts()
    flat = price_request(rows[0])
    assert flat["property_kind"] == "apartment" and flat["size_basis"] == "built_up"
    assert flat["status"] == "ready" and flat["area_id"] == 10
    assert flat["size_sqm"] == 100.0 and flat["bedrooms"] == 2
    assert flat["building"] == "Marina Gate 1" and flat["project"] == "Marina Gate"
    PriceRequest.parse(flat)  # the real contract, not just the dict shape above

    plot_villa = price_request(rows[2])
    assert plot_villa["property_kind"] == "villa" and plot_villa["size_basis"] == "plot"
    assert "bedrooms" not in plot_villa  # unknown bedrooms are left out, not sent as None
    PriceRequest.parse(plot_villa)

    built_villa = price_request(rows[3])
    assert built_villa["size_basis"] == "built_up" and built_villa["status"] == "off_plan"
    PriceRequest.parse(built_villa)


def test_kind_by_sub_type_covers_every_home_unit_sub_type():
    """KIND_BY_SUB_TYPE, HOME_UNIT_SUB_TYPES (config.py) and generate.py's own copy must agree
    on which sub types exist; this pins the fraud module's half of that invariant."""
    assert set(KIND_BY_SUB_TYPE) == set(HOME_UNIT_SUB_TYPES)


def test_price_request_skips_rather_than_crashes_on_an_unknown_sub_type():
    from listings.fraud import PriceInputError

    row = {**ATTRIBUTES.to_dicts()[0], "property_sub_type": "Penthouse Suite"}
    with pytest.raises(PriceInputError):
        price_request(row)


def test_bait_price_flags_only_the_cheap_listing():
    predictor = StubPredictor()
    flags, stats = bait_price_flags(ATTRIBUTES, predictor, CONFIG)
    assert flags["listing_id"].to_list() == [2]
    detail = json.loads(flags["detail"][0])
    assert detail["asking_price_aed"] == 400_000.0
    assert detail["range_80_low"] == 900_000.0
    assert detail["below_low_pct"] == pytest.approx(55.6)
    assert stats["bait_price_checked"] == 4.0
    assert stats["bait_price_skipped"] == 0.0


def test_listings_the_price_model_cannot_price_are_skipped_not_failed():
    predictor = StubPredictor(unsupported={2})
    flags, stats = bait_price_flags(ATTRIBUTES, predictor, CONFIG)
    assert 2 not in flags["listing_id"].to_list()
    assert stats["bait_price_unsupported"] == 1.0


def test_bait_price_flags_logs_progress(caplog):
    """log_every is a bait_price_flags-only knob (the CLI wants 2,000; run_fraud_checks does
    not plumb it through), so it is exercised directly here rather than via run_fraud_checks."""
    import logging

    predictor = StubPredictor()
    with caplog.at_level(logging.INFO, logger="listings.fraud"):
        bait_price_flags(ATTRIBUTES, predictor, CONFIG, log_every=1)
    assert any("priced" in record.message for record in caplog.records)


def test_photo_reuse_needs_several_areas():
    attributes = pl.DataFrame(
        {
            "listing_id": list(range(1, 8)),
            "photo_set_id": [5, 5, 5, 5, 5, 6, 6],
            "area_id": [1, 2, 3, 4, 5, 1, 2],
        }
    )
    flags = photo_reuse_flags(attributes, CONFIG)
    assert sorted(flags["listing_id"].to_list()) == [1, 2, 3, 4, 5]
    assert json.loads(flags["detail"][0])["areas"] == 5
    assert json.loads(flags["detail"][0])["photo_set_id"] == 5


def test_inconsistent_relist_flags_clusters_whose_prices_disagree():
    attributes = pl.DataFrame(
        {
            "listing_id": [1, 2, 3, 4, 5],
            "asking_price_aed": [1_000_000.0, 1_050_000.0, 1_400_000.0, 2_000_000.0, 2_020_000.0],
        }
    )
    flagged = pl.DataFrame({"listing_a": [1, 2, 4], "listing_b": [2, 3, 5]})
    flags = inconsistent_relist_flags(attributes, flagged, CONFIG)
    # 1-2-3 is one cluster spanning 1.0M-1.4M (40%); 4-5 spans 1% and is fine
    assert sorted(flags["listing_id"].to_list()) == [1, 2, 3]
    detail = json.loads(flags["detail"][0])
    assert detail["spread"] == pytest.approx(0.4)
    assert detail["cluster_size"] == 3


def test_a_missing_price_model_skips_bait_but_keeps_the_rest(loaded_corpus):
    settings, _, _ = loaded_corpus
    conn = settings.connect()
    # Same fixture-scale fix as test_write_fraud_flags_stores_them_per_run below: loaded_corpus
    # spans exactly 4 areas, so the production photo_reuse_min_areas=5 default can never fire
    # and result.flags would be empty by construction — which would let the assertions below
    # pass vacuously (not in [], set() <= anything) without exercising "keeps the rest" at all.
    config = dataclasses.replace(CONFIG, photo_reuse_min_areas=4)
    try:
        result = run_fraud_checks(
            conn, pl.DataFrame({"listing_a": [], "listing_b": []}), config, predictor=None
        )
    finally:
        conn.close()
    assert result.stats["bait_price_skipped"] == 1.0
    assert result.stats["bait_price_checked"] == 0.0
    assert result.stats["bait_price_unsupported"] == 0.0
    assert "bait_price" not in result.flags["flag"].to_list()
    assert "photo_reuse" in result.flags["flag"].to_list()  # the rest actually ran
    assert set(result.flags["flag"].to_list()) <= {"photo_reuse", "inconsistent_relist"}


def test_write_fraud_flags_stores_them_per_run(loaded_corpus):
    settings, _, corpus_run_id = loaded_corpus
    conn = settings.connect()
    # loaded_corpus's sales span exactly 4 areas (see tests/listings/conftest.py), so the
    # production photo_reuse_min_areas=5 default can never fire here — no photo set can ever
    # span a 5th area. Lower it to 4 for this write-path check only, the same
    # dataclasses.replace(CONFIG, ...) pattern test_listings_candidates.py already uses to fit
    # a threshold to this small fixture; every other test keeps the unmodified production CONFIG.
    config = dataclasses.replace(CONFIG, photo_reuse_min_areas=4)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO listings.detect_runs (corpus_run_id, threshold) VALUES (%s, 0.5) "
                "RETURNING detect_run_id",
                (corpus_run_id,),
            )
            detect_run_id = cur.fetchone()[0]
        result = run_fraud_checks(
            conn, pl.DataFrame({"listing_a": [], "listing_b": []}), config, predictor=None
        )
        written = write_fraud_flags(conn, result, detect_run_id)
        conn.commit()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*), count(DISTINCT flag) FROM listings.fraud_flags WHERE detect_run_id=%s",
                (detect_run_id,),
            )
            count, flags = cur.fetchone()
    finally:
        conn.close()
    assert count == written == result.flags.height
    assert flags >= 1


def test_fraud_sql_selects_no_free_duplicate_oracle():
    """Detection code (fraud checks included) must never read a ground-truth label column."""
    from listings.config import DETECTION_FORBIDDEN_COLUMNS
    from listings.fraud import FRAUD_LISTING_SQL

    for name in DETECTION_FORBIDDEN_COLUMNS:
        assert name not in FRAUD_LISTING_SQL


def test_a_price_shifted_repost_is_relist_flagged(loaded_corpus):
    """inconsistent_relist looks for duplicate clusters whose prices disagree, so it must be fed
    a price-BLIND duplicate decision: the headline model penalises a price gap so heavily that
    a price-shifted repost is exactly what it never flags (final review, Important 1)."""
    from listings.detect import run_detection
    from listings.truth import load_relist_truth

    settings, _, _ = loaded_corpus
    conn = settings.connect()
    try:
        result = run_detection(conn, CONFIG)
        relist_truth = set(load_relist_truth(conn, CONFIG.relist_price_spread)["listing_id"])
        fraud = run_fraud_checks(conn, result.relist_pairs, CONFIG, predictor=None)
    finally:
        conn.close()

    assert relist_truth, "the fixture must plant at least one repost shifted beyond the spread"
    relist = fraud.flags.filter(pl.col("flag") == "inconsistent_relist")
    caught = set(relist["listing_id"].to_list()) & relist_truth
    assert caught, "no price-shifted repost was relist-flagged"
    # The price-blind pairs are a superset of the headline decisions, scored by the same model
    # at the same threshold with only the price gap zeroed.
    headline = set(
        zip(
            *result.pairs.filter(pl.col("decision"))
            .select("listing_a", "listing_b")
            .to_dict(as_series=False)
            .values()
        )
    )
    blind = set(zip(result.relist_pairs["listing_a"], result.relist_pairs["listing_b"]))
    assert headline <= blind
    # ...and the truth listings it caught include ones the headline decision alone misses.
    headline_flags = inconsistent_relist_flags(
        fraud_attributes_for(settings), result.pairs.filter(pl.col("decision")), CONFIG
    )
    assert caught - set(headline_flags["listing_id"].to_list())


def fraud_attributes_for(settings):
    from listings.fraud import load_fraud_attributes

    conn = settings.connect()
    try:
        return load_fraud_attributes(conn)
    finally:
        conn.close()


def test_price_model_version_resolves_through_the_alias(temp_mlflow):
    import mlflow

    from listings.fraud import resolve_price_model_version

    mlflow.set_tracking_uri(temp_mlflow["tracking_uri"])
    client = mlflow.MlflowClient()
    client.create_registered_model("toy-price")
    for _ in range(2):
        client.create_model_version("toy-price", source="file:///nowhere", run_id=None)
    client.set_registered_model_alias("toy-price", "champion", "2")

    assert resolve_price_model_version("models:/toy-price@champion") == "2"
    assert resolve_price_model_version("models:/toy-price/1") == "1"
    assert resolve_price_model_version("models:/missing@champion") is None
    assert resolve_price_model_version("runs:/abc/model") is None
