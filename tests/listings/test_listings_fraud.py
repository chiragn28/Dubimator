import dataclasses
import json

import polars as pl
import pytest

from listings.config import DetectConfig
from listings.fraud import (
    bait_price_flags,
    inconsistent_relist_flags,
    photo_reuse_flags,
    price_request,
    run_fraud_checks,
    write_fraud_flags,
)

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

    plot_villa = price_request(rows[2])
    assert plot_villa["property_kind"] == "villa" and plot_villa["size_basis"] == "plot"
    assert "bedrooms" not in plot_villa  # unknown bedrooms are left out, not sent as None

    built_villa = price_request(rows[3])
    assert built_villa["size_basis"] == "built_up" and built_villa["status"] == "off_plan"


def test_bait_price_flags_only_the_cheap_listing():
    predictor = StubPredictor()
    flags, stats = bait_price_flags(ATTRIBUTES, predictor, CONFIG)
    assert flags["listing_id"].to_list() == [2]
    detail = json.loads(flags["detail"][0])
    assert detail["asking_price_aed"] == 400_000.0
    assert detail["range_80_low"] == 900_000.0
    assert stats["bait_price_checked"] == 4.0
    assert stats["bait_price_skipped"] == 0.0


def test_listings_the_price_model_cannot_price_are_skipped_not_failed():
    predictor = StubPredictor(unsupported={2})
    flags, stats = bait_price_flags(ATTRIBUTES, predictor, CONFIG)
    assert 2 not in flags["listing_id"].to_list()
    assert stats["bait_price_unsupported"] == 1.0


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


def test_a_missing_price_model_skips_bait_but_keeps_the_rest(loaded_corpus, caplog):
    settings, _, _ = loaded_corpus
    conn = settings.connect()
    try:
        result = run_fraud_checks(
            conn, pl.DataFrame({"listing_a": [], "listing_b": []}), CONFIG, predictor=None
        )
    finally:
        conn.close()
    assert result.stats["bait_price_skipped"] == 1.0
    assert "bait_price" not in result.flags["flag"].to_list()
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
