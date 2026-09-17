"""Phase 7: a new listing scored against the stored corpus, and the stored-flags reader."""

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from listings.config import PAIR_FEATURES, DetectConfig
from listings.detect import run_detection, write_detection
from listings.embed import FakeEmbedder
from listings.fraud import KIND_BY_SUB_TYPE, run_fraud_checks, write_fraud_flags
from listings.pairmodel import PairModel
from models.price.predictor import PriceInputError

NO_PHOTOS_NOTE = "no photos given: image features are 0"
DEFAULT_CONFIG = DetectConfig()


class FakePrice:
    """A price predictor that returns one fixed estimate, or raises a fixed error."""

    def __init__(self, estimate=1_000_000.0, low=900_000.0, high=1_100_000.0, error=None):
        self.estimate = SimpleNamespace(
            estimate_aed=estimate, range_80=(low, high), model_version="p7"
        )
        self.error = error
        self.requests = []

    def predict_one(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.estimate


@dataclass
class Request:
    title: str
    description: str
    asking_price_aed: float
    area_id: int
    property_kind: str
    status: str
    size_sqm: float
    size_basis: str = "built_up"
    building_name: str | None = None
    project_name: str | None = None
    bedrooms: int | None = None
    agent_id: int | None = None
    posted_at: date | None = None
    photo_ids: list[int] = field(default_factory=list)


@pytest.fixture
def detected(loaded_corpus):
    """The loaded corpus with a fitted pair model; yields (settings, detection result, conn)."""
    settings, _, corpus_run_id = loaded_corpus
    conn = settings.connect()
    try:
        result = run_detection(conn, DetectConfig())
        conn.rollback()
        yield SimpleNamespace(
            settings=settings, result=result, conn=conn, corpus_run_id=corpus_run_id
        )
    finally:
        conn.close()


@pytest.fixture
def make_checker(detected):
    from listings.check import ListingChecker

    pair = PairModel(
        detected.result.model, detected.result.threshold, PAIR_FEATURES, 1, version="t"
    )

    def make(price=None, config=DEFAULT_CONFIG):
        price = FakePrice() if price == "default" else price
        return ListingChecker.from_connection(
            detected.conn, FakeEmbedder(), pair, price=price, config=config
        )

    return make


def fetch(conn, sql, params=()):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    conn.rollback()
    return rows


def repost_of(conn, listing_id: int, **changes) -> Request:
    """A repost a day later: same text, price, size, place and photos, agent left unset.

    The agent is not copied (it defaults to -1, as in the API): the corpus's planted reposts
    come from a different agent, so the fitted model treats a shared agent as a mild
    counter-signal.
    """
    (row,) = fetch(
        conn,
        "SELECT title, description, asking_price_aed, area_id, building_name, project_name, "
        "property_type, property_sub_type, reg_type, size_sqm, bedrooms, posted_at "
        "FROM listings.listings WHERE listing_id = %s",
        (listing_id,),
    )
    photos = fetch(
        conn,
        "SELECT photo_id FROM listings.listing_photos WHERE listing_id = %s ORDER BY position",
        (listing_id,),
    )
    (title, description, price, area_id, building, project, ptype, sub_type, reg, size,
     bedrooms, posted_at) = row  # fmt: skip
    request = Request(
        title=title,
        description=description,
        asking_price_aed=price,
        area_id=area_id,
        building_name=building,
        project_name=project,
        property_kind="villa" if ptype == "villa" else KIND_BY_SUB_TYPE[sub_type],
        status=reg,
        size_sqm=size,
        size_basis="plot" if ptype == "villa" and sub_type is None else "built_up",
        bedrooms=bedrooms,
        posted_at=posted_at + timedelta(days=1),
        photo_ids=[photo_id for (photo_id,) in photos],
    )
    for name, value in changes.items():
        setattr(request, name, value)
    return request


def plain_listing(conn) -> int:
    """A unit (building and bedrooms known) whose photo set spans fewer than 3 areas.

    A villa has no building or bedrooms, so its exact repost scores 0 on same_building and
    bedrooms_equal (null never equals null, as in Phase 4) and may sit under the threshold.
    """
    rows = fetch(
        conn,
        "SELECT min(l.listing_id) FROM listings.listings l "
        "JOIN listings.listing_photos lp ON lp.listing_id = l.listing_id "
        "WHERE l.building_name IS NOT NULL AND l.bedrooms IS NOT NULL "
        "AND l.photo_set_id IN (SELECT photo_set_id FROM listings.listings "
        "GROUP BY photo_set_id HAVING count(DISTINCT area_id) < 3)",
    )
    return rows[0][0]


def unrelated_request(**changes) -> Request:
    request = Request(
        title="Quixotic zephyr lighthouse",
        description="Marmalade harpsichord, velvet glacier, obsidian carousel.",
        asking_price_aed=1_500_000.0,
        area_id=987_654,
        property_kind="apartment",
        status="ready",
        size_sqm=95.0,
        building_name="Nowhere Tower",
        project_name="Nowhere Project",
        bedrooms=2,
    )
    for name, value in changes.items():
        setattr(request, name, value)
    return request


def test_an_exact_repost_is_caught(detected, make_checker):
    original = plain_listing(detected.conn)
    checker = make_checker(price="default")
    result = checker.check(repost_of(detected.conn, original))

    assert set(result) == {"duplicates", "flags", "notes", "model_versions"}
    by_id = {item["listing_id"]: item for item in result["duplicates"]}
    assert original in by_id
    assert by_id[original]["score"] >= detected.result.threshold
    assert set(by_id[original]["signals"]) == set(PAIR_FEATURES)
    assert by_id[original]["signals"]["shared_photo_count"] >= 1
    scores = [item["score"] for item in result["duplicates"]]
    assert scores == sorted(scores, reverse=True) and len(scores) <= 20
    assert result["model_versions"] == {"pair": "t", "price": "p7"}
    assert NO_PHOTOS_NOTE not in result["notes"]


def test_an_unrelated_listing_is_clean(make_checker):
    result = make_checker(price="default").check(unrelated_request())
    assert result["duplicates"] == []
    assert NO_PHOTOS_NOTE in result["notes"]
    assert not [flag for flag in result["flags"] if flag["flag"] == "inconsistent_relist"]


def test_a_bait_price_is_flagged(make_checker):
    price = FakePrice(estimate=1_000_000.0, low=900_000.0, high=1_100_000.0)
    result = make_checker(price=price).check(unrelated_request(asking_price_aed=700_000.0))
    (bait,) = [flag for flag in result["flags"] if flag["flag"] == "bait_price"]
    assert bait["detail"] == {
        "asking_price_aed": 700_000.0,
        "estimate_aed": 1_000_000.0,
        "range_80_low": 900_000.0,
        "below_low_pct": 22.2,
    }
    (request,) = price.requests
    assert request.area_id == 987_654 and request.building == "Nowhere Tower"
    assert request.project == "Nowhere Project" and request.bedrooms == 2


def test_a_fair_price_is_not_flagged(make_checker):
    result = make_checker(price="default").check(unrelated_request(asking_price_aed=950_000.0))
    assert not [flag for flag in result["flags"] if flag["flag"] == "bait_price"]


def test_price_check_is_skipped_with_a_note(make_checker):
    failing = FakePrice(error=PriceInputError("size_sqm", "x"))
    result = make_checker(price=failing).check(unrelated_request())
    assert "price check skipped: size_sqm: x" in result["notes"]
    assert result["model_versions"]["price"] is None

    result = make_checker(price=None).check(unrelated_request())
    assert "price check skipped: price model unavailable" in result["notes"]
    assert result["model_versions"] == {"pair": "t", "price": None}


def test_an_invalid_price_request_is_skipped_with_a_note(make_checker):
    result = make_checker(price="default").check(unrelated_request(property_kind="castle"))
    notes = [note for note in result["notes"] if note.startswith("price check skipped: ")]
    assert notes and "property_kind" in notes[0]


def test_stock_photos_are_flagged_as_reused(detected, make_checker):
    ((set_id, areas),) = fetch(
        detected.conn,
        "SELECT photo_set_id, count(DISTINCT area_id) FROM listings.listings "
        "GROUP BY photo_set_id ORDER BY 2 DESC, 1 LIMIT 1",
    )
    # The small fixture corpus spans 4 areas and plants stock sets across >= 3 of them.
    assert areas >= 3
    photos = fetch(
        detected.conn,
        "SELECT lp.photo_id FROM listings.listing_photos lp "
        "JOIN listings.listings l ON l.listing_id = lp.listing_id "
        "WHERE l.photo_set_id = %s ORDER BY lp.listing_id, lp.position LIMIT 4",
        (set_id,),
    )
    # The request's own area (987_654) is not a corpus area, so it adds one to the spread.
    checker = make_checker(price="default", config=DetectConfig(photo_reuse_min_areas=areas + 1))
    result = checker.check(unrelated_request(photo_ids=[photo_id for (photo_id,) in photos]))
    (reuse,) = [flag for flag in result["flags"] if flag["flag"] == "photo_reuse"]
    assert reuse["detail"]["photo_set_id"] == set_id
    assert reuse["detail"]["areas"] == areas + 1
    assert reuse["detail"]["listings"] >= areas + 1
    assert NO_PHOTOS_NOTE not in result["notes"]


def test_plain_photos_are_not_flagged_as_reused(detected, make_checker):
    request = repost_of(detected.conn, plain_listing(detected.conn))
    checker = make_checker(price="default", config=DetectConfig(photo_reuse_min_areas=3))
    result = checker.check(request)
    assert not [flag for flag in result["flags"] if flag["flag"] == "photo_reuse"]


def test_a_price_shifted_repost_is_relist_flagged(detected, make_checker):
    original = plain_listing(detected.conn)
    request = repost_of(detected.conn, original)
    request.asking_price_aed *= 1.5
    result = make_checker(price="default").check(request)
    (relist,) = [flag for flag in result["flags"] if flag["flag"] == "inconsistent_relist"]
    detail = relist["detail"]
    assert detail["decision"] == "price_blind"
    assert detail["cluster_size"] >= 2
    assert detail["spread"] > DetectConfig().relist_price_spread
    assert detail["max_price_aed"] == pytest.approx(request.asking_price_aed)


def test_unknown_photos_are_rejected(make_checker):
    from listings.check import UnknownPhotos

    with pytest.raises(UnknownPhotos) as raised:
        make_checker(price="default").check(unrelated_request(photo_ids=[999999999]))
    assert raised.value.ids == [999999999]
    assert isinstance(raised.value, ValueError)


def write_stored_run(detected) -> int:
    conn, result = detected.conn, detected.result
    detect_run_id = write_detection(conn, result, detected.corpus_run_id)
    fraud = run_fraud_checks(conn, result.relist_pairs, DetectConfig(), predictor=None)
    write_fraud_flags(conn, fraud, detect_run_id)
    conn.commit()
    return detect_run_id


def test_stored_flags_return_the_latest_run(detected):
    from listings.check import stored_flags

    detect_run_id = write_stored_run(detected)
    conn = detected.conn
    ((a, b, score),) = fetch(
        conn,
        "SELECT listing_a, listing_b, score FROM listings.duplicate_pairs "
        "WHERE detect_run_id = %s ORDER BY score DESC, listing_a, listing_b LIMIT 1",
        (detect_run_id,),
    )
    found = stored_flags(conn, b)
    assert found["listing_id"] == b and found["detect_run_id"] == detect_run_id
    by_id = {item["listing_id"]: item for item in found["duplicates"]}
    assert by_id[a]["score"] == pytest.approx(score)
    assert set(by_id[a]["signals"]) == set(PAIR_FEATURES)
    assert stored_flags(conn, a)["duplicates"][0]["listing_id"] != a

    ((flagged, flag),) = fetch(
        conn,
        "SELECT listing_id, flag FROM listings.fraud_flags WHERE detect_run_id = %s "
        "ORDER BY listing_id LIMIT 1",
        (detect_run_id,),
    )
    flags = stored_flags(conn, flagged)["flags"]
    assert flag in {item["flag"] for item in flags}
    assert all(isinstance(item["detail"], dict) for item in flags)


def test_stored_flags_for_unknown_and_clean_listings(detected):
    from listings.check import stored_flags

    detect_run_id = write_stored_run(detected)
    conn = detected.conn
    assert stored_flags(conn, 999_999_999) is None
    ((clean,),) = fetch(
        conn,
        "SELECT min(listing_id) FROM listings.listings l WHERE NOT EXISTS ("
        " SELECT 1 FROM listings.duplicate_pairs d WHERE d.detect_run_id = %(run)s"
        " AND l.listing_id IN (d.listing_a, d.listing_b))"
        " AND NOT EXISTS (SELECT 1 FROM listings.fraud_flags f"
        " WHERE f.detect_run_id = %(run)s AND f.listing_id = l.listing_id)",
        {"run": detect_run_id},
    )
    assert stored_flags(conn, clean) == {
        "listing_id": clean,
        "detect_run_id": detect_run_id,
        "duplicates": [],
        "flags": [],
    }


def test_stored_flags_ignore_runs_of_an_older_corpus(detected):
    from listings.check import stored_flags

    write_stored_run(detected)
    conn = detected.conn
    ((listing_id,),) = fetch(conn, "SELECT min(listing_id) FROM listings.listings")
    with conn.cursor() as cur:  # a newer corpus run that has not been detected yet
        cur.execute(
            "INSERT INTO listings.corpus_runs (seed, photo_dataset_sha, counts) "
            "VALUES (0, NULL, '{}'::jsonb)"
        )
    conn.commit()
    assert stored_flags(conn, listing_id) == {
        "listing_id": listing_id,
        "detect_run_id": None,
        "duplicates": [],
        "flags": [],
    }


# --- without a database ---------------------------------------------------------------------


class ScriptedCursor:
    def __init__(self, conn):
        self.conn = conn
        self.rows = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.conn.executed.append(sql)
        answer = self.conn.answers.get(sql, [])
        if isinstance(answer, Exception):
            raise answer
        self.rows = list(answer)

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


class ScriptedConn:
    """Answers each SQL string with fixed rows (or raises); rollback can fail too."""

    def __init__(self, answers=None, rollback_error=None):
        self.answers = answers or {}
        self.rollback_error = rollback_error
        self.executed = []

    def cursor(self):
        return ScriptedCursor(self)

    def rollback(self):
        if self.rollback_error is not None:
            raise self.rollback_error


class StubPair:
    threshold = 0.5
    version = "pair-3"

    def scores(self, features):
        return np.zeros(features.height)

    def price_blind_scores(self, features):
        return np.zeros(features.height)


def bare_checker(conn, fraud_rows=(), price=None, price_version=None, config=DEFAULT_CONFIG):
    from listings.check import ListingChecker
    from listings.features import LISTING_ATTRIBUTE_SCHEMA

    fraud = pl.DataFrame(
        list(fraud_rows),
        schema={"listing_id": pl.Int64, "photo_set_id": pl.Int64, "area_id": pl.Int64},
        orient="row",
    )
    return ListingChecker(
        conn, FakeEmbedder(), StubPair(), price, config,
        pl.DataFrame(schema=LISTING_ATTRIBUTE_SCHEMA), {}, {}, {},
        {11: np.ones(4), 12: np.ones(4)}, fraud, price_version=price_version,
    )  # fmt: skip


def photo_conn():
    from listings import check

    return ScriptedConn({check.KNOWN_PHOTOS_SQL: [(11,), (12,)], check.PHOTO_SETS_SQL: [(7,)]})


# Photo set 7 is used by three stored listings in areas 1 and 2.
SET_7 = [(1, 7, 1), (2, 7, 2), (3, 7, 2)]


def test_photo_reuse_counts_the_new_listings_own_area():
    checker = bare_checker(photo_conn(), SET_7, config=DetectConfig(photo_reuse_min_areas=3))
    elsewhere = checker.check(unrelated_request(area_id=3, photo_ids=[11, 12]))
    (reuse,) = [flag for flag in elsewhere["flags"] if flag["flag"] == "photo_reuse"]
    assert reuse["detail"] == {"photo_set_id": 7, "areas": 3, "listings": 4}

    same_area = checker.check(unrelated_request(area_id=2, photo_ids=[11, 12]))
    assert not [flag for flag in same_area["flags"] if flag["flag"] == "photo_reuse"]


def test_model_versions_report_the_loaded_price_model():
    failing = FakePrice(error=PriceInputError("area_id", "unknown area"))
    checker = bare_checker(ScriptedConn(), price=failing, price_version="12")
    result = checker.check(unrelated_request())
    assert result["model_versions"] == {"pair": "pair-3", "price": "12"}

    no_price = bare_checker(ScriptedConn(), price=None, price_version=None)
    assert no_price.check(unrelated_request())["model_versions"]["price"] is None


def test_a_failed_rollback_does_not_hide_the_query_error():
    import psycopg2

    from listings import check

    lost = psycopg2.OperationalError("server closed the connection unexpectedly")
    conn = ScriptedConn(
        {check.KNOWN_PHOTOS_SQL: lost},
        rollback_error=psycopg2.InterfaceError("connection already closed"),
    )
    with pytest.raises(psycopg2.OperationalError):
        bare_checker(conn).check(unrelated_request(photo_ids=[11]))
    conn.answers = {check.LISTING_EXISTS_SQL: lost}
    with pytest.raises(psycopg2.OperationalError):
        check.stored_flags(conn, 5)


def test_stored_flags_with_no_run_for_the_latest_corpus():
    from listings import check

    conn = ScriptedConn({check.LISTING_EXISTS_SQL: [(1,)], check.LATEST_RUN_SQL: [(None,)]})
    assert check.stored_flags(conn, 5) == {
        "listing_id": 5, "detect_run_id": None, "duplicates": [], "flags": [],
    }  # fmt: skip
    assert "corpus_runs" in check.LATEST_RUN_SQL
    assert check.STORED_PAIRS_SQL not in conn.executed


def test_checker_selects_no_free_duplicate_oracle():
    """The checker must never read a ground-truth or provenance column."""
    from listings import check
    from listings.config import DETECTION_FORBIDDEN_COLUMNS

    source = Path(check.__file__).read_text(encoding="utf-8")
    for name in DETECTION_FORBIDDEN_COLUMNS:
        assert name not in source
