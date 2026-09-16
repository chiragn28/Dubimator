"""A new listing scored against the stored corpus, and the stored flags of a known one.

Spec: docs/superpowers/specs/2026-09-17-phase7-api-design.md ("Listing check").

The checker reuses the Phase 4 pair features and fraud rules, with the new listing as a
temporary attribute row (`NEW_LISTING_ID`) paired with its pgvector candidates. It reads
detection columns only.
"""

import threading
from datetime import date

import numpy as np
import polars as pl

from listings.config import PAIR_FEATURES, DetectConfig
from listings.embed import IMAGE_DIM, normalize
from listings.features import (
    LISTING_ATTRIBUTE_SCHEMA,
    build_features,
    load_listing_attributes,
    load_listing_vectors,
    load_photo_sets,
)
from listings.fraud import load_fraud_attributes
from listings.load import vector_literal
from models.price.predictor import PriceInputError, PriceRequest

NEW_LISTING_ID = -1
DEFAULT_CONFIG = DetectConfig()  # frozen, so one shared default is safe
MAX_DUPLICATES = 20
NO_PHOTOS_NOTE = "no photos given: image features are 0"
NO_PRICE_MODEL_NOTE = "price check skipped: price model unavailable"

TEXT_NEIGHBOURS_SQL = (
    "SELECT listing_id FROM listings.listing_embeddings WHERE text_embedding IS NOT NULL "
    "ORDER BY text_embedding <=> %s::vector LIMIT %s"
)
IMAGE_NEIGHBOURS_SQL = (
    "SELECT listing_id FROM listings.listing_embeddings WHERE image_embedding IS NOT NULL "
    "ORDER BY image_embedding <=> %s::vector LIMIT %s"
)
KNOWN_PHOTOS_SQL = "SELECT photo_id FROM listings.photos WHERE photo_id = ANY(%s)"
# Photos used by more than max_photo_fanout listings are agency stock (as in Phase 4).
SHARED_PHOTO_SQL = """
SELECT DISTINCT lp.listing_id
FROM listings.listing_photos lp
WHERE lp.photo_id = ANY(%(photos)s)
  AND lp.photo_id IN (
      SELECT photo_id FROM listings.listing_photos
      WHERE photo_id = ANY(%(photos)s)
      GROUP BY photo_id
      HAVING count(*) <= %(fanout)s
  )
ORDER BY lp.listing_id
"""
PHOTO_SETS_SQL = """
SELECT DISTINCT l.photo_set_id
FROM listings.listing_photos lp
JOIN listings.listings l ON l.listing_id = lp.listing_id
WHERE lp.photo_id = ANY(%s)
"""

LATEST_RUN_SQL = "SELECT max(detect_run_id) FROM listings.detect_runs"
LISTING_EXISTS_SQL = "SELECT 1 FROM listings.listings WHERE listing_id = %s"
STORED_PAIRS_SQL = """
SELECT CASE WHEN listing_a = %(id)s THEN listing_b ELSE listing_a END AS other,
       score, signals
FROM listings.duplicate_pairs
WHERE detect_run_id = %(run)s AND decision AND (listing_a = %(id)s OR listing_b = %(id)s)
ORDER BY score DESC, other
"""
STORED_FLAGS_SQL = """
SELECT flag, detail
FROM listings.fraud_flags
WHERE detect_run_id = %(run)s AND listing_id = %(id)s
ORDER BY flag
"""


class UnknownPhotos(ValueError):
    """Photo ids that are not in listings.photos; the API maps this to a 422."""

    def __init__(self, ids: list[int]):
        super().__init__(f"unknown photo ids: {ids}")
        self.ids = ids


def _field(request, name: str):
    return getattr(request, name, None)


class ListingChecker:
    def __init__(
        self,
        conn,
        embedder,
        pair_model,
        price,
        config: DetectConfig,
        attributes: pl.DataFrame,
        text_vectors: dict[int, np.ndarray],
        image_vectors: dict[int, np.ndarray],
        listing_photo_ids: dict[int, list[int]],
        photo_vectors: dict[int, np.ndarray],
        fraud_attributes: pl.DataFrame,
    ):
        self.conn = conn
        self.embedder = embedder
        self.pair_model = pair_model
        self.price = price
        self.config = config
        self.attributes = attributes
        self.text_vectors = text_vectors
        self.image_vectors = image_vectors
        self.listing_photo_ids = listing_photo_ids
        self.photo_vectors = photo_vectors
        self.lock = threading.Lock()
        self.latest_posted_at = attributes["posted_at"].max()
        self.prices = dict(
            zip(attributes["listing_id"].to_list(), attributes["asking_price_aed"].to_list())
        )
        self.image_dim = next((v.size for v in image_vectors.values()), IMAGE_DIM)
        spread = fraud_attributes.group_by("photo_set_id").agg(
            pl.col("area_id").n_unique().alias("areas"), pl.len().alias("listings")
        )
        self.set_spread = {
            row["photo_set_id"]: (row["areas"], row["listings"])
            for row in spread.iter_rows(named=True)
        }

    @classmethod
    def from_connection(cls, conn, embedder, pair_model, price, config=DEFAULT_CONFIG):
        attributes = load_listing_attributes(conn)
        text_vectors, image_vectors = load_listing_vectors(conn)
        listing_photo_ids, photo_vectors = load_photo_sets(conn)
        fraud_attributes = load_fraud_attributes(conn)
        conn.rollback()
        return cls(
            conn, embedder, pair_model, price, config, attributes, text_vectors,
            image_vectors, listing_photo_ids, photo_vectors, fraud_attributes,
        )  # fmt: skip

    # --- database reads, serialized on the checker's connection -------------------------

    def _fetch(self, sql: str, params, ef_search: int | None = None) -> list[tuple]:
        with self.lock:
            try:
                with self.conn.cursor() as cur:
                    if ef_search is not None:
                        cur.execute("SET hnsw.ef_search = %s", (ef_search,))
                    cur.execute(sql, params)
                    return cur.fetchall()
            finally:
                self.conn.rollback()

    def _neighbours(self, sql: str, vector: np.ndarray, limit: int) -> list[int]:
        if not np.any(vector):  # a zero vector has no cosine neighbours
            return []
        rows = self._fetch(sql, (vector_literal(vector), limit), self.config.ef_search)
        return [listing_id for (listing_id,) in rows]

    def _validate_photos(self, photo_ids: list[int]) -> None:
        if not photo_ids:
            return
        known = {photo_id for (photo_id,) in self._fetch(KNOWN_PHOTOS_SQL, (photo_ids,))}
        unknown = [photo_id for photo_id in photo_ids if photo_id not in known]
        if unknown:
            raise UnknownPhotos(unknown)

    def _shared_photo_listings(self, photo_ids: list[int]) -> list[int]:
        if not photo_ids:
            return []
        params = {"photos": photo_ids, "fanout": self.config.max_photo_fanout}
        return [listing_id for (listing_id,) in self._fetch(SHARED_PHOTO_SQL, params)]

    # --- vectors and features -----------------------------------------------------------

    def _image_vector(self, photo_ids: list[int]) -> np.ndarray:
        vectors = [self.photo_vectors[pid] for pid in photo_ids if pid in self.photo_vectors]
        if not vectors:
            return np.zeros(self.image_dim)
        return normalize(np.mean(vectors, axis=0))

    def _candidates(self, text_vec, image_vec, photo_ids: list[int]) -> list[int]:
        found = self._neighbours(TEXT_NEIGHBOURS_SQL, text_vec, self.config.text_top_k)
        if photo_ids:
            found += self._neighbours(IMAGE_NEIGHBOURS_SQL, image_vec, self.config.photo_top_k)
        found += self._shared_photo_listings(photo_ids)
        return list(dict.fromkeys(found))

    def _new_row(self, request) -> pl.DataFrame:
        agent_id, posted_at = _field(request, "agent_id"), _field(request, "posted_at")
        if isinstance(posted_at, str):
            posted_at = date.fromisoformat(posted_at)
        row = {
            "listing_id": NEW_LISTING_ID,
            "posted_at": posted_at or self.latest_posted_at,
            "asking_price_aed": float(request.asking_price_aed),
            "size_sqm": float(request.size_sqm),
            "bedrooms": _field(request, "bedrooms"),
            "area_id": int(request.area_id),
            "building_name": _field(request, "building_name"),
            "project_name": _field(request, "project_name"),
            "agent_id": -1 if agent_id is None else int(agent_id),
            "photo_set_id": -1,
        }
        return pl.DataFrame([row], schema=LISTING_ATTRIBUTE_SCHEMA)

    def _features(self, request, candidates, text_vec, image_vec, photo_ids) -> pl.DataFrame:
        # Only the candidates' rows and vectors are needed: build_features looks up each
        # pair's two listings, so the subset gives the same features as the full corpus.
        wanted = set(candidates)
        attributes = pl.concat(
            [self.attributes.filter(pl.col("listing_id").is_in(wanted)), self._new_row(request)]
        )

        def subset(vectors: dict) -> dict:
            return {key: vectors[key] for key in candidates if key in vectors}

        pairs = pl.DataFrame(
            {"listing_a": [NEW_LISTING_ID] * len(candidates), "listing_b": candidates},
            schema={"listing_a": pl.Int64, "listing_b": pl.Int64},
        )
        return build_features(
            pairs,
            attributes,
            {**subset(self.text_vectors), NEW_LISTING_ID: text_vec},
            {**subset(self.image_vectors), NEW_LISTING_ID: image_vec},
            {**subset(self.listing_photo_ids), NEW_LISTING_ID: photo_ids},
            self.photo_vectors,
        )

    # --- decisions and flags ------------------------------------------------------------

    def _duplicates(self, features: pl.DataFrame, scores: np.ndarray) -> list[dict]:
        threshold = self.pair_model.threshold
        found = [
            {
                "listing_id": int(row["listing_b"]),
                "score": float(score),
                "signals": {name: round(row[name], 4) for name in PAIR_FEATURES},
            }
            for row, score in zip(features.iter_rows(named=True), scores)
            if score >= threshold
        ]
        found.sort(key=lambda item: (-item["score"], item["listing_id"]))
        return found[:MAX_DUPLICATES]

    def _bait(self, request) -> tuple[dict | None, list[str], object]:
        """(flag or None, notes, estimate or None)."""
        if self.price is None:
            return None, [NO_PRICE_MODEL_NOTE], None
        payload = {
            "area_id": request.area_id,
            "project": _field(request, "project_name"),
            "building": _field(request, "building_name"),
            "property_kind": request.property_kind,
            "status": request.status,
            "size_sqm": request.size_sqm,
            "size_basis": _field(request, "size_basis"),
            "bedrooms": _field(request, "bedrooms"),
        }
        try:
            price_request = PriceRequest.parse(
                {key: value for key, value in payload.items() if value is not None}
            )
            estimate = self.price.predict_one(price_request)
        except PriceInputError as exc:
            return None, [f"price check skipped: {exc}"], None
        low = float(estimate.range_80[0])
        asking = float(request.asking_price_aed)
        if asking >= low * (1.0 - self.config.bait_margin):
            return None, [], estimate
        detail = {
            "asking_price_aed": asking,
            "estimate_aed": float(estimate.estimate_aed),
            "range_80_low": low,
            "below_low_pct": round((1.0 - asking / low) * 100.0, 1),
        }
        return {"flag": "bait_price", "detail": detail}, [], estimate

    def _photo_reuse(self, photo_ids: list[int]) -> dict | None:
        if not photo_ids:
            return None
        sets = [set_id for (set_id,) in self._fetch(PHOTO_SETS_SQL, (photo_ids,))]
        spreads = [self.set_spread[set_id] for set_id in sets if set_id in self.set_spread]
        if not spreads:
            return None
        areas, listings = max(spreads)
        if areas < self.config.photo_reuse_min_areas:
            return None
        return {"flag": "photo_reuse", "detail": {"areas": areas, "listings": listings}}

    def _relist(self, request, features: pl.DataFrame) -> dict | None:
        if features.is_empty():
            return None
        blind = self.pair_model.price_blind_scores(features)
        members = [
            int(listing_id)
            for listing_id, score in zip(features["listing_b"].to_list(), blind)
            if score >= self.pair_model.threshold and listing_id in self.prices
        ]
        if not members:
            return None
        values = [float(request.asking_price_aed)] + [self.prices[m] for m in members]
        if min(values) <= 0:
            return None
        spread = max(values) / min(values) - 1.0
        if spread <= self.config.relist_price_spread:
            return None
        detail = {
            "spread": round(spread, 4),
            "cluster_size": len(values),
            "decision": "price_blind",
            "min_price_aed": min(values),
            "max_price_aed": max(values),
        }
        return {"flag": "inconsistent_relist", "detail": detail}

    # --- entry point --------------------------------------------------------------------

    def stored(self, listing_id: int) -> dict | None:
        """The latest detect run's stored flags for `listing_id`, or None if it is unknown.

        A thin wrapper around the module-level `stored_flags`, so API routes deal with one
        object (the checker) rather than a checker plus a bare connection.
        """
        return stored_flags(self.conn, listing_id)

    def check(self, request) -> dict:
        photo_ids = list(dict.fromkeys(_field(request, "photo_ids") or []))
        self._validate_photos(photo_ids)
        notes = [] if photo_ids else [NO_PHOTOS_NOTE]

        text = f"{request.title}\n{request.description}"
        text_vec = np.asarray(self.embedder.embed_texts([text])[0], dtype=np.float64)
        image_vec = self._image_vector(photo_ids)
        candidates = self._candidates(text_vec, image_vec, photo_ids)
        features = self._features(request, candidates, text_vec, image_vec, photo_ids)
        scores = self.pair_model.scores(features) if features.height else np.zeros(0)

        bait, bait_notes, estimate = self._bait(request)
        notes += bait_notes
        flags = [
            flag
            for flag in (bait, self._photo_reuse(photo_ids), self._relist(request, features))
            if flag is not None
        ]
        price_version = getattr(estimate, "model_version", None) if estimate else None
        return {
            "duplicates": self._duplicates(features, scores),
            "flags": flags,
            "notes": notes,
            "model_versions": {"pair": self.pair_model.version, "price": price_version or None},
        }


def stored_flags(conn, listing_id: int) -> dict | None:
    """The latest detect run's duplicates and fraud flags for a stored listing, or None."""
    try:
        with conn.cursor() as cur:
            cur.execute(LISTING_EXISTS_SQL, (listing_id,))
            if cur.fetchone() is None:
                return None
            cur.execute(LATEST_RUN_SQL)
            (run,) = cur.fetchone()
            result = {"listing_id": listing_id, "detect_run_id": run, "duplicates": [], "flags": []}
            if run is None:
                return result
            params = {"id": listing_id, "run": run}
            cur.execute(STORED_PAIRS_SQL, params)
            result["duplicates"] = [
                {"listing_id": other, "score": float(score), "signals": signals}
                for other, score, signals in cur.fetchall()
            ]
            cur.execute(STORED_FLAGS_SQL, params)
            result["flags"] = [{"flag": flag, "detail": detail} for flag, detail in cur.fetchall()]
            return result
    finally:
        conn.rollback()
