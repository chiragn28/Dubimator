"""Search one query end to end: parse, retrieve, collapse, build features, rank, explain.

Phase 6's API holds a SearchEngine; `search()` is the spec's convenience entry point.
"""

import math
import time
from dataclasses import dataclass

import polars as pl

from listings.embed import SentenceTransformerEmbedder, resolve_device
from listings.fraud import resolve_price_model_version
from search.config import SearchConfig
from search.features import (
    build_features,
    load_listing_attributes,
    load_predicted_flags,
    query_frame,
    reference_date,
)
from search.lexicon import load_lexicon
from search.parse import ParsedQuery, parse
from search.ranker import load_champion
from search.retrieve import load_clusters, retrieve
from search.store import read_estimates

FALLBACK = "fallback_fused"
EMPTY_NOTE = "empty query: add an area, a budget, a property type or a few words"
NO_MATCH_NOTE = "no listings match; try widening the budget or removing a filter"
FALLBACK_NOTE = "no ranking model is registered; showing retrieval order"
ERROR_NOTES = {
    "budget_min_exceeds_max": "the minimum budget is above the maximum, so the budget was ignored"
}
FLAG_REASONS = (
    ("flag_bait_price", "flagged: bait price"),
    ("flag_photo_reuse", "flagged: photo reuse"),
    ("flag_inconsistent_relist", "flagged: inconsistent relist"),
)
VALUE_BAND = 0.05
_LOAD_CHAMPION = object()


@dataclass(frozen=True)
class Hit:
    listing_id: int
    score: float
    title: str
    area_name: str | None
    bedrooms: int | None
    asking_price_aed: float
    size_sqm: float
    reasons: tuple[str, ...]
    duplicates_hidden: int


@dataclass(frozen=True)
class SearchResult:
    parsed: ParsedQuery
    results: tuple[Hit, ...]
    ranker: str
    notes: tuple[str, ...]
    timings_ms: dict[str, float]


def _finite(value) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def _bedrooms(count: int) -> str:
    return "studio" if count == 0 else f"{count} bedroom{'' if count == 1 else 's'}"


def reasons(parsed: ParsedQuery, row: dict) -> tuple[str, ...]:
    out = []
    if parsed.area_ids and parsed.area_name:
        out.append("area ✓" if row["area_match"] == 1.0 else "different area")
    if parsed.property_type:
        out.append("type ✓" if row["type_match"] == 1.0 else "different type")
    if parsed.building and row["building_match"] == 1.0:
        out.append(f"{parsed.building} ✓")
    if parsed.bedrooms is not None:
        if not _finite(row["beds_diff"]) or row["bedrooms"] is None:
            out.append("bedrooms not listed")
        elif row["beds_diff"] == 0:
            out.append(f"{_bedrooms(parsed.bedrooms)} ✓")
        else:
            out.append(f"{_bedrooms(int(row['bedrooms']))} (asked {parsed.bedrooms})")
    if parsed.budget_max is not None and _finite(row["price_over_max"]):
        over = row["price_over_max"]
        out.append(f"{over:.0%} over budget" if over > 0 else "within budget")
    if (
        parsed.budget_min is not None
        and _finite(row["price_under_min"])
        and row["price_under_min"] > 0
    ):
        out.append(f"{row['price_under_min']:.0%} under your minimum")
    if parsed.min_size_sqm is not None and _finite(row["size_ratio"]):
        ratio = row["size_ratio"]
        out.append("size ✓" if ratio >= 1 else f"{1 - ratio:.0%} smaller than asked")
    description = (row.get("description") or "").lower()
    for amenity in parsed.amenities:
        out.append(f"{amenity} ✓" if amenity.lower() in description else f"no {amenity}")
    value = row["price_to_estimate"]
    if _finite(value) and value < 1 - VALUE_BAND:
        out.append(f"priced {1 - value:.0%} below estimate")
    elif _finite(value) and value > 1 + VALUE_BAND:
        out.append(f"priced {value - 1:.0%} above estimate")
    out.extend(text for column, text in FLAG_REASONS if row[column] == 1.0)
    return tuple(out)


def query_notes(parsed: ParsedQuery) -> list[str]:
    notes = []
    for kind, text in parsed.unrecognised:
        label = "area not recognised" if kind == "place" else "property type not listed"
        notes.append(f"{label}: {text}")
    notes.extend(ERROR_NOTES.get(error, error) for error in parsed.errors)
    return notes


class SearchEngine:
    """One engine holds one database connection, and loads the lexicon, clusters, attributes,
    flags, estimates and ranker once, at construction time. Reuse one long-lived engine per
    worker process rather than building a new one per request.

    Not safe for concurrent use from multiple threads: it shares one psycopg2 connection.
    Serialize access to a shared engine, or give each worker its own engine (and connection).
    """

    def __init__(
        self,
        conn,
        embedder,
        config: SearchConfig = SearchConfig(),  # noqa: B008 — frozen dataclass, safe as a default
        ranker=_LOAD_CHAMPION,
    ):
        self.conn = conn
        self.embedder = embedder
        self.config = config
        self.lexicon = load_lexicon(conn)
        self.clusters = load_clusters(conn)
        self.attributes = load_listing_attributes(conn)
        self.flags = load_predicted_flags(conn)
        self.estimates = read_estimates(conn)
        self.reference = reference_date(self.attributes)
        if ranker is _LOAD_CHAMPION:
            self.ranker = load_champion(config.ranker_uri)
            version = (
                resolve_price_model_version(config.ranker_uri) if self.ranker is not None else None
            )
            self.ranker_label = f"{config.ranker_name}/v{version}" if self.ranker else FALLBACK
        else:
            self.ranker = ranker
            self.ranker_label = f"{ranker.kind} (injected)" if ranker is not None else FALLBACK

    def search(self, text: str, k: int = 10) -> SearchResult:
        started = time.perf_counter()
        parsed = parse(text, self.lexicon)
        notes = query_notes(parsed)
        if self.ranker is None:
            notes.append(FALLBACK_NOTE)
        timings = {"parse": (time.perf_counter() - started) * 1000}
        if parsed.is_empty:
            return self._result(parsed, (), [EMPTY_NOTE, *notes], timings, started)

        step = time.perf_counter()
        vector = self.embedder.embed_texts([text])[0]
        candidates = retrieve(self.conn, parsed, text, vector, self.clusters, self.config)
        timings["retrieve"] = (time.perf_counter() - step) * 1000
        if candidates.height == 0:
            return self._result(parsed, (), [*notes, NO_MATCH_NOTE], timings, started)

        step = time.perf_counter()
        frame = build_features(
            candidates.with_columns(pl.lit(0, dtype=pl.Int64).alias("query_id")),
            query_frame({0: parsed}),
            self.attributes,
            self.flags,
            self.estimates,
            self.reference,
        )
        scores = (
            -candidates["fused_pos"].to_numpy().astype(float)
            if self.ranker is None
            else self.ranker.score(frame)
        )
        ranked = (
            frame.with_columns(
                pl.Series("score", scores, dtype=pl.Float64), candidates["fused_pos"]
            )
            .sort(["score", "fused_pos"], descending=[True, False])
            .head(k)
            .join(
                self.attributes.select(
                    "listing_id",
                    "title",
                    "area_name",
                    "bedrooms",
                    "asking_price_aed",
                    "size_sqm",
                    "description",
                ),
                on="listing_id",
                how="left",
                maintain_order="left",
            )
        )
        hits = tuple(
            Hit(
                listing_id=row["listing_id"],
                score=float(row["score"]),
                title=row["title"],
                area_name=row["area_name"],
                bedrooms=row["bedrooms"],
                asking_price_aed=row["asking_price_aed"],
                size_sqm=row["size_sqm"],
                reasons=reasons(parsed, row),
                duplicates_hidden=int(row["cluster_size"]) - 1,
            )
            for row in ranked.iter_rows(named=True)
        )
        timings["rank"] = (time.perf_counter() - step) * 1000
        return self._result(parsed, hits, notes, timings, started)

    def _result(self, parsed, hits, notes, timings, started) -> SearchResult:
        timings.setdefault("retrieve", 0.0)
        timings.setdefault("rank", 0.0)
        timings["total"] = (time.perf_counter() - started) * 1000
        return SearchResult(parsed, tuple(hits), self.ranker_label, tuple(notes), timings)


_ENGINES: dict[int, SearchEngine] = {}


def search(conn, text: str, k: int = 10) -> SearchResult:
    """One engine per connection object, with the real MiniLM embedder on the best device.

    The `_ENGINES` cache is keyed by connection object and never evicts: every entry keeps its
    connection, its loaded data and its own MiniLM embedder alive for as long as the process
    runs. Pass the same long-lived connection on every call — never a fresh per-request
    connection, which would leak an engine (and a connection) on each call.
    """
    engine = _ENGINES.get(id(conn))
    if engine is None:
        embedder = SentenceTransformerEmbedder(device=resolve_device("auto"))
        engine = _ENGINES[id(conn)] = SearchEngine(conn, embedder)
    return engine.search(text, k)
