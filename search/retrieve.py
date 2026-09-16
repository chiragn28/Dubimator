"""Candidate retrieval: a semantic channel (pgvector over the Phase 4 MiniLM text vectors) and a
full-text channel (Postgres FTS), filtered by the parsed area and type, fused with reciprocal
rank fusion, then collapsed to one listing per detected duplicate cluster."""

import re
from dataclasses import dataclass
from datetime import date

import polars as pl

from listings.load import vector_literal
from search.config import KIND_SQL, SearchConfig
from search.parse import ParsedQuery

CANDIDATE_SCHEMA = {
    "listing_id": pl.Int64,
    "semantic_cos": pl.Float64,
    "semantic_pos": pl.Int64,
    "fulltext_rank": pl.Float64,
    "fulltext_pos": pl.Int64,
    "rrf_score": pl.Float64,
    "fused_pos": pl.Int64,
    "cluster_size": pl.Int64,
}
FUSED_SCHEMA = {
    name: dtype
    for name, dtype in CANDIDATE_SCHEMA.items()
    if name not in ("fused_pos", "cluster_size")
}
_WORD = re.compile(r"[a-z0-9]+")

SEMANTIC_SQL = """
SELECT e.listing_id, 1 - (e.text_embedding <=> %s::vector) AS cos
FROM listings.listing_embeddings AS e
JOIN listings.listings AS l USING (listing_id)
WHERE {filters}
ORDER BY e.text_embedding <=> %s::vector
LIMIT %s
"""
FULLTEXT_SQL = """
SELECT l.listing_id, ts_rank_cd(l.search_tsv, q) AS rank
FROM listings.listings AS l, websearch_to_tsquery('english', %s) AS q
WHERE l.search_tsv @@ q AND {filters}
ORDER BY rank DESC, l.listing_id
LIMIT %s
"""
LATEST_PAIRS_SQL = """
SELECT listing_a, listing_b FROM listings.duplicate_pairs
WHERE decision AND detect_run_id = %s
"""


@dataclass(frozen=True)
class DuplicateClusters:
    cluster_of: dict[int, int]  # member listing_id -> cluster id (its smallest member)
    size: dict[int, int]  # cluster id -> member count
    posted: dict[int, date]  # member listing_id -> posted_at

    def cluster_size(self, listing_id: int) -> int:
        cluster = self.cluster_of.get(listing_id)
        return 1 if cluster is None else self.size[cluster]


def _find(parent: dict[int, int], node: int) -> int:
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def load_clusters(conn) -> DuplicateClusters:
    """Connected components of the latest Phase 4 detect run's flagged pairs."""
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('listings.detect_runs')")
        exists = cur.fetchone()[0] is not None
        run_id = None
        if exists:
            cur.execute("SELECT max(detect_run_id) FROM listings.detect_runs")
            run_id = cur.fetchone()[0]
        if run_id is None:
            raise RuntimeError(
                "no Phase 4 detection results — run `python -m listings detect` first"
            )
        cur.execute(LATEST_PAIRS_SQL, (run_id,))
        pairs = cur.fetchall()
        members = sorted({listing for pair in pairs for listing in pair})
        cur.execute(
            "SELECT listing_id, posted_at FROM listings.listings WHERE listing_id = ANY(%s)",
            (members,),
        )
        posted = dict(cur.fetchall())
    parent = {listing: listing for listing in members}
    for a, b in pairs:
        root_a, root_b = _find(parent, a), _find(parent, b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)
    cluster_of = {listing: _find(parent, listing) for listing in members}
    size: dict[int, int] = {}
    for cluster in cluster_of.values():
        size[cluster] = size.get(cluster, 0) + 1
    return DuplicateClusters(cluster_of, size, posted)


def _filters(parsed: ParsedQuery) -> tuple[str, list]:
    clauses, params = [], []
    if parsed.area_ids:
        clauses.append("l.area_id = ANY(%s)")
        params.append(list(parsed.area_ids))
    if parsed.property_type:
        clauses.append(f"{KIND_SQL} = %s")
        params.append(parsed.property_type)
    return (" AND ".join(clauses) or "TRUE"), params


def fulltext_terms(parsed: ParsedQuery, text: str) -> str:
    """OR-joined words: the free text and amenities, else every word of the query."""
    words = _WORD.findall(" ".join([parsed.free_text, *parsed.amenities]).lower())
    if not words:
        words = _WORD.findall((text or "").lower())
    unique = [word for word in dict.fromkeys(words) if len(word) > 1 and word not in ("or", "and")]
    return " or ".join(unique)


def semantic_channel(cur, vector, parsed: ParsedQuery, config: SearchConfig):
    where, params = _filters(parsed)
    literal = vector_literal(vector)
    cur.execute("SET LOCAL hnsw.ef_search = %s", (config.ef_search,))
    cur.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
    cur.execute(SEMANTIC_SQL.format(filters=where), [literal, *params, literal, config.semantic_k])
    # relaxed_order may return rows slightly out of order: restore it
    return sorted(cur.fetchall(), key=lambda row: (-row[1], row[0]))


def fulltext_channel(cur, terms: str, parsed: ParsedQuery, config: SearchConfig):
    if not terms:
        return []
    where, params = _filters(parsed)
    cur.execute(FULLTEXT_SQL.format(filters=where), [terms, *params, config.fulltext_k])
    return cur.fetchall()


def fuse(semantic, fulltext, rrf_k: int) -> pl.DataFrame:
    rows: dict[int, dict] = {}

    def row_for(listing_id: int) -> dict:
        return rows.setdefault(
            listing_id,
            {
                "listing_id": listing_id,
                "semantic_cos": None,
                "semantic_pos": None,
                "fulltext_rank": 0.0,
                "fulltext_pos": None,
                "rrf_score": 0.0,
            },
        )

    for position, (listing_id, cosine) in enumerate(semantic, start=1):
        row = row_for(listing_id)
        row.update(semantic_cos=float(cosine), semantic_pos=position)
        row["rrf_score"] += 1.0 / (rrf_k + position)
    for position, (listing_id, rank) in enumerate(fulltext, start=1):
        row = row_for(listing_id)
        row.update(fulltext_rank=float(rank), fulltext_pos=position)
        row["rrf_score"] += 1.0 / (rrf_k + position)
    frame = pl.DataFrame(list(rows.values()), schema=FUSED_SCHEMA)
    return frame.sort(["rrf_score", "listing_id"], descending=[True, False])


def collapse(fused: pl.DataFrame, clusters: DuplicateClusters, k: int) -> pl.DataFrame:
    """One listing per cluster: the earliest-posted member that was actually retrieved."""
    rows = fused.to_dicts()
    best: dict[int, tuple[tuple, int]] = {}
    for row in rows:
        listing_id = row["listing_id"]
        cluster = clusters.cluster_of.get(listing_id, listing_id)
        key = (clusters.posted.get(listing_id, date.max), listing_id)
        if cluster not in best or key < best[cluster][0]:
            best[cluster] = (key, listing_id)
    keep = {listing_id for _, listing_id in best.values()}
    kept = [row for row in rows if row["listing_id"] in keep][:k]
    for position, row in enumerate(kept, start=1):
        row["fused_pos"] = position
        row["cluster_size"] = clusters.cluster_size(row["listing_id"])
    return pl.DataFrame(kept, schema=CANDIDATE_SCHEMA)


def retrieve(
    conn, parsed: ParsedQuery, text: str, vector, clusters: DuplicateClusters, config: SearchConfig
) -> pl.DataFrame:
    if parsed.is_empty:
        return pl.DataFrame(schema=CANDIDATE_SCHEMA)
    with conn.cursor() as cur:
        semantic = semantic_channel(cur, vector, parsed, config)
        fulltext = fulltext_channel(cur, fulltext_terms(parsed, text), parsed, config)
    return collapse(fuse(semantic, fulltext, config.rrf_k), clusters, config.candidate_k)
