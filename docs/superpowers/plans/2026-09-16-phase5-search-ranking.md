# Phase 5 — Property Search Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Search the Phase 4 listings corpus from free text ("2BR in Dubai Marina under 1.5M"). A rule parser reads the query; semantic and full-text retrieval are fused; Phase 4 duplicates are collapsed; and a learned GBDT ranker orders the results. The ranker is measured against rule-graded synthetic queries and three baselines.

**Architecture:** A new top-level package `search/`, one module per job, following `listings/`:
- **Pipeline:** config → store (schema `search`) → lexicon + parse → queries + grade → retrieve → features → train (with evaluate) → engine.
- **CLI:** everything is driven by `python -m search queries|train|evaluate|query`.
- **Embeddings:** query embedding goes through Phase 4's `Embedder` protocol, so tests inject `FakeEmbedder`.

**Tech Stack:** Python 3.11, Polars, psycopg2, pgvector 0.8.6 (HNSW + iterative scan), Postgres full-text search, sentence-transformers all-MiniLM-L6-v2 (CUDA), XGBoost 3.2 `rank:ndcg` (CUDA), LightGBM 4.7 `lambdarank`, Optuna 5, MLflow 2.17.2, pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-phase5-search-ranking-design.md`

## Global Constraints

**Corpus and splits**
- Default corpus is 6,000 queries. Kind shares: `specified` 0.70, `vague` 0.20, `no_match` 0.10.
- 30 phrasing templates. Splits are assigned **by template id**: train 60%, tune 20%, report 20%.
- No template id and no exact query text may appear in two splits.

**Grades**
- Grades are 0–3.
  - **3:** every stated slot holds.
  - **2:** exactly one near miss (bedrooms ±1; price ≤10% over max or ≤10% under min; size ≤10% under the minimum; one of two amenities missing), with every other slot exact.
  - **1:** area and type hold where stated.
  - **0:** otherwise.
- A wrong area is never a near miss.
- Listings with a non-null ground-truth `fraud_label` are capped at grade 1.

**Retrieval**
- Semantic channel: top 200, `hnsw.ef_search = 400`, `hnsw.iterative_scan = relaxed_order`.
- Full-text channel: top 200, `ts_rank_cd` over the generated column `listings.listings.search_tsv` (config `english`, title and description), GIN index `listings_search_tsv_idx`.
- Fusion: RRF with k = 60, keeping 200 candidates.
- Parsed area and type are hard filters. Bedrooms, budget and size are never filtered.

**Features and label isolation**
- The 25 features, in this exact order (`search.config.FEATURES`):
  `beds_diff, beds_stated, price_over_max, price_under_min, budget_stated, size_ratio, area_match, area_stated, type_match, type_stated, building_match, amenity_hits, amenity_asked, semantic_cos, fulltext_rank, semantic_pos, fulltext_pos, rrf_score, price_to_estimate, within_interval, flag_bait_price, flag_photo_reuse, flag_inconsistent_relist, cluster_size, days_since_posted`.
- Features are computed from the **parsed** query, never from `true_slots`.
- `fraud_label`, `dup_group_id`, `control_group_id`, `true_slots` and `is_synthetic` may appear only in `search/grade.py`, `search/queries.py`, `search/evaluate.py` and `search/store.py`. `store.py` writes the query set, and its `read_queries()` leaves `true_slots` out. A test scans every other `search/*.py` file for them.

**Rankers and registration**
- Champion candidate: XGBoost `objective="rank:ndcg"`, `eval_metric="ndcg@10"`, `lambdarank_pair_method="topk"`, `device="cuda"` in the real run.
- Challenger: LightGBM `objective="lambdarank"`, `metric="ndcg"`, `eval_at=[10]`.
- Tuning: Optuna, 40 trials per model, TPE seeded, early stopping after 50 rounds on the tune split.
- The report split is never used to fit, tune or select a model.
- Registration gate:
  - The winner (higher tune NDCG@10) is registered as `zestimator-search-ranker` with alias `champion`.
  - This happens only if its report NDCG@10 exceeds `baseline_fused`'s **and** its bootstrap 95% CI lower bound exceeds `baseline_fused`'s point estimate.
  - Otherwise `gate.passed = 0` and nothing is registered.

**Metrics and MLflow**
- Metric definitions:
  - `ndcg_at_10` uses gains `2^grade − 1`.
  - `mrr` is the reciprocal rank of the first result with grade ≥ 2 (0 when there is none).
  - `precision_at_5` is the share of the top 5 with grade 3.
  - All three are averaged over report queries of kind `specified`/`vague` that have at least one candidate with grade ≥ 1.
  - For `no_match` queries, the metric is `kind.no_match.mean_grade_top10`.
- Bootstrap: 1,000 resamples over queries with a fixed seed.
- Undefined values are NaN. `log_run` drops non-finite values.
- MLflow experiment `search-ranking`.

**Environment and machine hazards**
- `load_dotenv()` runs before any settings are read. `DbSettings.from_env()` raises without `POSTGRES_PORT`; the project DB is on 5433.
- NEVER connect to port 5432 (a native Windows Postgres).
- Never run `docker compose down -v`.
- Windows/LightGBM hazard: pyarrow ships an older msvcp140.dll.
  - `search/__init__.py` must import `models.price` first; that package preloads the system DLL.
  - Never import pandas/pyarrow in a module that can load before `search/__init__.py`.
  - `tests/conftest.py` already imports `models.price` first. Do not reorder those imports.

**Tests**
- Tests never download a model and never need a GPU. They inject `listings.embed.FakeEmbedder` and force `device="cpu"`.
- Every MLflow-touching test uses the `temp_mlflow` fixture (copied into `tests/search/conftest.py`). Only the real Task 10 run writes to the server at 127.0.0.1:5000.
- Tests run **sequentially** (one pytest process at a time; they share the `zestimator_test` database). Never background your own test run.
- Test layout: no `__init__.py` under `tests/`. Files are `tests/search/test_search_<topic>.py`, and shared helpers are fixtures in `tests/search/conftest.py`.

**Git and commands**
- Work directly on `master`. Stage specific files only; never `git add -A`. Never commit `.env`, `data/`, `mlruns/`.
- Add `data/search/` to `.gitignore` in Task 1.
- Commit trailer: `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- Run from the repo root `C:\Users\cnaya\OneDrive\Desktop\zestimator` in Git Bash, always through `uv run`.
- At the end of every task:
  1. Run `uv run ruff format .`.
  2. `uv run ruff check . && uv run ruff format --check .` must then be clean.
  3. `uv run pytest -q -W error tests/search` must pass.

## Controller rulings made while planning

1. **Judgments store the retrieval signals.**
   - `search.judgments` carries `semantic_cos, semantic_pos, fulltext_rank, fulltext_pos, rrf_score, fused_pos, cluster_size` next to `grade`.
   - Training and evaluation therefore never re-run retrieval for 6,000 queries.
   - This is a spec amendment, and Task 10 records it.
2. **Duplicate-collapse representative.**
   - The representative is the earliest-posted member **among the retrieved members** of a cluster (ties go to the lowest `listing_id`), not the earliest member overall.
   - Its retrieval signals and features are then its own, never borrowed from a sibling.
   - This is a spec amendment.
3. **`ParsedQuery.unrecognised` is a tuple of `(kind, text)` pairs**, with kind `"place"` or `"type"`. The engine can then say "area not recognised: …" or "property type not listed: land". This is a spec amendment.
4. **Corpus pinning.**
   - `search.queries` and `search.listing_estimates` carry `corpus_run_id`.
   - `train`/`evaluate` refuse to run when the queries were built on an older corpus.
   - `queries` recomputes estimates when theirs are stale.
5. **The search test corpus is hand-built** (`tests/search/conftest.py`, 240 listings over six real Dubai area names), not produced by the Phase 4 generator.
   - It needs no photo archive, loads in about a second, and makes grades hand-checkable.
   - Phase 4's fixtures live in `tests/listings/conftest.py` and are not visible to `tests/search/` anyway.
6. **`search/engine.py` exposes a `SearchEngine` class** (caches: lexicon, clusters, attributes, flags, estimates, ranker) **plus the spec's module-level `search(conn, text, k=10)`**, which builds one real-embedder engine per process. Phase 6 will hold a `SearchEngine`.
7. **`train` fits, evaluates, gates and logs in one MLflow run.**
   - `evaluate` re-scores the currently registered champion and the three baselines on the report split in a separate run, so results can be reproduced without refitting.
   - The "Phase 4 effects" and the no-trust ablation are logged by `train`, which has both fitted models in hand.
8. **The 30 templates are frames:** 6 prefixes × 5 slot orders. A frame id also picks the phrasing style for bedrooms and budget, so a held-out frame differs in word order, opening words and phrasing family. Per-query randomness (alias vs official name, number format) is independent of the frame.

## File map

| File | Responsibility | Task |
|---|---|---|
| `search/__init__.py`, `search/config.py`, `search/sql/schema.sql`, `search/store.py`, `.gitignore`, `tests/search/conftest.py` | package, constants, schema, Postgres I/O, seeded test corpus | 1 |
| `search/lexicon.py`, `search/parse.py` | place lexicon; pure query parser | 2 |
| `search/grade.py`, `search/queries.py` | grades; synthetic queries, templates, splits | 3 |
| `search/retrieve.py` | semantic + full-text channels, RRF, duplicate clusters and collapse | 4 |
| `search/features.py` | listing attributes, predicted flags, price estimates, the 25 features; leakage scan | 5 |
| `search/metrics.py` | NDCG/MRR/P@5, bootstrap CI, per-query tables (pure) | 6 |
| `search/ranker.py`, `search/train.py` | ranker fit (XGB/LGBM) + Optuna, pyfunc, registry gate | 7 |
| `search/evaluate.py` | contenders, baselines, Phase 4 effects, parser accuracy, artifacts, MLflow | 8 |
| `search/engine.py` | `SearchEngine`, `search()`, reasons, fallback | 9 |
| `search/__main__.py`, `tests/search/test_search_integration.py`, real run, `README.md`, spec amendments | CLI, end-to-end test, 6,000-query run, write-up | 10 |

(`search/metrics.py` is split out of `evaluate.py` so that the pure metric maths gets its own reviewable task. `evaluate.py` then stays the only module in that pair that reads labels.)

---

## Task 1: Package, config, schema, store, seeded test corpus

**Files:**
- Create:
  - `search/__init__.py`
  - `search/config.py`
  - `search/sql/schema.sql`
  - `search/store.py`
  - `tests/search/search_fixtures.py`
  - `tests/search/conftest.py`
  - `tests/search/test_search_store.py`
- Modify: `.gitignore`

**Interfaces:**
- **Produces, from `search.config`:**
  - Constants: `KINDS`, `SPLITS`, `QUERY_KINDS`, `SLOTS`, `FEATURES`, `TRUST_FEATURES`, `SEARCH_FORBIDDEN`, `LABEL_READERS`, `SQFT_TO_SQM`, `KIND_SQL`, `DATA_DIR`.
  - Functions: `listing_kind(property_type: str, sub_type: str | None) -> str | None`.
  - Classes: `SearchConfig` (a frozen dataclass).
- **Produces, from `search.store`:**
  - `apply_schema(conn)`
  - `latest_corpus_run(conn) -> int` (re-exported from `listings.load`)
  - `replace_query_set(conn, queries: pl.DataFrame, judgments: pl.DataFrame) -> None`
  - `read_queries(conn, splits: tuple[str, ...] | None = None) -> pl.DataFrame`
    - Columns: `query_id, text, template_id, kind, split, seed_listing_id, n_grade3, corpus_run_id`.
  - `read_judgments(conn, query_ids: list[int] | None = None) -> pl.DataFrame`
    - Columns: `query_id, listing_id, grade, semantic_cos, semantic_pos, fulltext_rank, fulltext_pos, rrf_score, fused_pos, cluster_size`.
  - `queries_corpus_run(conn) -> int | None`
  - `replace_estimates(conn, estimates: pl.DataFrame, corpus_run_id: int, version: str | None) -> int`
  - `read_estimates(conn) -> pl.DataFrame`
    - Columns: `listing_id, estimate, low, high`.
  - `estimates_corpus_run(conn) -> int | None`
  - Schemas: `QUERY_SCHEMA`, `JUDGMENT_SCHEMA`, `ESTIMATE_SCHEMA`.
- **Produces, from `tests/search/search_fixtures.py`** (a plain module; test files import it as `from search_fixtures import ...`, because `tests/search/` is on `sys.path` during collection and the basename is unique. `from conftest import` is avoided because several `conftest.py` files share that module name):
  - Constants: `AREAS`, `ALIASES`, `BUILDINGS`, `N_CLONES`, `BAIT_IDS`, `PREDICTED_BAIT_IDS`.
  - Functions:
    - `build_search_listings(n=240, seed=5) -> pl.DataFrame` (the Phase 4 `LISTING_SCHEMA`).
    - `seed_search_db(settings, listings) -> int` (returns the corpus_run_id).
- **Produces, from `tests/search/conftest.py`** (fixtures only):
    - `search_listings`
    - `search_db` → `(settings, listings, corpus_run_id)`, with the dld, listings and search schemas applied.
    - `fake_embedder`
    - `temp_mlflow`

- [ ] **Step 1: Ignore the search data dir**

Append one line to `.gitignore`, after `data/listings/`:

```
data/search/
```

- [ ] **Step 2: Write the package and config**

`search/__init__.py`:

```python
"""Property search ranking (Phase 5)."""

import models.price  # noqa: F401 — Windows DLL preload, must run before pandas/pyarrow/LightGBM
```

`search/config.py`:

```python
"""Constants and tunables for property search.

Spec: docs/superpowers/specs/2026-09-16-phase5-search-ranking-design.md
"""

from dataclasses import dataclass
from pathlib import Path

from listings.generate import SUB_KINDS

DATA_DIR = Path("data/search")
KINDS = ("flat", "hotel_apartment", "townhouse", "villa")
SPLITS = ("train", "tune", "report")
QUERY_KINDS = ("specified", "vague", "no_match")
SLOTS = (
    "area", "building", "bedrooms", "property_type",
    "budget_min", "budget_max", "min_size_sqm", "amenities",
)  # fmt: skip
SQFT_TO_SQM = 0.092903

# The ranker's inputs, in the order it sees them. Computed from the PARSED query only.
FEATURES = (
    "beds_diff", "beds_stated", "price_over_max", "price_under_min", "budget_stated",
    "size_ratio", "area_match", "area_stated", "type_match", "type_stated", "building_match",
    "amenity_hits", "amenity_asked", "semantic_cos", "fulltext_rank", "semantic_pos",
    "fulltext_pos", "rrf_score", "price_to_estimate", "within_interval", "flag_bait_price",
    "flag_photo_reuse", "flag_inconsistent_relist", "cluster_size", "days_since_posted",
)  # fmt: skip
TRUST_FEATURES = (
    "flag_bait_price", "flag_photo_reuse", "flag_inconsistent_relist", "cluster_size",
)  # fmt: skip

# Ground truth and generator internals. Only the modules in LABEL_READERS may name these.
SEARCH_FORBIDDEN = ("fraud_label", "dup_group_id", "control_group_id", "true_slots", "is_synthetic")
# store.py only persists the query set; its read_queries() leaves true_slots out.
LABEL_READERS = ("grade.py", "queries.py", "evaluate.py", "store.py")


def listing_kind(property_type: str, sub_type: str | None) -> str | None:
    """The corpus's four listing kinds, derived exactly as listings.generate._sub_kind does."""
    if property_type == "villa":
        return "villa"
    return SUB_KINDS.get(sub_type or "")


# The same derivation in SQL, for retrieval filters. Table alias must be `l`.
KIND_SQL = (
    "(CASE WHEN l.property_type = 'villa' THEN 'villa' "
    + " ".join(
        f"WHEN l.property_sub_type = '{sub_type}' THEN '{kind}'"
        for sub_type, kind in sorted(SUB_KINDS.items())
    )
    + " END)"
)


@dataclass(frozen=True)
class SearchConfig:
    n_queries: int = 6_000
    kind_shares: tuple[float, float, float] = (0.70, 0.20, 0.10)  # specified, vague, no_match
    split_shares: tuple[float, float, float] = (0.60, 0.20, 0.20)  # train, tune, report
    alias_probability: float = 0.4
    building_probability: float = 0.1
    budget_headroom: tuple[float, float] = (0.0, 0.30)
    size_slack: tuple[float, float] = (0.0, 0.20)
    near_margin: float = 0.10
    semantic_k: int = 200
    fulltext_k: int = 200
    candidate_k: int = 200
    rrf_k: int = 60
    ef_search: int = 400
    n_trials: int = 40
    early_stopping_rounds: int = 50
    max_rounds: int = 2_000
    n_bootstrap: int = 1_000
    ndcg_k: int = 10
    device: str = "cuda"
    ranker_name: str = "zestimator-search-ranker"
    ranker_uri: str = "models:/zestimator-search-ranker@champion"
    price_model_uri: str = "models:/zestimator-price@champion"
    experiment: str = "search-ranking"
    seed: int = 7
```

- [ ] **Step 3: Write the schema**

`search/sql/schema.sql`:

```sql
CREATE SCHEMA IF NOT EXISTS search;

-- Full-text channel. Generated, so `python -m listings build` (COPY with an explicit column
-- list) fills it without knowing it exists.
ALTER TABLE listings.listings
    ADD COLUMN IF NOT EXISTS search_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', title || ' ' || description)) STORED;
CREATE INDEX IF NOT EXISTS listings_search_tsv_idx
    ON listings.listings USING gin (search_tsv);

CREATE TABLE IF NOT EXISTS search.queries (
    query_id integer PRIMARY KEY,
    text text NOT NULL,
    template_id integer NOT NULL,
    kind text NOT NULL CHECK (kind IN ('specified', 'vague', 'no_match')),
    split text NOT NULL CHECK (split IN ('train', 'tune', 'report')),
    true_slots jsonb NOT NULL,
    seed_listing_id bigint NOT NULL,
    n_grade3 integer NOT NULL,
    corpus_run_id integer NOT NULL
);

-- One row per retrieved candidate. No foreign key to listings: a rebuilt corpus makes these
-- stale, which corpus_run_id on search.queries detects.
CREATE TABLE IF NOT EXISTS search.judgments (
    query_id integer NOT NULL REFERENCES search.queries (query_id),
    listing_id bigint NOT NULL,
    grade smallint NOT NULL CHECK (grade BETWEEN 0 AND 3),
    semantic_cos double precision,
    semantic_pos integer,
    fulltext_rank double precision NOT NULL,
    fulltext_pos integer,
    rrf_score double precision NOT NULL,
    fused_pos integer NOT NULL,
    cluster_size integer NOT NULL,
    PRIMARY KEY (query_id, listing_id)
);

CREATE TABLE IF NOT EXISTS search.listing_estimates (
    listing_id bigint PRIMARY KEY,
    estimate double precision NOT NULL,
    low double precision NOT NULL,
    high double precision NOT NULL,
    price_model_version text,
    corpus_run_id integer NOT NULL
);

COMMENT ON TABLE search.queries IS
    'Synthetic, rule-graded search queries (Phase 5). Not real user queries.';
```

- [ ] **Step 4: Write the seeded test corpus**

`tests/search/search_fixtures.py`:

```python
import random
from datetime import date, timedelta

import numpy as np
import polars as pl

from ingestion.normalize import match_key
from listings.generate import LISTING_SCHEMA
from listings.text import ListingFacts, make_description, make_title
from search.config import listing_kind

AREAS = (
    (1, "Dubai Marina"),
    (2, "Jumeirah Village Circle"),
    (3, "Business Bay"),
    (4, "Downtown Dubai"),
    (5, "Arabian Ranches"),
    (6, "Palm Jumeirah"),
)
ALIASES = (("JVC", 2), ("Marina", 1), ("Downtown", 4), ("The Palm", 6))
BUILDINGS = {
    1: ("Marina Gate", "Princess Tower"),
    2: ("Belgravia Heights",),
    3: ("Executive Towers", "Bay Square"),
    4: ("Burj Views", "Address Residence"),
    5: (None,),
    6: ("Shoreline Apartments",),
}
SUB_TYPES = ("Flat", "Flat", "Hotel Apartment", "Stacked Townhouses")
N_CLONES = 10
BAIT_IDS = tuple(range(11, 17))  # ground-truth bait listings
PREDICTED_BAIT_IDS = BAIT_IDS[:4]  # the "detector" only caught four of them


def build_search_listings(n: int = 240, seed: int = 5) -> pl.DataFrame:
    """Hand-built listings over six real Dubai area names. The last N_CLONES rows are exact
    reposts of listings 1..N_CLONES; listings BAIT_IDS are bait-priced (ground truth)."""
    rng = random.Random(seed)
    rows = []
    for i in range(n - N_CLONES):
        area_id, area_name = AREAS[i % len(AREAS)]
        villa = area_id == 5 or (area_id == 6 and i % 3 == 0)
        property_type = "villa" if villa else "unit"
        sub_type = None if villa else SUB_TYPES[i % len(SUB_TYPES)]
        kind = listing_kind(property_type, sub_type)
        building = None if villa else BUILDINGS[area_id][i % len(BUILDINGS[area_id])]
        bedrooms = 3 + i % 3 if villa else i % 4
        size = 250.0 + 10.0 * (i % 20) if villa else 40.0 + 25.0 * bedrooms + i % 7
        price = round((size * (9_000 + 1_500 * area_id) + (i % 11) * 20_000) / 10_000) * 10_000
        listing_id = i + 1
        fraud_label = None
        if listing_id in BAIT_IDS:
            fraud_label, price = "bait_price", round(price * 0.5 / 10_000) * 10_000
        facts = ListingFacts(
            bedrooms, size, area_name, None, building, f"Project {area_id}", property_type, kind
        )
        rows.append(
            {
                "listing_id": listing_id,
                "source_transaction_id": f"s{i}",
                "agent_id": i % 20 + 1,
                "posted_at": date(2023, 1, 1) + timedelta(days=i % 180),
                "title": make_title(rng, facts),
                "description": make_description(rng, facts),
                "asking_price_aed": float(price),
                "area_id": area_id,
                "area_name": area_name,
                "building_name": building,
                "project_name": f"Project {area_id}",
                "property_type": property_type,
                "property_sub_type": sub_type,
                "reg_type": "ready",
                "size_sqm": size,
                "bedrooms": bedrooms,
                "photo_set_id": 1,
                "dup_group_id": None,
                "control_group_id": None,
                "fraud_label": fraud_label,
                "is_synthetic": True,
            }
        )
    for offset in range(N_CLONES):
        source = dict(rows[offset])
        source.update(
            listing_id=n - N_CLONES + offset + 1,
            source_transaction_id=f"c{offset}",
            posted_at=source["posted_at"] + timedelta(days=5),
            dup_group_id=source["listing_id"],
            agent_id=source["agent_id"] % 20 + 1,
        )
        rows.append(source)
    return pl.DataFrame(rows, schema=LISTING_SCHEMA)


def _area_frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    areas = pl.DataFrame(
        {
            "area_id": [area_id for area_id, _ in AREAS],
            "name_en": [name for _, name in AREAS],
            "name_ar": [None] * len(AREAS),
            "match_key": [match_key(name) for _, name in AREAS],
            "market_sales": [100] * len(AREAS),
        },
        schema={
            "area_id": pl.Int64,
            "name_en": pl.Utf8,
            "name_ar": pl.Utf8,
            "match_key": pl.Utf8,
            "market_sales": pl.Int64,
        },
    )
    official = [(match_key(name), name, area_id, "official") for area_id, name in AREAS]
    curated = [(match_key(alias), alias, area_id, "curated") for alias, area_id in ALIASES]
    aliases = pl.DataFrame(
        official + curated,
        schema={"alias_key": pl.Utf8, "alias": pl.Utf8, "area_id": pl.Int64, "source": pl.Utf8},
        orient="row",
    )
    return areas, aliases


def seed_search_db(settings, listings: pl.DataFrame) -> int:
    """dld areas + aliases, the listings corpus with fake text vectors and HNSW indexes, one
    detect run (clone pairs flagged, four predicted bait flags), and the search schema."""
    from ingestion.load import apply_schema as apply_dld_schema
    from ingestion.load import copy_frame
    from listings.embed import IMAGE_DIM, FakeEmbedder
    from listings.load import apply_schema as apply_listings_schema
    from listings.load import (
        create_vector_indexes,
        start_corpus_run,
        vector_literal,
        write_listing_embeddings,
    )
    from search.store import apply_schema as apply_search_schema

    areas, aliases = _area_frames()
    conn = settings.connect()
    try:
        apply_dld_schema(conn)
        with conn.cursor() as cur:
            copy_frame(cur, "dld.areas", areas)
            copy_frame(cur, "dld.area_aliases", aliases)
        apply_listings_schema(conn)
        corpus_run_id = start_corpus_run(conn, 5, "test", {"listings": listings.height})
        with conn.cursor() as cur:
            copy_frame(
                cur,
                "listings.listings",
                listings.with_columns(pl.lit(corpus_run_id, dtype=pl.Int64).alias("corpus_run_id")),
            )
        texts = [f"{t}\n{d}" for t, d in zip(listings["title"], listings["description"])]
        vectors = FakeEmbedder().embed_texts(texts)
        image = np.zeros(IMAGE_DIM)
        image[0] = 1.0  # a zero vector has no cosine; any fixed unit vector will do
        write_listing_embeddings(
            conn,
            pl.DataFrame(
                {
                    "listing_id": listings["listing_id"],
                    "text_embedding": [vector_literal(v) for v in vectors],
                    "image_embedding": [vector_literal(image)] * listings.height,
                }
            ),
        )
        create_vector_indexes(conn)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO listings.detect_runs (corpus_run_id, threshold) "
                "VALUES (%s, 0.5) RETURNING detect_run_id",
                (corpus_run_id,),
            )
            (detect_run_id,) = cur.fetchone()
            clones = listings.filter(pl.col("dup_group_id").is_not_null())
            for clone_id, source_id in clones.select("listing_id", "dup_group_id").iter_rows():
                cur.execute(
                    "INSERT INTO listings.duplicate_pairs "
                    "VALUES (%s, %s, %s, 0.99, true, '{}'::jsonb)",
                    (detect_run_id, min(clone_id, source_id), max(clone_id, source_id)),
                )
            for listing_id in PREDICTED_BAIT_IDS:
                cur.execute(
                    "INSERT INTO listings.fraud_flags VALUES (%s, %s, 'bait_price', '{}'::jsonb)",
                    (detect_run_id, listing_id),
                )
        apply_search_schema(conn)
        conn.commit()
    finally:
        conn.close()
    return corpus_run_id
```

`tests/search/conftest.py`:

```python
import pytest
from search_fixtures import build_search_listings, seed_search_db


@pytest.fixture
def search_listings():
    return build_search_listings()


@pytest.fixture
def search_db(pg_test_db, search_listings):
    corpus_run_id = seed_search_db(pg_test_db, search_listings)
    return pg_test_db, search_listings, corpus_run_id


@pytest.fixture
def fake_embedder():
    from listings.embed import FakeEmbedder

    return FakeEmbedder()


@pytest.fixture
def temp_mlflow(tmp_path, monkeypatch):
    """Throwaway MLflow tracking + registry store. Never the real server, never ./mlruns."""
    import mlflow

    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr("mlflow.tracking._tracking_service.utils._tracking_uri", uri)
    monkeypatch.setattr("mlflow.tracking.fluent._active_experiment_id", None)
    yield {"tracking_uri": uri, "artifact_location": (tmp_path / "artifacts").as_uri()}
    while mlflow.active_run():
        mlflow.end_run()
```

- [ ] **Step 5: Write the failing tests**

`tests/search/test_search_store.py`:

```python
import polars as pl
import pytest

from listings.generate import SUB_KINDS
from search.config import FEATURES, KIND_SQL, KINDS, TRUST_FEATURES, listing_kind
from search.store import (
    ESTIMATE_SCHEMA,
    JUDGMENT_SCHEMA,
    QUERY_SCHEMA,
    apply_schema,
    estimates_corpus_run,
    queries_corpus_run,
    read_estimates,
    read_judgments,
    read_queries,
    replace_estimates,
    replace_query_set,
)


def test_features_are_the_25_in_order_and_trust_is_a_subset():
    assert len(FEATURES) == 25 and len(set(FEATURES)) == 25
    assert FEATURES[0] == "beds_diff" and FEATURES[-1] == "days_since_posted"
    assert set(TRUST_FEATURES) <= set(FEATURES)


@pytest.mark.parametrize(
    ("property_type", "sub_type", "kind"),
    [
        ("villa", None, "villa"),
        ("unit", "Flat", "flat"),
        ("unit", "Hotel Apartment", "hotel_apartment"),
        ("unit", "Stacked Townhouses", "townhouse"),
        ("unit", "Office", None),
    ],
)
def test_listing_kind(property_type, sub_type, kind):
    assert listing_kind(property_type, sub_type) == kind


def test_kind_sql_names_every_sub_type_and_kind():
    for sub_type, kind in SUB_KINDS.items():
        assert f"'{sub_type}'" in KIND_SQL and f"'{kind}'" in KIND_SQL
    assert set(SUB_KINDS.values()) | {"villa"} == set(KINDS)


def _queries(corpus_run_id: int, ids=(1, 2)) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "query_id": list(ids),
            "text": [f"2BR in Dubai Marina #{i}" for i in ids],
            "template_id": [0] * len(ids),
            "kind": ["specified"] * len(ids),
            "split": ["train"] * len(ids),
            "true_slots": ['{"bedrooms": 2}'] * len(ids),
            "seed_listing_id": [1] * len(ids),
            "n_grade3": [3] * len(ids),
            "corpus_run_id": [corpus_run_id] * len(ids),
        },
        schema=QUERY_SCHEMA,
    )


def _judgments(ids=(1, 2)) -> pl.DataFrame:
    rows = [
        (qid, listing, 3 - pos, 0.9, pos + 1, 0.1, None, 0.03, pos + 1, 1)
        for qid in ids
        for pos, listing in enumerate((5, 6, 7))
    ]
    return pl.DataFrame(rows, schema=JUDGMENT_SCHEMA, orient="row")


def test_schema_is_idempotent_and_fills_the_full_text_column(search_db):
    settings, listings, _ = search_db
    conn = settings.connect()
    try:
        apply_schema(conn)
        apply_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM listings.listings "
                "WHERE search_tsv @@ to_tsquery('english', 'marina')"
            )
            (hits,) = cur.fetchone()
            cur.execute("SELECT indexname FROM pg_indexes WHERE schemaname = 'listings'")
            indexes = {row[0] for row in cur.fetchall()}
    finally:
        conn.close()
    assert hits >= listings.filter(pl.col("area_id") == 1).height
    assert "listings_search_tsv_idx" in indexes


def test_query_set_round_trips_and_is_replaced_whole(search_db):
    settings, _, corpus_run_id = search_db
    conn = settings.connect()
    try:
        replace_query_set(conn, _queries(corpus_run_id), _judgments())
        conn.commit()
        replace_query_set(conn, _queries(corpus_run_id, ids=(7,)), _judgments(ids=(7,)))
        conn.commit()
        queries, judgments = read_queries(conn), read_judgments(conn)
        run = queries_corpus_run(conn)
    finally:
        conn.close()
    assert queries["query_id"].to_list() == [7]
    assert "true_slots" not in queries.columns
    assert judgments.height == 3 and judgments["grade"].to_list() == [3, 2, 1]
    assert judgments["fulltext_pos"].null_count() == 3
    assert run == corpus_run_id


def test_read_queries_filters_by_split(search_db):
    settings, _, corpus_run_id = search_db
    frame = _queries(corpus_run_id).with_columns(
        pl.Series("split", ["train", "report"])
    )
    conn = settings.connect()
    try:
        replace_query_set(conn, frame, _judgments())
        conn.commit()
        report = read_queries(conn, splits=("report",))
        some = read_judgments(conn, query_ids=[2])
    finally:
        conn.close()
    assert report["query_id"].to_list() == [2]
    assert set(some["query_id"].to_list()) == {2}


def test_estimates_round_trip_with_their_corpus_run(search_db):
    settings, _, corpus_run_id = search_db
    estimates = pl.DataFrame(
        {"listing_id": [1, 2], "estimate": [1e6, 2e6], "low": [9e5, 1.8e6], "high": [1.1e6, 2.2e6]},
        schema=ESTIMATE_SCHEMA,
    )
    conn = settings.connect()
    try:
        assert estimates_corpus_run(conn) is None
        written = replace_estimates(conn, estimates, corpus_run_id, "2")
        conn.commit()
        back, run = read_estimates(conn), estimates_corpus_run(conn)
    finally:
        conn.close()
    assert written == 2 and run == corpus_run_id
    assert back.sort("listing_id")["estimate"].to_list() == [1e6, 2e6]
```

- [ ] **Step 6: Run the tests to verify they fail**

Run: `uv run pytest -q tests/search/test_search_store.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'search.store'`.

- [ ] **Step 7: Write the store**

`search/store.py`:

```python
"""Postgres I/O for schema `search`. Callers own the transaction."""

from pathlib import Path

import polars as pl

from ingestion.load import LoadInvariantError, copy_frame
from listings.load import latest_corpus_run  # noqa: F401 — re-exported for search modules

SCHEMA_SQL = Path(__file__).parent / "sql" / "schema.sql"
QUERY_SCHEMA = {
    "query_id": pl.Int64,
    "text": pl.Utf8,
    "template_id": pl.Int64,
    "kind": pl.Utf8,
    "split": pl.Utf8,
    "true_slots": pl.Utf8,  # JSON text; COPY casts it to jsonb
    "seed_listing_id": pl.Int64,
    "n_grade3": pl.Int64,
    "corpus_run_id": pl.Int64,
}
QUERY_COLUMNS = tuple(name for name in QUERY_SCHEMA if name != "true_slots")
JUDGMENT_SCHEMA = {
    "query_id": pl.Int64,
    "listing_id": pl.Int64,
    "grade": pl.Int64,
    "semantic_cos": pl.Float64,
    "semantic_pos": pl.Int64,
    "fulltext_rank": pl.Float64,
    "fulltext_pos": pl.Int64,
    "rrf_score": pl.Float64,
    "fused_pos": pl.Int64,
    "cluster_size": pl.Int64,
}
ESTIMATE_SCHEMA = {
    "listing_id": pl.Int64,
    "estimate": pl.Float64,
    "low": pl.Float64,
    "high": pl.Float64,
}
LOCK_TIMEOUT = "60s"


def apply_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL.read_text(encoding="utf-8"))


def _count(cur, table: str) -> int:
    cur.execute(f"SELECT count(*) FROM {table}")
    return cur.fetchone()[0]


def replace_query_set(conn, queries: pl.DataFrame, judgments: pl.DataFrame) -> None:
    """Truncate search.queries and search.judgments and COPY these in, verified."""
    with conn.cursor() as cur:
        cur.execute(f"SET LOCAL lock_timeout = '{LOCK_TIMEOUT}'")
        cur.execute("TRUNCATE search.judgments, search.queries")
        copy_frame(cur, "search.queries", queries.select(list(QUERY_SCHEMA)))
        copy_frame(cur, "search.judgments", judgments.select(list(JUDGMENT_SCHEMA)))
        for table, expected in (
            ("search.queries", queries.height),
            ("search.judgments", judgments.height),
        ):
            loaded = _count(cur, table)
            if loaded != expected:
                raise LoadInvariantError(f"{table}: loaded {loaded} rows but expected {expected}")


def read_queries(conn, splits: tuple[str, ...] | None = None) -> pl.DataFrame:
    columns = ", ".join(QUERY_COLUMNS)
    sql = f"SELECT {columns} FROM search.queries"
    params: tuple = ()
    if splits is not None:
        sql += " WHERE split = ANY(%s)"
        params = (list(splits),)
    with conn.cursor() as cur:
        cur.execute(sql + " ORDER BY query_id", params)
        rows = cur.fetchall()
    schema = {name: QUERY_SCHEMA[name] for name in QUERY_COLUMNS}
    return pl.DataFrame(rows, schema=schema, orient="row")


def read_judgments(conn, query_ids: list[int] | None = None) -> pl.DataFrame:
    sql = f"SELECT {', '.join(JUDGMENT_SCHEMA)} FROM search.judgments"
    params: tuple = ()
    if query_ids is not None:
        sql += " WHERE query_id = ANY(%s)"
        params = (list(query_ids),)
    with conn.cursor() as cur:
        cur.execute(sql + " ORDER BY query_id, fused_pos", params)
        rows = cur.fetchall()
    return pl.DataFrame(rows, schema=JUDGMENT_SCHEMA, orient="row")


def queries_corpus_run(conn) -> int | None:
    with conn.cursor() as cur:
        cur.execute("SELECT max(corpus_run_id) FROM search.queries")
        return cur.fetchone()[0]


def replace_estimates(
    conn, estimates: pl.DataFrame, corpus_run_id: int, version: str | None
) -> int:
    frame = estimates.select(list(ESTIMATE_SCHEMA)).with_columns(
        pl.lit(version, dtype=pl.Utf8).alias("price_model_version"),
        pl.lit(corpus_run_id, dtype=pl.Int64).alias("corpus_run_id"),
    )
    with conn.cursor() as cur:
        cur.execute(f"SET LOCAL lock_timeout = '{LOCK_TIMEOUT}'")
        cur.execute("TRUNCATE search.listing_estimates")
        copy_frame(cur, "search.listing_estimates", frame)
        return _count(cur, "search.listing_estimates")


def read_estimates(conn) -> pl.DataFrame:
    with conn.cursor() as cur:
        cur.execute(f"SELECT {', '.join(ESTIMATE_SCHEMA)} FROM search.listing_estimates")
        return pl.DataFrame(cur.fetchall(), schema=ESTIMATE_SCHEMA, orient="row")


def estimates_corpus_run(conn) -> int | None:
    with conn.cursor() as cur:
        cur.execute("SELECT max(corpus_run_id) FROM search.listing_estimates")
        return cur.fetchone()[0]
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `uv run pytest -q -W error tests/search/test_search_store.py`
Expected: all tests PASS.

If `copy_frame` rejects `true_slots` because a Utf8 value contains a double quote, that is expected CSV quoting (Polars quotes the field), and COPY handles it. If it still fails, report the exact error; do not change the column type.

- [ ] **Step 9: Lint, full search tests, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
uv run pytest -q -W error tests/search
git add .gitignore search/__init__.py search/config.py search/sql/schema.sql search/store.py tests/search/search_fixtures.py tests/search/conftest.py tests/search/test_search_store.py
git commit -m "feat(search): package, config, schema and store with a seeded test corpus

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 2: Place lexicon and the query parser

**Files:**
- Create: `search/lexicon.py`, `search/parse.py`, `tests/search/test_search_parse.py`

**Interfaces:**
- Consumes:
  - `ingestion.normalize.match_key`
  - `listings.text.AMENITIES`
  - `search.config.SQFT_TO_SQM`
  - Fixture `search_db` (Task 1)
  - `tests/search/search_fixtures.py` constants `AREAS`, `ALIASES`, `BUILDINGS`
- Produces (`search.lexicon`):
  - `Place(kind: str, name: str, area_ids: tuple[int, ...])`, frozen. `kind` is `"area" | "building" | "project"`.
  - `Lexicon(entries: dict[str, Place], max_tokens: int)`, with:
    - `Lexicon.from_rows(areas, aliases, buildings, projects)`
    - `.match(tokens: list[str]) -> list[tuple[int, int, Place]]`
  - `load_lexicon(conn) -> Lexicon`
- Produces (`search.parse`):
  - `ParsedQuery`, a frozen dataclass. Fields: `area_ids, area_name, building, bedrooms, property_type, budget_min, budget_max, min_size_sqm, amenities, free_text, unrecognised, errors`.
    - Property `.is_empty`.
    - Method `.to_dict()`, which returns a JSON-able dict.
  - `parse(text: str, lexicon: Lexicon) -> ParsedQuery`
  - `AMENITY_SYNONYMS: dict[str, str]`, mapping a lower-case phrase to its canonical `AMENITIES` entry.

Parsing order is fixed, because each pass blanks what it consumed with `"\x00"` so later passes can't reuse it:
1. size
2. budget
3. places (lexicon, longest match first)
4. bedrooms
5. property type
6. amenities
7. unrecognised places
8. free text

`"\x00"` is also how the unrecognised-place scan knows that a preposition was followed by an already-matched place.

- [ ] **Step 1: Write the failing tests**

`tests/search/test_search_parse.py`:

```python
import pytest
from search_fixtures import ALIASES, AREAS, BUILDINGS

from search.lexicon import Lexicon, Place, load_lexicon
from search.parse import AMENITY_SYNONYMS, parse

LEXICON = Lexicon.from_rows(
    areas=AREAS,
    aliases=ALIASES,
    buildings=[(name, area) for area, names in BUILDINGS.items() for name in names if name],
    projects=[(f"Project {area}", area) for area, _ in AREAS],
)

CASES = [
    # bedrooms
    ("2BR in Dubai Marina", {"bedrooms": 2, "area_ids": (1,), "area_name": "Dubai Marina"}),
    ("2 bed flat", {"bedrooms": 2, "property_type": "flat"}),
    ("3-bedroom villa", {"bedrooms": 3, "property_type": "villa"}),
    ("two bedroom apartment", {"bedrooms": 2, "property_type": "flat"}),
    ("studio in JVC", {"bedrooms": 0, "area_ids": (2,), "area_name": "JVC"}),
    ("Studio apartment", {"bedrooms": 0, "property_type": "flat"}),
    ("4 bedrooms", {"bedrooms": 4}),
    ("1 BHK", {"bedrooms": 1}),
    ("seven bed villa", {"bedrooms": 7, "property_type": "villa"}),
    # budget
    ("under 1.5M", {"budget_max": 1_500_000.0, "budget_min": None}),
    ("below AED 900k", {"budget_max": 900_000.0}),
    ("max 2,000,000", {"budget_max": 2_000_000.0}),
    ("up to 1.2 million", {"budget_max": 1_200_000.0}),
    ("budget 850k", {"budget_max": 850_000.0}),
    ("900k-1.2M", {"budget_min": 900_000.0, "budget_max": 1_200_000.0}),
    ("1-1.5M", {"budget_min": 1_000_000.0, "budget_max": 1_500_000.0}),
    ("between 1M and 2M", {"budget_min": 1_000_000.0, "budget_max": 2_000_000.0}),
    ("from 800k to 1.1m", {"budget_min": 800_000.0, "budget_max": 1_100_000.0}),
    ("from 2M", {"budget_min": 2_000_000.0, "budget_max": None}),
    ("over AED 3 million", {"budget_min": 3_000_000.0}),
    ("AED 1.5M villa", {"budget_max": 1_500_000.0, "property_type": "villa"}),
    ("1.5m aed apartment", {"budget_max": 1_500_000.0, "property_type": "flat"}),
    ("at least 5000000", {"budget_min": 5_000_000.0}),
    ("under 5", {"budget_max": None, "budget_min": None}),
    (
        "2M-1M",
        {"budget_min": None, "budget_max": None, "errors": ("budget_min_exceeds_max",)},
    ),
    # size
    ("over 100 sqm", {"min_size_sqm": 100.0, "budget_min": None}),
    ("at least 1,200 sq ft", {"min_size_sqm": 111.48, "budget_min": None}),
    ("150+ sqm", {"min_size_sqm": 150.0}),
    ("villa 300 m2", {"min_size_sqm": 300.0, "property_type": "villa"}),
    ("min 90 square meters", {"min_size_sqm": 90.0}),
    # property type
    ("hotel apartment in Business Bay", {"property_type": "hotel_apartment", "area_ids": (3,)}),
    ("serviced apartments downtown", {"property_type": "hotel_apartment", "area_ids": (4,)}),
    ("townhouse", {"property_type": "townhouse"}),
    ("town house", {"property_type": "townhouse"}),
    ("villas in arabian ranches", {"property_type": "villa", "area_ids": (5,)}),
    ("flat", {"property_type": "flat"}),
    ("villa or apartment", {"property_type": "villa"}),
    (
        "plot in Dubai Marina",
        {"property_type": None, "area_ids": (1,), "unrecognised": (("type", "plot"),)},
    ),
    ("land", {"property_type": None, "unrecognised": (("type", "land"),)}),
    # places
    ("apartment in the palm", {"area_ids": (6,), "area_name": "The Palm"}),
    ("Jumeirah Village Circle 2 bed", {"area_ids": (2,), "bedrooms": 2}),
    ("flat marina", {"area_ids": (1,), "area_name": "Marina"}),
    ("Dubai Marina", {"area_name": "Dubai Marina"}),
    ("Downtown vs Dubai Marina", {"area_ids": (4,), "area_name": "Downtown"}),
    ("in Al Barsha", {"area_ids": (), "unrecognised": (("place", "al barsha"),), "free_text": ""}),
    ("near metro", {"unrecognised": (), "free_text": "metro"}),
    (
        "villa in Dubai Marina quiet",
        {"area_ids": (1,), "unrecognised": (), "free_text": "quiet"},
    ),
    # buildings and projects
    (
        "flat at Marina Gate",
        {"building": "Marina Gate", "area_ids": (1,), "area_name": None, "unrecognised": ()},
    ),
    (
        "Princess Tower, Dubai Marina",
        {"building": "Princess Tower", "area_name": "Dubai Marina", "area_ids": (1,)},
    ),
    (
        "2BR Executive Towers business bay under 2M",
        {"building": "Executive Towers", "area_ids": (3,), "bedrooms": 2, "budget_max": 2e6},
    ),
    ("villa in project 5", {"building": "Project 5", "area_ids": (5,)}),
    # amenities
    ("villa with pool", {"amenities": ("shared pool",)}),
    ("apartment with sea view and balcony", {"amenities": ("sea view", "balcony")}),
    ("with a swimming pool and gym", {"amenities": ("shared pool", "gym access")}),
    ("maids room", {"amenities": ("maid's room",), "free_text": ""}),
    ("children's play area", {"amenities": ("children's play area",)}),
    # free text and emptiness
    ("quiet family villa", {"free_text": "quiet family", "property_type": "villa"}),
    ("investment property in JVC", {"free_text": "investment", "area_ids": (2,)}),
    ("cheap", {"free_text": "cheap"}),
    # whole queries
    (
        "looking for a 2 bed apartment in Downtown Dubai under AED 2.5M with balcony",
        {
            "bedrooms": 2,
            "property_type": "flat",
            "area_ids": (4,),
            "area_name": "Downtown Dubai",
            "budget_max": 2_500_000.0,
            "amenities": ("balcony",),
            "free_text": "",
            "unrecognised": (),
        },
    ),
    (
        "Need 3BR villa Arabian Ranches 3M-4.5M over 300 sqm",
        {
            "bedrooms": 3,
            "property_type": "villa",
            "area_ids": (5,),
            "budget_min": 3e6,
            "budget_max": 4.5e6,
            "min_size_sqm": 300.0,
            "free_text": "",
        },
    ),
    (
        "show me studio flats near JVC max 600k",
        {"bedrooms": 0, "property_type": "flat", "area_ids": (2,), "budget_max": 6e5},
    ),
]


@pytest.mark.parametrize(("text", "expected"), CASES, ids=[case[0] for case in CASES])
def test_parse(text, expected):
    parsed = parse(text, LEXICON)
    for field, value in expected.items():
        actual = getattr(parsed, field)
        if isinstance(value, float):
            assert actual == pytest.approx(value, abs=0.01), field
        else:
            assert actual == value, field


@pytest.mark.parametrize("text", ["", "   ", "under 5", "the a for"])
def test_empty_queries(text):
    assert parse(text, LEXICON).is_empty


def test_a_slot_makes_a_query_non_empty():
    assert not parse("villa", LEXICON).is_empty
    assert not parse("cheap", LEXICON).is_empty


def test_to_dict_is_json_ready():
    import json

    parsed = parse("2BR in Dubai Marina under 1.5M with pool", LEXICON)
    data = json.loads(json.dumps(parsed.to_dict()))
    assert data["area_ids"] == [1] and data["amenities"] == ["shared pool"]
    assert data["unrecognised"] == []


def test_every_synonym_maps_to_a_real_amenity():
    from listings.text import AMENITIES

    assert set(AMENITY_SYNONYMS.values()) <= set(AMENITIES)
    assert all(amenity in AMENITY_SYNONYMS for amenity in AMENITIES)


def test_areas_win_over_buildings_with_the_same_key_and_short_names_are_ignored():
    lexicon = Lexicon.from_rows(
        areas=[(1, "Dubai Marina")],
        aliases=[("Marina", 1)],
        buildings=[("Marina", 9), ("Tower", 1), ("Bay Gate", 2)],
        projects=[("Bay Gate", 3)],
    )
    assert lexicon.entries["marina"].kind == "area"
    assert "tower" not in lexicon.entries  # one token, under six characters
    assert lexicon.entries["bay gate"].kind == "building"
    assert lexicon.entries["bay gate"].area_ids == (2,)


def test_an_alias_shared_by_two_areas_keeps_both_ids():
    lexicon = Lexicon.from_rows(
        areas=[(1, "Al Barsha First"), (2, "Al Barsha South")],
        aliases=[("Barsha", 1), ("Barsha", 2)],
        buildings=[],
        projects=[],
    )
    assert parse("villa in al barsha", lexicon).area_ids == (1, 2)


def test_place_names_keep_their_slot_words():
    lexicon = Lexicon.from_rows(
        areas=[(7, "Dubai Studio City"), (8, "Town Square")],
        aliases=[],
        buildings=[("Villa Lantana", 7)],
        projects=[],
    )
    parsed = parse("2 bed villa in Dubai Studio City under 1.2M", lexicon)
    assert parsed.bedrooms == 2 and parsed.area_ids == (7,) and parsed.property_type == "villa"
    parsed = parse("townhouse in town square", lexicon)
    assert parsed.property_type == "townhouse" and parsed.area_ids == (8,)
    parsed = parse("studio at villa lantana", lexicon)
    assert parsed.bedrooms == 0 and parsed.building == "Villa Lantana"
    assert parsed.property_type is None


def test_bedroom_numbers_are_never_money():
    parsed = parse("2 bed 1.5M", LEXICON)
    assert parsed.bedrooms == 2 and parsed.budget_max == 1_500_000.0
    parsed = parse("5 bedroom villa 5M-6M", LEXICON)
    assert (parsed.bedrooms, parsed.budget_min, parsed.budget_max) == (5, 5e6, 6e6)


def test_lexicon_loads_from_postgres(search_db):
    settings, _, _ = search_db
    conn = settings.connect()
    try:
        lexicon = load_lexicon(conn)
    finally:
        conn.close()
    assert lexicon.entries["dubai marina"].area_ids == (1,)
    assert lexicon.entries["jvc"] == Place("area", "JVC", (2,))
    assert lexicon.entries["marina gate"].kind == "building"
    assert lexicon.entries["project 3"].kind == "project"
    assert parse("2 bed at Marina Gate", lexicon).area_ids == (1,)
```

Note: `from search_fixtures import ...` works because pytest (prepend import mode, no `__init__.py`) puts `tests/search/` on `sys.path` while collecting it. Ruff may sort `search_fixtures` among the third-party imports; accept ruff's order.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/search/test_search_parse.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'search.lexicon'`.

- [ ] **Step 3: Write the lexicon**

`search/lexicon.py`:

```python
"""Place names a query can mention: areas (official names and aliases), buildings, projects."""

from dataclasses import dataclass

from ingestion.normalize import match_key

MIN_SINGLE_TOKEN_CHARS = 6  # one-word building names shorter than this are too generic
MAX_PHRASE_TOKENS = 6


@dataclass(frozen=True)
class Place:
    kind: str  # "area" | "building" | "project"
    name: str
    area_ids: tuple[int, ...]


@dataclass(frozen=True)
class Lexicon:
    entries: dict[str, Place]  # match_key -> place
    max_tokens: int

    @classmethod
    def from_rows(cls, areas, aliases, buildings, projects) -> "Lexicon":
        """areas: (area_id, name); aliases, buildings, projects: (name, area_id).

        Areas win any key they share with a building or project; buildings win over projects.
        """
        area_ids: dict[str, set[int]] = {}
        display: dict[str, str] = {}
        for area_id, name in [*areas, *((a, n) for n, a in aliases)]:
            key = match_key(name)
            if key:
                area_ids.setdefault(key, set()).add(int(area_id))
                display.setdefault(key, name)
        entries = {
            key: Place("area", display[key], tuple(sorted(ids))) for key, ids in area_ids.items()
        }
        for kind, rows in (("building", buildings), ("project", projects)):
            grouped: dict[str, tuple[str, set[int]]] = {}
            for name, area_id in rows:
                key = match_key(name)
                if not key or key in entries:
                    continue
                if len(key.split()) < 2 and len(key) < MIN_SINGLE_TOKEN_CHARS:
                    continue
                grouped.setdefault(key, (name, set()))[1].add(int(area_id))
            for key, (name, ids) in grouped.items():
                entries[key] = Place(kind, name, tuple(sorted(ids)))
        longest = max((len(key.split()) for key in entries), default=1)
        return cls(entries, min(longest, MAX_PHRASE_TOKENS))

    def match(self, tokens: list[str]) -> list[tuple[int, int, Place]]:
        """Greedy, left to right, longest phrase first: (start, end, place) token spans."""
        spans = []
        index = 0
        while index < len(tokens):
            for size in range(min(self.max_tokens, len(tokens) - index), 0, -1):
                place = self.entries.get(" ".join(tokens[index : index + size]))
                if place is not None:
                    spans.append((index, index + size, place))
                    index += size
                    break
            else:
                index += 1
        return spans


LEXICON_SQL = {
    "areas": "SELECT area_id, name_en FROM dld.areas ORDER BY area_id",
    "aliases": "SELECT alias, area_id FROM dld.area_aliases ORDER BY alias, area_id",
    "buildings": (
        "SELECT DISTINCT building_name, area_id FROM listings.listings "
        "WHERE building_name IS NOT NULL ORDER BY 1, 2"
    ),
    "projects": (
        "SELECT DISTINCT project_name, area_id FROM listings.listings "
        "WHERE project_name IS NOT NULL ORDER BY 1, 2"
    ),
}


def load_lexicon(conn) -> Lexicon:
    rows = {}
    with conn.cursor() as cur:
        for name, sql in LEXICON_SQL.items():
            cur.execute(sql)
            rows[name] = cur.fetchall()
    return Lexicon.from_rows(**rows)
```

- [ ] **Step 4: Write the parser**

`search/parse.py`:

```python
"""Query text -> structured slots. Pure: the place lexicon is passed in.

Each pass blanks what it consumed with BLANK, so later passes cannot reuse it, and the
unrecognised-place scan can tell that a preposition was followed by an already-matched place.
"""

import re
from dataclasses import asdict, dataclass

from listings.text import AMENITIES
from search.config import SQFT_TO_SQM
from search.lexicon import Lexicon

BLANK = "\x00"
MIN_BARE_AMOUNT = 10_000.0  # a number with no suffix and no "AED" is money only from here up
SCALE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6}
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
AMENITY_SYNONYMS = {
    **{amenity: amenity for amenity in AMENITIES},
    "pool": "shared pool",
    "swimming pool": "shared pool",
    "parking": "covered parking",
    "gym": "gym access",
    "security": "24/7 security",
    "fitted kitchen": "fully fitted kitchen",
    "wardrobes": "built-in wardrobes",
    "maids room": "maid's room",
    "maid room": "maid's room",
    "play area": "children's play area",
    "kids play area": "children's play area",
    "central ac": "central air conditioning",
}
STOPWORDS = frozenset(
    "a an the i we me my for in at near around with and or vs looking want need to buy rent "
    "show find searching search please property properties home homes budget of al under "
    "below max up over above from between less more than least most min".split()
)
PLACE_PREPOSITIONS = frozenset({"in", "at", "near", "around"})
NEARBY_WORDS = frozenset({"metro", "beach", "school", "schools", "mall", "airport", "park"})
PLACE_STOP = (STOPWORDS - {"al"}) | NEARBY_WORDS

_SIZE = re.compile(
    r"(?:(?:over|above|at least|min(?:imum)?|more than|from)\s+)?"
    r"(?P<n>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*\+?\s*"
    r"(?P<u>sq\.?\s*f(?:ee)?t|sqft|square\s+f(?:ee|oo)t"
    r"|sq\.?\s*m(?:eters?|etres?)?|sqm|m2|m²|square\s+met(?:er|re)s?)(?![a-z0-9])"
)
_BEDS = re.compile(
    r"(?<![a-z0-9])(?:(?P<studio>studios?)"
    r"|(?P<n>\d)\s*-?\s*(?:br|bhk|beds?|bedrooms?)"
    r"|(?P<w>one|two|three|four|five|six|seven)[\s-]*(?:br|bhk|beds?|bedrooms?))(?![a-z])"
)
_MONEY = (
    r"(?P<aedX>aed\s*)?(?<![0-9.,])(?P<nX>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
    r"\s*(?P<sX>million|thousand|mn|m|k)?(?![a-z0-9])(?P<postX>\s*aed(?![a-z]))?"
)
M1, M2 = _MONEY.replace("X", "1"), _MONEY.replace("X", "2")
_RANGE = re.compile(rf"(?:between\s+|from\s+)?{M1}\s*(?:-|–|to|and)\s*{M2}")
_MAX = re.compile(
    r"(?:under|below|max(?:imum)?|up\s+to|less\s+than|within|budget(?:\s+of)?|at\s+most"
    rf"|no\s+more\s+than)\s*:?\s*{M1}"
)
_MIN = re.compile(
    rf"(?:from|over|above|at\s+least|more\s+than|min(?:imum)?|starting\s+(?:at|from))\s*:?\s*{M1}"
)
_BARE = re.compile(M1)
_TYPES = (
    (re.compile(r"(?<![a-z])(?:hotel|serviced)\s+apartments?(?![a-z])"), "hotel_apartment"),
    (re.compile(r"(?<![a-z])(?:town\s*houses?|townhomes?)(?![a-z])"), "townhouse"),
    (re.compile(r"(?<![a-z])villas?(?![a-z])"), "villa"),
    (re.compile(r"(?<![a-z])(?:apartments?|flats?)(?![a-z])"), "flat"),
)
_UNLISTED = re.compile(r"(?<![a-z])(?P<w>plots?|land)(?![a-z])")
_AMENITY_PATTERNS = tuple(
    (re.compile(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])"), canonical)
    for phrase, canonical in sorted(AMENITY_SYNONYMS.items(), key=lambda item: -len(item[0]))
)
_TOKEN = re.compile(r"[a-z0-9]+")
_WORD = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class ParsedQuery:
    area_ids: tuple[int, ...] = ()
    area_name: str | None = None
    building: str | None = None  # a building or project name, as the lexicon spells it
    bedrooms: int | None = None
    property_type: str | None = None
    budget_min: float | None = None
    budget_max: float | None = None
    min_size_sqm: float | None = None
    amenities: tuple[str, ...] = ()
    free_text: str = ""
    unrecognised: tuple[tuple[str, str], ...] = ()  # ("place" | "type", text)
    errors: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not (
            self.area_ids
            or self.building
            or self.bedrooms is not None
            or self.property_type
            or self.budget_min is not None
            or self.budget_max is not None
            or self.min_size_sqm is not None
            or self.amenities
            or self.free_text
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["area_ids"] = list(self.area_ids)
        data["amenities"] = list(self.amenities)
        data["unrecognised"] = [list(item) for item in self.unrecognised]
        data["errors"] = list(self.errors)
        return data


def _blank(text: str, start: int, end: int) -> str:
    return text[:start] + BLANK * (end - start) + text[end:]


def _size(text: str) -> tuple[str, float | None]:
    value = None
    for match in _SIZE.finditer(text):
        amount = float(match["n"].replace(",", ""))
        if "f" in match["u"]:
            amount *= SQFT_TO_SQM
        if value is None:
            value = round(amount, 2)
        text = _blank(text, *match.span())
    return text, value


def _bedrooms(text: str) -> tuple[str, int | None]:
    value = None
    for match in _BEDS.finditer(text):
        if match["studio"]:
            count = 0
        elif match["n"]:
            count = int(match["n"])
        else:
            count = NUMBER_WORDS[match["w"]]
        if value is None:
            value = count
        text = _blank(text, *match.span())
    return text, value


def _amount(match, index: str, inherit: tuple[str | None, bool] | None = None) -> float | None:
    suffix = match[f"s{index}"]
    has_aed = bool(match[f"aed{index}"] or match[f"post{index}"])
    if inherit is not None:
        suffix = suffix or inherit[0]
        has_aed = has_aed or inherit[1]
    value = float(match[f"n{index}"].replace(",", "")) * SCALE.get(suffix or "", 1.0)
    return value if (suffix or has_aed or value >= MIN_BARE_AMOUNT) else None


def _budget(text: str) -> tuple[str, float | None, float | None]:
    low = high = None
    for match in _RANGE.finditer(text):
        second = _amount(match, "2")
        first = _amount(match, "1", inherit=(match["s2"], bool(match["aed2"] or match["post2"])))
        if first is not None and second is not None:
            low, high = first, second
            text = _blank(text, *match.span())
            break
    for pattern, which in ((_MAX, "max"), (_MIN, "min")):
        if (high if which == "max" else low) is not None:
            continue
        for match in pattern.finditer(text):
            amount = _amount(match, "1")
            if amount is not None:
                if which == "max":
                    high = amount
                else:
                    low = amount
                text = _blank(text, *match.span())
                break
    if low is None and high is None:
        for match in _BARE.finditer(text):
            amount = _amount(match, "1")
            if amount is not None:
                high = amount
                text = _blank(text, *match.span())
                break
    return text, low, high


def _places(text: str, lexicon: Lexicon) -> tuple[str, list]:
    tokens = [(m.group(), m.start(), m.end()) for m in _TOKEN.finditer(text) if m.group() != "al"]
    found = []
    for start, end, place in lexicon.match([token for token, _, _ in tokens]):
        text = _blank(text, tokens[start][1], tokens[end - 1][2])
        found.append(place)
    return text, found


def _types(text: str) -> tuple[str, str | None, list[str]]:
    matches = []
    for pattern, kind in _TYPES:
        for match in pattern.finditer(text):
            matches.append((match.start(), kind))
            text = _blank(text, *match.span())
    unlisted = []
    for match in _UNLISTED.finditer(text):
        unlisted.append("plot" if match["w"].startswith("plot") else "land")
        text = _blank(text, *match.span())
    kind = min(matches)[1] if matches else None
    return text, kind, unlisted


def _amenities(text: str) -> tuple[str, tuple[str, ...]]:
    found = []
    for pattern, canonical in _AMENITY_PATTERNS:
        for match in pattern.finditer(text):
            found.append((match.start(), canonical))
            text = _blank(text, *match.span())
    ordered: list[str] = []
    for _, canonical in sorted(found):
        if canonical not in ordered:
            ordered.append(canonical)
    return text, tuple(ordered)


def _unrecognised_places(text: str) -> tuple[str, list[str]]:
    words = [(m.group(), m.start(), m.end()) for m in _WORD.finditer(text)]
    found = []
    index = 0
    while index < len(words):
        word, _, end = words[index]
        if word not in PLACE_PREPOSITIONS:
            index += 1
            continue
        phrase = []
        cursor, previous_end = index + 1, end
        while cursor < len(words) and len(phrase) < 3:
            token, start, stop = words[cursor]
            if BLANK in text[previous_end:start] or token in PLACE_STOP or token.isdigit():
                break
            phrase.append((token, start, stop))
            previous_end = stop
            cursor += 1
        if phrase:
            found.append(" ".join(token for token, _, _ in phrase))
            text = _blank(text, phrase[0][1], phrase[-1][2])
        index = cursor
    return text, found


def _free_text(text: str) -> str:
    words = (word.strip("'") for word in _WORD.findall(text))
    return " ".join(
        word for word in words if len(word) > 1 and word not in STOPWORDS and not word.isdigit()
    )


def parse(text: str, lexicon: Lexicon) -> ParsedQuery:
    working = f" {(text or '').lower().replace('’', chr(39))} "
    working, min_size = _size(working)
    working, budget_min, budget_max = _budget(working)
    errors = []
    if budget_min is not None and budget_max is not None and budget_min > budget_max:
        errors.append("budget_min_exceeds_max")
        budget_min = budget_max = None
    working, places = _places(working, lexicon)
    working, bedrooms = _bedrooms(working)
    working, property_type, unlisted = _types(working)
    working, amenities = _amenities(working)
    working, unknown = _unrecognised_places(working)
    area = next((place for place in places if place.kind == "area"), None)
    named = next((place for place in places if place.kind != "area"), None)
    if area is not None:
        area_ids = area.area_ids
    elif named is not None:
        area_ids = named.area_ids
    else:
        area_ids = ()
    return ParsedQuery(
        area_ids=area_ids,
        area_name=area.name if area else None,
        building=named.name if named else None,
        bedrooms=bedrooms,
        property_type=property_type,
        budget_min=budget_min,
        budget_max=budget_max,
        min_size_sqm=min_size,
        amenities=amenities,
        free_text=_free_text(working),
        unrecognised=tuple(
            [("type", word) for word in unlisted] + [("place", phrase) for phrase in unknown]
        ),
        errors=tuple(errors),
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q -W error tests/search/test_search_parse.py`
Expected: all cases PASS.

If a case fails, fix the **parser**, not the case. The one exception is a case whose expectation contradicts this task's written rules; if you change such a case, list it in your report with the reason.

- [ ] **Step 6: Lint, full search tests, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
uv run pytest -q -W error tests/search
git add search/lexicon.py search/parse.py tests/search/test_search_parse.py
git commit -m "feat(search): place lexicon and a rule parser for free-text property queries

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 3: Relevance grades and synthetic queries

**Files:**
- Create:
  - `search/grade.py`
  - `search/queries.py`
  - `tests/search/test_search_grade.py`
  - `tests/search/test_search_queries.py`

**Interfaces:**
- **Consumes:**
  - `search.config` (`SearchConfig`, `QUERY_KINDS`, `SPLITS`, `SQFT_TO_SQM`, `KIND_SQL`, `listing_kind`)
  - `search.parse.AMENITY_SYNONYMS`, `search.parse.parse`
  - `search.lexicon.Lexicon`
  - `listings.text.AMENITIES`
  - `ingestion.normalize.match_key`, `ingestion.normalize.map_unique`
  - `search_fixtures.build_search_listings`, `search_fixtures.ALIASES`, `search_fixtures.AREAS`, `search_fixtures.BUILDINGS`
- **Produces (`search.grade`):**
  - `GRADE_SCHEMA: dict`: the columns a grading frame must have.
  - `grade_frame(true_slots: dict, listings: pl.DataFrame, near_margin: float = 0.10) -> pl.Series` (Int64, named `grade`).
  - `grade(true_slots: dict, listing: dict, near_margin: float = 0.10) -> int`.
- **Produces (`search.queries`):**
  - Constants:
    - `TRUE_SLOT_KEYS`
    - `PREFIXES` (6), `ORDERS` (5), `N_TEMPLATES` (= 30)
    - `VAGUE_WORDS`
    - `MIN_BUDGET = 10_000.0`
  - `grading_frame(listings: pl.DataFrame) -> pl.DataFrame`
    - Takes Phase 4 listing columns and adds `kind`, `building_key` and `project_key`.
  - `load_grading_listings(conn) -> pl.DataFrame` (a grading frame, read from Postgres).
  - `load_area_aliases(conn) -> dict[int, list[str]]` (non-official aliases only).
  - `assign_template_splits(seed: int, shares: tuple[float, float, float]) -> dict[int, str]`
  - `generate_queries(grading: pl.DataFrame, aliases_by_area: dict[int, list[str]], config: SearchConfig) -> pl.DataFrame`
    - Columns: `query_id, text, template_id, kind, split, true_slots` (JSON text), `seed_listing_id, n_grade3`.
  - `true_slots(row_json: str) -> dict`
    - Parses one query's JSON.

**Grading rules.** This is the spec's "Grading" section, made exact.
- **Hard slots:** area, type and building (a building or project name) either hold or fail. They are never near misses.
- **Soft slots:** each soft slot is judged *exact* or *near*, and a near result is a superset of exact:

| Soft slot | Exact | Near |
|---|---|---|
| Bedrooms | difference = 0 | difference ≤ 1 |
| Budget | `min ≤ price ≤ max` | `price ≥ min × 0.9` and `price ≤ max × 1.1` |
| Size | `size ≥ min` | `size ≥ min × 0.9` |
| Amenities (k asked) | all k present | k − 1 present, only when k = 2 |

- **Grade 3:** every hard slot holds, and every soft slot is exact.
- **Grade 2:** every hard slot holds, every soft slot is at least near, and exactly one soft slot is near but not exact.
- **Grade 1:** otherwise, when the area (if stated) and the type (if stated) hold.
- **Grade 0:** otherwise.
- **Nulls:** a null bedroom count on the listing fails both the exact and the near test.
- **Fraud cap:** if the listing's `fraud_label` is not null, the grade is at most 1.

- [ ] **Step 1: Write the failing grade tests**

`tests/search/test_search_grade.py`:

```python
import polars as pl
import pytest
from search_fixtures import build_search_listings

from search.grade import grade, grade_frame
from search.queries import grading_frame

LISTING = {
    "listing_id": 1,
    "area_id": 1,
    "kind": "flat",
    "building_key": "marina gate",
    "project_key": "project 1",
    "bedrooms": 2,
    "size_sqm": 100.0,
    "asking_price_aed": 1_000_000.0,
    "description": "Features include balcony, sea view.",
    "fraud_label": None,
}
BASE = {"area_id": 1, "property_type": "flat"}


@pytest.mark.parametrize(
    ("slots", "expected"),
    [
        ({**BASE, "bedrooms": 2, "budget_max": 1_000_000, "amenities": ["balcony"]}, 3),
        ({**BASE, "bedrooms": 3}, 2),
        ({**BASE, "bedrooms": 4}, 1),
        ({**BASE, "budget_max": 952_000}, 2),  # 5% over max
        ({**BASE, "budget_max": 869_000}, 1),  # 15% over max
        ({**BASE, "budget_min": 1_050_000}, 2),  # under min by under 10%
        ({**BASE, "budget_min": 1_200_000}, 1),
        ({**BASE, "min_size_sqm": 105.0}, 2),
        ({**BASE, "min_size_sqm": 125.0}, 1),
        ({**BASE, "bedrooms": 3, "budget_max": 952_000}, 1),  # two near misses
        ({**BASE, "amenities": ["balcony", "shared pool"]}, 2),
        ({**BASE, "amenities": ["shared pool"]}, 1),
        ({**BASE, "amenities": ["balcony", "sea view", "gym access"]}, 1),
        ({**BASE, "amenities": ["BALCONY", "Sea View"]}, 3),
        ({"area_id": 2, "property_type": "flat", "bedrooms": 2}, 0),
        ({"area_id": 1, "property_type": "villa", "bedrooms": 2}, 0),
        ({"area_id": 2, "bedrooms": 3}, 0),  # a wrong area is never a near miss
        ({**BASE, "building": "Marina Gate", "bedrooms": 2}, 3),
        ({**BASE, "building": "Project 1"}, 3),
        ({**BASE, "building": "Princess Tower", "bedrooms": 2}, 1),
        ({"bedrooms": 2}, 3),
        ({"property_type": "villa"}, 0),
        ({"amenities": ["shared pool"]}, 1),
    ],
)
def test_grade(slots, expected):
    assert grade(slots, LISTING) == expected


def test_fraud_label_caps_at_one():
    slots = {**BASE, "bedrooms": 2}
    assert grade(slots, {**LISTING, "fraud_label": "bait_price"}) == 1
    assert grade({"area_id": 2}, {**LISTING, "fraud_label": "bait_price"}) == 0


def test_a_listing_without_bedrooms_misses_a_bedroom_slot():
    assert grade({**BASE, "bedrooms": 2}, {**LISTING, "bedrooms": None}) == 1


def test_the_near_margin_is_configurable():
    assert grade({**BASE, "budget_max": 869_000}, LISTING, near_margin=0.2) == 2


def test_frame_grades_match_row_grades_on_the_test_corpus():
    frame = grading_frame(build_search_listings())
    slots = {"area_id": 1, "property_type": "flat", "bedrooms": 1, "budget_max": 1_500_000}
    grades = grade_frame(slots, frame)
    assert grades.name == "grade" and grades.len() == frame.height
    one_by_one = [grade(slots, row) for row in frame.to_dicts()]
    assert grades.to_list() == one_by_one
    assert set(one_by_one) >= {0, 1, 3}
```

- [ ] **Step 2: Run the grade tests to verify they fail**

Run: `uv run pytest -q tests/search/test_search_grade.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'search.grade'`.

- [ ] **Step 3: Write the grader**

`search/grade.py`:

```python
"""Relevance grades 0-3 for (query, listing) pairs, from the query's TRUE slots.

This module reads ground truth on purpose: grading is where true slots and fraud labels meet.
Rules: docs/superpowers/plans/2026-09-16-phase5-search-ranking.md, Task 3.
"""

import polars as pl

from ingestion.normalize import match_key

GRADE_SCHEMA = {
    "listing_id": pl.Int64,
    "area_id": pl.Int64,
    "kind": pl.Utf8,
    "building_key": pl.Utf8,
    "project_key": pl.Utf8,
    "bedrooms": pl.Int64,
    "size_sqm": pl.Float64,
    "asking_price_aed": pl.Float64,
    "description": pl.Utf8,
    "fraud_label": pl.Utf8,
}


def _all(expressions: list[pl.Expr]) -> pl.Expr:
    return pl.all_horizontal(expressions) if expressions else pl.lit(True)


def _soft_slots(true_slots: dict, margin: float) -> list[tuple[pl.Expr, pl.Expr]]:
    """(exact, near) per stated soft slot; near is always a superset of exact."""
    soft = []
    bedrooms = true_slots.get("bedrooms")
    if bedrooms is not None:
        difference = (pl.col("bedrooms") - bedrooms).abs()
        soft.append((difference == 0, difference <= 1))
    low, high = true_slots.get("budget_min"), true_slots.get("budget_max")
    if low is not None or high is not None:
        price = pl.col("asking_price_aed")
        exact, near = [], []
        if low is not None:
            exact.append(price >= low)
            near.append(price >= low * (1.0 - margin))
        if high is not None:
            exact.append(price <= high)
            near.append(price <= high * (1.0 + margin))
        soft.append((_all(exact), _all(near)))
    min_size = true_slots.get("min_size_sqm")
    if min_size is not None:
        size = pl.col("size_sqm")
        soft.append((size >= min_size, size >= min_size * (1.0 - margin)))
    amenities = [amenity.lower() for amenity in true_slots.get("amenities") or []]
    if amenities:
        text = pl.col("description").str.to_lowercase()
        hits = pl.sum_horizontal(
            [text.str.contains(amenity, literal=True).cast(pl.Int64) for amenity in amenities]
        )
        allowed_missing = 1 if len(amenities) == 2 else 0
        soft.append((hits == len(amenities), hits >= len(amenities) - allowed_missing))
    return [(exact.fill_null(False), near.fill_null(False)) for exact, near in soft]


def grade_frame(true_slots: dict, listings: pl.DataFrame, near_margin: float = 0.10) -> pl.Series:
    area_ok, type_ok, hard = pl.lit(True), pl.lit(True), []
    if true_slots.get("area_id") is not None:
        area_ok = pl.col("area_id") == true_slots["area_id"]
        hard.append(area_ok)
    if true_slots.get("property_type"):
        type_ok = (pl.col("kind") == true_slots["property_type"]).fill_null(False)
        hard.append(type_ok)
    if true_slots.get("building"):
        key = match_key(true_slots["building"])
        hard.append(
            ((pl.col("building_key") == key) | (pl.col("project_key") == key)).fill_null(False)
        )
    soft = _soft_slots(true_slots, near_margin)
    hard_ok = _all(hard)
    all_near = _all([near for _, near in soft])
    near_count = (
        pl.sum_horizontal([(~exact & near).cast(pl.Int64) for exact, near in soft])
        if soft
        else pl.lit(0)
    )
    graded = (
        pl.when(hard_ok & all_near & (near_count == 0))
        .then(3)
        .when(hard_ok & all_near & (near_count == 1))
        .then(2)
        .when(area_ok & type_ok)
        .then(1)
        .otherwise(0)
    )
    capped = pl.when(pl.col("fraud_label").is_not_null()).then(pl.min_horizontal(graded, 1))
    expression = capped.otherwise(graded).cast(pl.Int64).alias("grade")
    return listings.select(expression).to_series()


def grade(true_slots: dict, listing: dict, near_margin: float = 0.10) -> int:
    row = {name: listing.get(name) for name in GRADE_SCHEMA}
    frame = pl.DataFrame([row], schema=GRADE_SCHEMA)
    return int(grade_frame(true_slots, frame, near_margin)[0])
```

- [ ] **Step 4: Run the grade tests (`grading_frame` still missing)**

Run: `uv run pytest -q tests/search/test_search_grade.py -k "not frame_grades"`
Expected: all tests PASS. The deselected test needs `grading_frame`, which Step 7 adds.

- [ ] **Step 5: Write the failing query tests**

`tests/search/test_search_queries.py`:

```python
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
            same = parsed.budget_max is not None and abs(parsed.budget_max - slots["budget_max"]) < 1
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
```

- [ ] **Step 6: Run the query tests to verify they fail**

Run: `uv run pytest -q tests/search/test_search_queries.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'search.queries'`.

- [ ] **Step 7: Write the generator**

`search/queries.py`:

```python
"""Synthetic search queries with their true slots.

Each query is seeded from a real (non-fraud) listing, then rendered through one of N_TEMPLATES
frames: a prefix, a slot order, and a phrasing style for bedrooms and budget. Queries are split
by frame, so the report split uses frames the ranker never saw. This module reads ground truth
(fraud labels) only to choose seeds and to grade.
"""

import json
import math
import random

import polars as pl

from ingestion.normalize import map_unique, match_key
from listings.text import AMENITIES
from search.config import KIND_SQL, QUERY_KINDS, SPLITS, SQFT_TO_SQM, SearchConfig, listing_kind
from search.grade import GRADE_SCHEMA, grade_frame
from search.parse import AMENITY_SYNONYMS

TRUE_SLOT_KEYS = (
    "area_id", "area_name", "building", "bedrooms", "property_type",
    "budget_min", "budget_max", "min_size_sqm", "amenities",
)  # fmt: skip
PREFIXES = ("", "looking for", "need a", "searching for", "want to buy", "show me")
ORDERS = (
    ("beds_type", "area", "budget", "size", "amenities"),
    ("area", "beds_type", "budget", "amenities", "size"),
    ("beds_type", "budget", "area", "size", "amenities"),
    ("budget", "beds_type", "area", "amenities", "size"),
    ("beds_type", "amenities", "area", "budget", "size"),
)
N_TEMPLATES = len(PREFIXES) * len(ORDERS)
VAGUE_WORDS = ("family", "investment", "quiet", "modern", "spacious", "affordable", "luxury")
TYPE_WORDS = {
    "flat": ("apartment", "flat"),
    "hotel_apartment": ("hotel apartment", "serviced apartment"),
    "townhouse": ("townhouse",),
    "villa": ("villa",),
}
NUMBER_NAMES = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}
MAX_PHRASED_BEDROOMS = 7
MIN_BUDGET = 10_000.0
NO_MATCH_BUDGET_SHARE = 0.3
PHRASES_BY_AMENITY: dict[str, list[str]] = {}
for _phrase, _canonical in sorted(AMENITY_SYNONYMS.items()):
    PHRASES_BY_AMENITY.setdefault(_canonical, []).append(_phrase)

GRADING_SQL = f"""
SELECT l.listing_id, l.area_id, l.area_name, {KIND_SQL} AS kind, l.building_name,
       l.project_name, l.bedrooms, l.size_sqm, l.asking_price_aed, l.description, l.fraud_label
FROM listings.listings AS l
ORDER BY l.listing_id
"""
GRADING_READ_SCHEMA = {
    "listing_id": pl.Int64,
    "area_id": pl.Int64,
    "area_name": pl.Utf8,
    "kind": pl.Utf8,
    "building_name": pl.Utf8,
    "project_name": pl.Utf8,
    "bedrooms": pl.Int64,
    "size_sqm": pl.Float64,
    "asking_price_aed": pl.Float64,
    "description": pl.Utf8,
    "fraud_label": pl.Utf8,
}


def _with_keys(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        map_unique(frame["building_name"], match_key, pl.Utf8).alias("building_key"),
        map_unique(frame["project_name"], match_key, pl.Utf8).alias("project_key"),
    ).select(*GRADE_SCHEMA, "area_name", "building_name", "project_name")


def grading_frame(listings: pl.DataFrame) -> pl.DataFrame:
    """Phase 4 listing columns -> the frame grade_frame and generate_queries read."""
    kinds = [
        listing_kind(property_type, sub_type)
        for property_type, sub_type in listings.select("property_type", "property_sub_type").rows()
    ]
    frame = listings.with_columns(pl.Series("kind", kinds, dtype=pl.Utf8)).with_columns(
        pl.col("bedrooms").cast(pl.Int64), pl.col("area_id").cast(pl.Int64)
    )
    return _with_keys(frame)


def load_grading_listings(conn) -> pl.DataFrame:
    with conn.cursor() as cur:
        cur.execute(GRADING_SQL)
        frame = pl.DataFrame(cur.fetchall(), schema=GRADING_READ_SCHEMA, orient="row")
    return _with_keys(frame)


def load_area_aliases(conn) -> dict[int, list[str]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT area_id, alias FROM dld.area_aliases WHERE source <> 'official' "
            "ORDER BY area_id, alias"
        )
        rows = cur.fetchall()
    aliases: dict[int, list[str]] = {}
    for area_id, alias in rows:
        aliases.setdefault(area_id, []).append(alias)
    return aliases


def assign_template_splits(seed: int, shares: tuple[float, float, float]) -> dict[int, str]:
    template_ids = list(range(N_TEMPLATES))
    random.Random(seed).shuffle(template_ids)
    n_train = round(N_TEMPLATES * shares[0])
    n_tune = round(N_TEMPLATES * shares[1])
    splits = {}
    for position, template_id in enumerate(template_ids):
        if position < n_train:
            splits[template_id] = SPLITS[0]
        elif position < n_train + n_tune:
            splits[template_id] = SPLITS[1]
        else:
            splits[template_id] = SPLITS[2]
    return splits


def true_slots(row_json: str) -> dict:
    return json.loads(row_json)


def _ceil_nice(amount: float) -> float:
    step = 100_000.0 if amount >= 1_000_000 else 10_000.0
    return max(MIN_BUDGET, math.ceil(amount / step) * step)


def _floor_nice(amount: float) -> float:
    step = 100_000.0 if amount >= 1_000_000 else 10_000.0
    return max(MIN_BUDGET, math.floor(amount / step) * step)


def _money_text(rng: random.Random, amount: float) -> str:
    if amount >= 1_000_000:
        millions = f"{amount / 1e6:g}"
        options = (f"{millions}M", f"{amount:,.0f}", f"AED {millions} million", f"{millions}m AED")
    else:
        thousands = f"{amount / 1e3:g}"
        options = (f"{thousands}k", f"{amount:,.0f}", f"AED {thousands}k")
    return rng.choice(options)


def _budget_text(rng: random.Random, low: float | None, high: float, style: int) -> str:
    top = _money_text(rng, high)
    if low is None:
        return ("under {}", "below {}", "max {}", "up to {}", "budget {}")[style].format(top)
    bottom = _money_text(rng, low)
    template = ("{}-{}", "between {} and {}", "from {} to {}", "{} to {}", "{} - {}")[style]
    return template.format(bottom, top)


def _beds_text(bedrooms: int, style: int) -> str:
    if bedrooms == 0:
        return "studio"
    return (f"{bedrooms}BR", f"{bedrooms} bed", f"{NUMBER_NAMES[bedrooms]} bedroom")[style]


def _choose_slots(
    rng: random.Random,
    listing: dict,
    kind: str,
    aliases_by_area: dict[int, list[str]],
    min_price: dict[tuple[int, str], float],
    config: SearchConfig,
) -> dict:
    slots: dict = dict.fromkeys(TRUE_SLOT_KEYS)
    slots["amenities"] = []
    render = {"area_text": None, "amenity_words": [], "vague": None, "size_text": None}
    description = (listing["description"] or "").lower()
    present = [amenity for amenity in AMENITIES if amenity in description]
    bedrooms = listing["bedrooms"]
    available = ["area", "property_type"]
    if bedrooms is not None and 0 <= bedrooms <= MAX_PHRASED_BEDROOMS:
        available.append("bedrooms")
    available += ["budget", "min_size_sqm"]
    if present:
        available.append("amenities")

    if kind == "vague":
        pool = [slot for slot in ("area", "property_type", "amenities") if slot in available]
        chosen = rng.sample(pool, rng.randint(1, min(2, len(pool))))
        render["vague"] = rng.choice(VAGUE_WORDS)
    elif kind == "specified":
        chosen = rng.sample(available, rng.randint(3, len(available)))
    else:  # no_match: area + type + an impossible budget, plus any other slots
        rest = [slot for slot in available if slot not in ("area", "property_type", "budget")]
        chosen = ["area", "property_type", "budget", *rng.sample(rest, rng.randint(0, len(rest)))]

    if "area" in chosen:
        slots["area_id"], slots["area_name"] = listing["area_id"], listing["area_name"]
        aliases = aliases_by_area.get(listing["area_id"], [])
        use_alias = aliases and rng.random() < config.alias_probability
        render["area_text"] = rng.choice(aliases) if use_alias else listing["area_name"]
        named = listing["building_name"] or listing["project_name"]
        if kind == "specified" and named and rng.random() < config.building_probability:
            slots["building"] = named
    if "property_type" in chosen:
        slots["property_type"] = listing["kind"]
    if "bedrooms" in chosen:
        slots["bedrooms"] = bedrooms
    if "budget" in chosen:
        price = listing["asking_price_aed"]
        if kind == "no_match":
            cheapest = min_price[(listing["area_id"], listing["kind"])]
            slots["budget_max"] = _floor_nice(NO_MATCH_BUDGET_SHARE * cheapest)
        else:
            slots["budget_max"] = _ceil_nice(price * (1 + rng.uniform(*config.budget_headroom)))
            if rng.random() < 0.5:
                low = _floor_nice(price * (1 - rng.uniform(*config.budget_headroom)))
                if low < slots["budget_max"]:
                    slots["budget_min"] = low
    if "min_size_sqm" in chosen:
        target = listing["size_sqm"] * (1 - rng.uniform(*config.size_slack))
        if rng.random() < 0.3:
            square_feet = max(10, int(target / SQFT_TO_SQM // 10 * 10))
            slots["min_size_sqm"] = round(square_feet * SQFT_TO_SQM, 2)
            render["size_text"] = rng.choice(
                (f"at least {square_feet:,} sq ft", f"over {square_feet} sqft")
            )
        else:
            square_metres = max(10, int(target // 10 * 10))
            slots["min_size_sqm"] = float(square_metres)
            render["size_text"] = rng.choice(
                (
                    f"over {square_metres} sqm",
                    f"{square_metres}+ sqm",
                    f"min {square_metres} sqm",
                    f"at least {square_metres} m2",
                )
            )
    if "amenities" in chosen:
        picked = rng.sample(present, min(len(present), rng.randint(1, 2)))
        slots["amenities"] = picked
        render["amenity_words"] = [rng.choice(PHRASES_BY_AMENITY[amenity]) for amenity in picked]
    return {**slots, **render}


def render_query(rng: random.Random, slots: dict, template_id: int) -> str:
    prefix = PREFIXES[template_id // len(ORDERS)]
    order = ORDERS[template_id % len(ORDERS)]
    beds_style, budget_style = template_id % 3, (template_id // 3) % 5
    words = []
    if slots["vague"]:
        words.append(slots["vague"])
    if slots["bedrooms"] is not None:
        words.append(_beds_text(slots["bedrooms"], beds_style))
    if slots["property_type"]:
        words.append(rng.choice(TYPE_WORDS[slots["property_type"]]))
    pieces = {"beds_type": " ".join(words), "area": "", "budget": "", "size": "", "amenities": ""}
    if slots["area_text"]:
        if slots["building"]:
            pieces["area"] = f"at {slots['building']}, {slots['area_text']}"
        else:
            connector = rng.choice(("in", "at", "near", ""))
            pieces["area"] = f"{connector} {slots['area_text']}".strip()
    if slots["budget_max"] is not None:
        pieces["budget"] = _budget_text(
            rng, slots["budget_min"], slots["budget_max"], budget_style
        )
    if slots["size_text"]:
        pieces["size"] = slots["size_text"]
    if slots["amenity_words"]:
        pieces["amenities"] = "with " + " and ".join(slots["amenity_words"])
    parts = [prefix, *(pieces[name] for name in order)]
    return " ".join(" ".join(part for part in parts if part).split())


def generate_queries(
    grading: pl.DataFrame, aliases_by_area: dict[int, list[str]], config: SearchConfig
) -> pl.DataFrame:
    rng = random.Random(config.seed)
    splits = assign_template_splits(config.seed, config.split_shares)
    seeds = grading.filter(pl.col("fraud_label").is_null() & pl.col("kind").is_not_null())
    seed_rows = seeds.sort("listing_id").to_dicts()
    if not seed_rows:
        raise ValueError("no unlabelled listings to seed queries from")
    cheapest = grading.filter(pl.col("kind").is_not_null()).group_by("area_id", "kind").agg(
        pl.col("asking_price_aed").min()
    )
    min_price = {(a, k): p for a, k, p in cheapest.iter_rows()}
    rows, seen = [], set()
    attempts = 0
    while len(rows) < config.n_queries:
        attempts += 1
        if attempts > 20 * config.n_queries:
            raise RuntimeError(
                f"only {len(rows)} distinct queries after {attempts} attempts; "
                "the corpus is too small for n_queries"
            )
        listing = rng.choice(seed_rows)
        kind = rng.choices(QUERY_KINDS, weights=config.kind_shares)[0]
        template_id = rng.randrange(N_TEMPLATES)
        slots = _choose_slots(rng, listing, kind, aliases_by_area, min_price, config)
        text = render_query(rng, slots, template_id)
        if text in seen:
            continue
        seen.add(text)
        truth = {key: slots[key] for key in TRUE_SLOT_KEYS}
        grades = grade_frame(truth, grading, config.near_margin)
        rows.append(
            {
                "query_id": len(rows) + 1,
                "text": text,
                "template_id": template_id,
                "kind": kind,
                "split": splits[template_id],
                "true_slots": json.dumps(truth, allow_nan=False),
                "seed_listing_id": listing["listing_id"],
                "n_grade3": int((grades == 3).sum()),
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "query_id": pl.Int64,
            "text": pl.Utf8,
            "template_id": pl.Int64,
            "kind": pl.Utf8,
            "split": pl.Utf8,
            "true_slots": pl.Utf8,
            "seed_listing_id": pl.Int64,
            "n_grade3": pl.Int64,
        },
    )
```

- [ ] **Step 8: Run all Task 3 tests**

Run: `uv run pytest -q -W error tests/search/test_search_grade.py tests/search/test_search_queries.py`
Expected: all tests PASS.

If `test_the_parser_reads_generated_queries_back` fails for a slot:
- Print the first ten failing texts.
- Fix whichever side is wrong: the generator's phrasing or the parser's rule. Add a case for it to `tests/search/test_search_parse.py`.
- Do not lower the 0.97 bar. It is a contract between two modules written in this plan.

- [ ] **Step 9: Lint, full search tests, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
uv run pytest -q -W error tests/search
git add search/grade.py search/queries.py tests/search/test_search_grade.py tests/search/test_search_queries.py
git commit -m "feat(search): rule grades and synthetic queries split by phrasing frame

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 4: Retrieval, fusion, duplicate collapse, and the graded query set

**Files:**
- Create: `search/retrieve.py`, `tests/search/test_search_retrieve.py`
- Modify: `search/queries.py`: add `build_query_set` and its imports

**Interfaces:**
- **Consumes:**
  - `search.parse.ParsedQuery`, `search.parse.parse`
  - `search.lexicon.Lexicon`, `search.lexicon.load_lexicon`
  - `search.config.KIND_SQL`, `search.config.SearchConfig`
  - `search.grade.grade_frame`
  - `search.store.JUDGMENT_SCHEMA`, `search.store.latest_corpus_run`, `search.store.replace_query_set`, `search.store.read_judgments`
  - `listings.load.vector_literal`
  - `listings.embed.Embedder` (the protocol); tests use `FakeEmbedder`
  - Task 3: `load_grading_listings`, `load_area_aliases`, `generate_queries`, `true_slots`
- **Produces, from `search.retrieve`:**
  - `CANDIDATE_SCHEMA`, with columns `listing_id, semantic_cos, semantic_pos, fulltext_rank, fulltext_pos, rrf_score, fused_pos, cluster_size`
  - `DuplicateClusters(cluster_of: dict[int, int], size: dict[int, int], posted: dict[int, date])`, which has `.cluster_size(listing_id) -> int`
  - `load_clusters(conn) -> DuplicateClusters`, which raises `RuntimeError` when there is no Phase 4 detect run
  - `fulltext_terms(parsed: ParsedQuery, text: str) -> str`
  - `fuse(semantic: list[tuple[int, float]], fulltext: list[tuple[int, float]], rrf_k: int) -> pl.DataFrame`, returning every column except `fused_pos` and `cluster_size`
  - `collapse(fused: pl.DataFrame, clusters: DuplicateClusters, k: int) -> pl.DataFrame`, returning `CANDIDATE_SCHEMA`
  - `retrieve(conn, parsed, text, vector, clusters, config) -> pl.DataFrame`, returning `CANDIDATE_SCHEMA`
- **Produces, from `search.queries`:**
  - `build_query_set(conn, embedder, lexicon, config) -> tuple[pl.DataFrame, pl.DataFrame]`
    - The first frame is queries with `corpus_run_id` added, matching `store.QUERY_SCHEMA`.
    - The second frame is judgments, matching `store.JUDGMENT_SCHEMA`.

- [ ] **Step 1: Write the failing tests**

`tests/search/test_search_retrieve.py`:

```python
import dataclasses
from datetime import date

import polars as pl
import pytest
from search_fixtures import N_CLONES

from search.config import SearchConfig, listing_kind
from search.lexicon import load_lexicon
from search.parse import ParsedQuery, parse
from search.queries import build_query_set
from search.retrieve import (
    CANDIDATE_SCHEMA,
    DuplicateClusters,
    collapse,
    fulltext_terms,
    fuse,
    load_clusters,
    retrieve,
)
from search.store import read_judgments, read_queries, replace_query_set

CONFIG = dataclasses.replace(SearchConfig(), ef_search=100)


def test_fuse_adds_reciprocal_ranks_and_breaks_ties_by_id():
    fused = fuse([(1, 0.9), (2, 0.8)], [(2, 0.5), (3, 0.4)], rrf_k=60)
    assert fused["listing_id"].to_list() == [2, 1, 3]
    two, one, three = fused.to_dicts()
    assert two["rrf_score"] == pytest.approx(1 / 62 + 1 / 61)
    assert (two["semantic_pos"], two["fulltext_pos"]) == (2, 1)
    assert one["fulltext_rank"] == 0.0 and one["fulltext_pos"] is None
    assert three["semantic_cos"] is None and three["semantic_pos"] is None
    tied = fuse([(9, 0.5)], [(4, 0.5)], rrf_k=60)
    assert tied["listing_id"].to_list() == [4, 9]


def _clusters():
    return DuplicateClusters(
        cluster_of={1: 1, 231: 1, 7: 7, 8: 7},
        size={1: 2, 7: 2},
        posted={1: date(2023, 1, 1), 231: date(2023, 1, 6), 7: date(2023, 2, 1), 8: date(2023, 1, 5)},
    )


def test_collapse_keeps_the_earliest_retrieved_member_in_its_own_position():
    fused = fuse([(231, 0.9), (1, 0.8), (5, 0.7), (7, 0.6), (8, 0.5)], [], rrf_k=60)
    out = collapse(fused, _clusters(), k=10)
    assert out["listing_id"].to_list() == [1, 5, 8]
    assert out["fused_pos"].to_list() == [1, 2, 3]
    assert out["cluster_size"].to_list() == [2, 1, 2]
    assert out.schema == pl.Schema(CANDIDATE_SCHEMA)


def test_collapse_keeps_a_lone_retrieved_clone_and_truncates_after_collapsing():
    fused = fuse([(231, 0.9), (5, 0.8), (6, 0.7), (7, 0.6), (8, 0.5)], [], rrf_k=60)
    out = collapse(fused, _clusters(), k=3)
    assert out["listing_id"].to_list() == [231, 5, 6]
    assert _clusters().cluster_size(999) == 1


def test_fulltext_terms_prefer_free_text_and_amenities():
    parsed = ParsedQuery(free_text="quiet family", amenities=("sea view",))
    assert fulltext_terms(parsed, "ignored") == "quiet or family or sea or view"
    assert fulltext_terms(ParsedQuery(bedrooms=2), "2BR in Dubai Marina") == (
        "2br or in or dubai or marina"
    )


def test_clusters_come_from_the_latest_detect_run(search_db):
    settings, listings, _ = search_db
    conn = settings.connect()
    try:
        clusters = load_clusters(conn)
    finally:
        conn.close()
    n = listings.height
    assert len(clusters.size) == N_CLONES
    assert clusters.cluster_size(1) == 2 and clusters.cluster_size(n) == 2
    assert clusters.cluster_of[n - N_CLONES + 1] == clusters.cluster_of[1]
    assert clusters.cluster_size(100) == 1


def test_missing_detect_run_fails_loudly(search_db):
    settings, _, _ = search_db
    conn = settings.connect()
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE listings.detect_runs CASCADE")
        with pytest.raises(RuntimeError, match="python -m listings detect"):
            load_clusters(conn)
    finally:
        conn.close()


def _retrieve(settings, text, embedder, config=CONFIG):
    conn = settings.connect()
    try:
        lexicon, clusters = load_lexicon(conn), load_clusters(conn)
        parsed = parse(text, lexicon)
        vector = embedder.embed_texts([text])[0]
        return parsed, retrieve(conn, parsed, text, vector, clusters, config)
    finally:
        conn.close()


def test_area_and_type_filter_both_channels(search_db, fake_embedder):
    settings, listings, _ = search_db
    _, out = _retrieve(settings, "2 bed apartment in Dubai Marina with balcony", fake_embedder)
    assert out.height > 0
    attributes = listings.select("listing_id", "area_id", "property_type", "property_sub_type")
    joined = out.join(attributes, on="listing_id")
    assert (joined["area_id"] == 1).all()
    kinds = {listing_kind(t, s) for t, s in joined.select("property_type", "property_sub_type").rows()}
    assert kinds == {"flat"}
    assert out["fused_pos"].to_list() == list(range(1, out.height + 1))
    assert out["fulltext_pos"].null_count() < out.height  # "balcony" hits the full-text channel
    assert out.schema == pl.Schema(CANDIDATE_SCHEMA)


def test_no_listing_matches_an_impossible_filter(search_db, fake_embedder):
    settings, _, _ = search_db
    _, out = _retrieve(settings, "villa in Dubai Marina", fake_embedder)  # no villas there
    assert out.height == 0 and out.schema == pl.Schema(CANDIDATE_SCHEMA)


def test_unfiltered_query_ranks_by_similarity_and_collapses_clusters(search_db, fake_embedder):
    settings, listings, _ = search_db
    wide = dataclasses.replace(CONFIG, semantic_k=500, fulltext_k=500, candidate_k=500)
    parsed, out = _retrieve(settings, "family home with sea view", fake_embedder, wide)
    assert not parsed.area_ids and parsed.property_type is None
    assert out.height >= 150  # most of the 230 distinct listings in the tiny corpus
    semantic = out.filter(pl.col("semantic_pos").is_not_null()).sort("semantic_pos")
    assert semantic["semantic_cos"].is_sorted(descending=True)
    ids = set(out["listing_id"])
    first_clone = listings.height - N_CLONES + 1
    for source in range(1, N_CLONES + 1):
        assert not {source, first_clone + source - 1} <= ids  # never both members of a cluster


def test_candidate_k_caps_the_result(search_db, fake_embedder):
    settings, _, _ = search_db
    config = dataclasses.replace(CONFIG, candidate_k=5)
    _, out = _retrieve(settings, "apartment", fake_embedder, config)
    assert out.height == 5


def test_an_empty_query_retrieves_nothing(search_db, fake_embedder):
    settings, _, _ = search_db
    parsed, out = _retrieve(settings, "under 5", fake_embedder)
    assert parsed.is_empty and out.height == 0


def test_build_query_set_grades_every_candidate(search_db, fake_embedder):
    settings, _, corpus_run_id = search_db
    config = dataclasses.replace(CONFIG, n_queries=60)
    conn = settings.connect()
    try:
        lexicon = load_lexicon(conn)
        queries, judgments = build_query_set(conn, fake_embedder, lexicon, config)
        replace_query_set(conn, queries, judgments)
        conn.commit()
        stored_queries, stored = read_queries(conn), read_judgments(conn)
    finally:
        conn.close()
    assert queries.height == 60 and (queries["corpus_run_id"] == corpus_run_id).all()
    assert stored_queries.height == 60 and stored.height == judgments.height > 0
    assert set(stored["grade"].unique()) <= {0, 1, 2, 3}
    assert set(stored["query_id"]) <= set(stored_queries["query_id"])
    per_query = stored.group_by("query_id").agg(
        pl.col("fused_pos").min().alias("first"),
        pl.len().alias("n"),
        pl.col("fused_pos").max().alias("last"),
        (pl.col("grade") == 3).sum().alias("judged3"),
    )
    assert (per_query["first"] == 1).all() and (per_query["n"] == per_query["last"]).all()
    checked = per_query.join(stored_queries, on="query_id")
    assert (checked["judged3"] <= checked["n_grade3"]).all()
    no_match = checked.filter(pl.col("kind") == "no_match")
    assert (no_match["judged3"] == 0).all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/search/test_search_retrieve.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'search.retrieve'`.

- [ ] **Step 3: Write retrieval**

`search/retrieve.py`:

```python
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
    unique = [
        word for word in dict.fromkeys(words) if len(word) > 1 and word not in ("or", "and")
    ]
    return " or ".join(unique)


def semantic_channel(cur, vector, parsed: ParsedQuery, config: SearchConfig):
    where, params = _filters(parsed)
    literal = vector_literal(vector)
    cur.execute("SET LOCAL hnsw.ef_search = %s", (config.ef_search,))
    cur.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
    cur.execute(
        SEMANTIC_SQL.format(filters=where), [literal, *params, literal, config.semantic_k]
    )
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
```

- [ ] **Step 4: Add `build_query_set` to `search/queries.py`**

Add these imports next to the existing ones:

```python
import logging

from search.lexicon import Lexicon
from search.parse import parse
from search.retrieve import load_clusters, retrieve
from search.store import JUDGMENT_SCHEMA, latest_corpus_run
```

Add `LOGGER = logging.getLogger(__name__)` below the imports.

Append this function at the end of the file:

```python
def build_query_set(
    conn, embedder, lexicon: Lexicon, config: SearchConfig, log_every: int = 500
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Generate queries, retrieve each one's candidates, and grade them."""
    grading = load_grading_listings(conn)
    queries = generate_queries(grading, load_area_aliases(conn), config)
    corpus_run_id = latest_corpus_run(conn)
    clusters = load_clusters(conn)
    vectors = embedder.embed_texts(queries["text"].to_list())
    frames = []
    rows = queries.select("query_id", "text", "true_slots").iter_rows()
    for index, ((query_id, text, raw_slots), vector) in enumerate(zip(rows, vectors), start=1):
        candidates = retrieve(conn, parse(text, lexicon), text, vector, clusters, config)
        if candidates.height:
            listings = candidates.select("listing_id").join(
                grading, on="listing_id", how="left", maintain_order="left"
            )
            grades = grade_frame(true_slots(raw_slots), listings, config.near_margin)
            frames.append(
                candidates.with_columns(pl.lit(query_id, dtype=pl.Int64).alias("query_id"), grades)
            )
        if index % log_every == 0:
            LOGGER.info("retrieved and graded %s of %s queries", index, queries.height)
    judgments = (
        pl.concat(frames).select(list(JUDGMENT_SCHEMA)).cast(JUDGMENT_SCHEMA)
        if frames
        else pl.DataFrame(schema=JUDGMENT_SCHEMA)
    )
    queries = queries.with_columns(pl.lit(corpus_run_id, dtype=pl.Int64).alias("corpus_run_id"))
    return queries, judgments
```

Check for an import cycle. `search.retrieve` imports only `listings.load`, `search.config` and `search.parse`, so importing it from `search.queries` is safe.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q -W error tests/search/test_search_retrieve.py`
Expected: all tests PASS.

If `SET LOCAL hnsw.iterative_scan` raises `unrecognized configuration parameter`, check the extension version:
- Run `uv run python -c` with a query of `SELECT extversion FROM pg_extension WHERE extname='vector'` against the **test** database.
- Report the version and stop. The spec requires pgvector ≥ 0.8 (the project pins 0.8.6).

- [ ] **Step 6: Lint, full search tests, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
uv run pytest -q -W error tests/search
git add search/retrieve.py search/queries.py tests/search/test_search_retrieve.py
git commit -m "feat(search): semantic and full-text retrieval fused by RRF, collapsed by duplicate cluster

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 5: Features, price estimates, predicted flags, and the leakage scan

**Files:**
- Create:
  - `search/features.py`
  - `tests/search/test_search_features.py`

**Interfaces:**
- **Consumes:**
  - `search.config`: `FEATURES`, `KIND_SQL`, `SEARCH_FORBIDDEN`, `LABEL_READERS`, `SearchConfig`
  - `search.parse`: `ParsedQuery`, `parse`
  - `search.lexicon.Lexicon`
  - `search.store`:
    - `latest_corpus_run`
    - `estimates_corpus_run`, `replace_estimates`, `read_estimates`
    - `ESTIMATE_SCHEMA`
    - `read_queries`, `read_judgments`
  - `listings.fraud`:
    - `load_fraud_attributes`
    - `load_price_predictor`
    - `price_request`
    - `resolve_price_model_version`
    - `PriceInputError`
  - `ingestion.normalize`: `map_unique`, `match_key`
- **Produces (`search.features`):**
  - `ATTRIBUTE_SCHEMA`, and `load_listing_attributes(conn) -> pl.DataFrame`. The frame has:
    - `ATTRIBUTE_SCHEMA` columns, plus `building_key` and `project_key`;
    - no label columns.
  - `FLAG_COLUMNS: dict[str, str]`, and `load_predicted_flags(conn) -> pl.DataFrame`.
    - Columns: `listing_id` and the three `flag_*` columns, as 0.0/1.0.
  - `compute_estimates(rows: pl.DataFrame, predictor, log_every: int = 2000) -> pl.DataFrame`.
    - Columns: `ESTIMATE_SCHEMA`.
  - `refresh_estimates(conn, config: SearchConfig) -> dict[str, float]`.
    - Keys: `estimates_cached`, `value_features_skipped`, `estimates_written`, `estimates_unsupported`.
  - `reference_date(attributes: pl.DataFrame) -> date`: the latest `posted_at`.
  - `query_frame(parsed: dict[int, ParsedQuery]) -> pl.DataFrame`.
    - Columns: `query_id`, `q_area_ids`, `q_building_key`, `q_bedrooms`, `q_type`, `q_budget_min`, `q_budget_max`, `q_min_size`, `q_amenities`.
  - `build_features(candidates, queries, attributes, flags, estimates, reference: date) -> pl.DataFrame`.
    - Columns: `query_id`, `listing_id`, then `FEATURES` in order.
    - All features are Float64, and "not stated" or "unknown" is NaN.
    - Row order follows `candidates`.
  - `feature_table(conn, lexicon: Lexicon) -> pl.DataFrame`.
    - Columns: `query_id`, `split`, `kind`, `listing_id`, `grade`, `fused_pos`, then `FEATURES`.
    - One row per judgment, sorted by `query_id` then `fused_pos`.

**Feature definitions (exact).** "NaN" means the query did not state the slot, or the value is unknown.

| Feature | Value |
|---|---|
| `beds_diff` | abs(listing bedrooms − query bedrooms); NaN if either is null |
| `beds_stated` | 1.0 if the query states bedrooms, else 0.0 |
| `price_over_max` | asking ÷ budget_max − 1 |
| `price_under_min` | 1 − asking ÷ budget_min (positive when the listing is under the minimum) |
| `budget_stated` | 1.0 if either budget bound is stated |
| `size_ratio` | size_sqm ÷ min_size |
| `area_match` | 1.0 if the listing's area is in the parsed `area_ids`; NaN if no area was parsed |
| `area_stated` | 1.0 if an area was parsed |
| `type_match` | 1.0 if the listing kind equals the parsed type; NaN if no type |
| `type_stated` | 1.0 if a type was parsed |
| `building_match` | 1.0 if `building_key` or `project_key` equals `match_key(parsed.building)`; NaN if none |
| `amenity_hits` | number of parsed amenities found in the lower-cased description |
| `amenity_asked` | number of parsed amenities |
| `semantic_cos` | from retrieval; NaN when the listing came only from full text |
| `fulltext_rank` | from retrieval; 0.0 when the listing came only from semantic |
| `semantic_pos`, `fulltext_pos` | positions as float; NaN when absent |
| `rrf_score` | from retrieval |
| `price_to_estimate` | asking ÷ estimate; NaN when there is no estimate |
| `within_interval` | 1.0 if low ≤ asking ≤ high, 0.0 otherwise; NaN when there is no estimate |
| `flag_bait_price`, `flag_photo_reuse`, `flag_inconsistent_relist` | predicted flags from the latest detect run; 0.0 when absent |
| `cluster_size` | from retrieval |
| `days_since_posted` | days from `posted_at` to `reference` |

- [ ] **Step 1: Write the failing tests**

`tests/search/test_search_features.py`:

```python
import dataclasses
import math
import re
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from search_fixtures import PREDICTED_BAIT_IDS

import search.features as features_module
from models.price.predictor import PriceInputError
from search.config import FEATURES, LABEL_READERS, SEARCH_FORBIDDEN, SearchConfig
from search.features import (
    build_features,
    compute_estimates,
    feature_table,
    load_listing_attributes,
    load_predicted_flags,
    query_frame,
    reference_date,
    refresh_estimates,
)
from search.lexicon import load_lexicon
from search.parse import ParsedQuery
from search.queries import build_query_set
from search.retrieve import CANDIDATE_SCHEMA
from search.store import ESTIMATE_SCHEMA, read_estimates, replace_query_set

NAN = float("nan")


def _attributes() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "listing_id": [1, 2],
            "area_id": [1, 2],
            "kind": ["flat", "villa"],
            "building_key": ["marina gate", None],
            "project_key": ["project 1", "project 2"],
            "bedrooms": [2, None],
            "size_sqm": [100.0, 300.0],
            "asking_price_aed": [1_000_000.0, 3_000_000.0],
            "posted_at": [date(2023, 1, 1), date(2023, 1, 11)],
            "description": ["Features include balcony, sea view.", "Features include shared pool."],
        }
    )


def _candidates() -> pl.DataFrame:
    rows = [
        (1, 1, 0.9, 1, 0.2, 1, 0.03, 1, 2),
        (1, 2, None, None, 0.1, 2, 0.02, 2, 1),
        (2, 2, 0.5, 1, 0.0, None, 0.01, 1, 1),
    ]
    schema = {"query_id": pl.Int64, **CANDIDATE_SCHEMA}
    return pl.DataFrame(rows, schema=schema, orient="row")


PARSED = {
    1: ParsedQuery(
        area_ids=(1,),
        building="Marina Gate",
        bedrooms=3,
        property_type="flat",
        budget_max=900_000.0,
        min_size_sqm=80.0,
        amenities=("balcony", "shared pool"),
    ),
    2: ParsedQuery(free_text="quiet"),
}
FLAGS = pl.DataFrame(
    {
        "listing_id": [2],
        "flag_bait_price": [1.0],
        "flag_photo_reuse": [0.0],
        "flag_inconsistent_relist": [0.0],
    }
)
ESTIMATES = pl.DataFrame(
    {"listing_id": [1], "estimate": [1_250_000.0], "low": [1_100_000.0], "high": [1_400_000.0]},
    schema=ESTIMATE_SCHEMA,
)


def _same(actual, expected):
    if isinstance(expected, float) and math.isnan(expected):
        return math.isnan(actual)
    return actual == pytest.approx(expected)


@pytest.fixture
def table():
    return build_features(
        _candidates(), query_frame(PARSED), _attributes(), FLAGS, ESTIMATES, date(2023, 1, 21)
    )


def test_columns_order_and_types(table):
    assert table.columns == ["query_id", "listing_id", *FEATURES]
    assert all(table.schema[name] == pl.Float64 for name in FEATURES)
    assert table.select("query_id", "listing_id").rows() == [(1, 1), (1, 2), (2, 2)]


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (
            0,
            {
                "beds_diff": 1.0, "beds_stated": 1.0, "price_over_max": 1e6 / 9e5 - 1,
                "price_under_min": NAN, "budget_stated": 1.0, "size_ratio": 1.25,
                "area_match": 1.0, "area_stated": 1.0, "type_match": 1.0, "type_stated": 1.0,
                "building_match": 1.0, "amenity_hits": 1.0, "amenity_asked": 2.0,
                "semantic_cos": 0.9, "fulltext_rank": 0.2, "semantic_pos": 1.0,
                "fulltext_pos": 1.0, "rrf_score": 0.03, "price_to_estimate": 0.8,
                "within_interval": 0.0, "flag_bait_price": 0.0, "flag_photo_reuse": 0.0,
                "flag_inconsistent_relist": 0.0, "cluster_size": 2.0, "days_since_posted": 20.0,
            },
        ),
        (
            1,
            {
                "beds_diff": NAN, "area_match": 0.0, "type_match": 0.0, "building_match": 0.0,
                "amenity_hits": 1.0, "semantic_cos": NAN, "semantic_pos": NAN,
                "fulltext_pos": 2.0, "price_to_estimate": NAN, "within_interval": NAN,
                "flag_bait_price": 1.0, "cluster_size": 1.0, "days_since_posted": 10.0,
                "size_ratio": 3.75,
            },
        ),
        (
            2,
            {
                "beds_diff": NAN, "beds_stated": 0.0, "price_over_max": NAN,
                "budget_stated": 0.0, "size_ratio": NAN, "area_match": NAN, "area_stated": 0.0,
                "type_match": NAN, "type_stated": 0.0, "building_match": NAN,
                "amenity_hits": 0.0, "amenity_asked": 0.0, "fulltext_rank": 0.0,
                "fulltext_pos": NAN,
            },
        ),
    ],
)  # fmt: skip
def test_feature_values(table, row, expected):
    values = table.row(row, named=True)
    for name, value in expected.items():
        assert _same(values[name], value), name


def test_within_interval_is_one_inside_the_range():
    estimates = ESTIMATES.with_columns(pl.lit(900_000.0).alias("low"))
    out = build_features(
        _candidates(), query_frame(PARSED), _attributes(), FLAGS, estimates, date(2023, 1, 21)
    )
    assert out["within_interval"][0] == 1.0


def test_value_features_are_nan_without_estimates():
    empty = pl.DataFrame(schema=ESTIMATE_SCHEMA)
    out = build_features(
        _candidates(), query_frame(PARSED), _attributes(), FLAGS, empty, date(2023, 1, 21)
    )
    assert out["price_to_estimate"].is_nan().all() and out["within_interval"].is_nan().all()


def test_empty_candidates_give_an_empty_table():
    empty = pl.DataFrame(schema={"query_id": pl.Int64, **CANDIDATE_SCHEMA})
    out = build_features(
        empty, query_frame(PARSED), _attributes(), FLAGS, ESTIMATES, date(2023, 1, 21)
    )
    assert out.height == 0 and out.columns == ["query_id", "listing_id", *FEATURES]


def test_attributes_flags_and_reference_date_load_from_postgres(search_db):
    settings, listings, _ = search_db
    conn = settings.connect()
    try:
        attributes = load_listing_attributes(conn)
        flags = load_predicted_flags(conn)
    finally:
        conn.close()
    assert attributes.height == listings.height
    assert not set(attributes.columns) & set(SEARCH_FORBIDDEN)
    assert {"building_key", "project_key", "kind", "description"} <= set(attributes.columns)
    assert sorted(flags["listing_id"]) == list(PREDICTED_BAIT_IDS)
    assert (flags["flag_bait_price"] == 1.0).all() and (flags["flag_photo_reuse"] == 0.0).all()
    assert reference_date(attributes) == listings["posted_at"].max()


class StubPredictor:
    def predict_one(self, request):
        if request["area_id"] == 6:
            raise PriceInputError("area_id", "unsupported in this stub")
        price = request["size_sqm"] * 10_000.0
        return SimpleNamespace(estimate_aed=price, range_80=(price * 0.9, price * 1.1))


def test_compute_estimates_skips_unsupported_rows(search_db):
    from listings.fraud import load_fraud_attributes

    settings, listings, _ = search_db
    conn = settings.connect()
    try:
        rows = load_fraud_attributes(conn)
    finally:
        conn.close()
    estimates = compute_estimates(rows, StubPredictor(), log_every=50)
    unsupported = listings.filter(pl.col("area_id") == 6).height
    assert estimates.height == listings.height - unsupported
    first = estimates.filter(pl.col("listing_id") == 1).row(0, named=True)
    size = listings.filter(pl.col("listing_id") == 1)["size_sqm"][0]
    assert first["estimate"] == pytest.approx(size * 10_000)
    assert first["low"] < first["estimate"] < first["high"]


def test_refresh_estimates_writes_once_per_corpus(search_db, monkeypatch):
    settings, listings, _ = search_db
    calls = []
    monkeypatch.setattr(
        features_module, "load_price_predictor", lambda uri: calls.append(uri) or StubPredictor()
    )
    monkeypatch.setattr(features_module, "resolve_price_model_version", lambda uri: "9")
    conn = settings.connect()
    try:
        first = refresh_estimates(conn, SearchConfig())
        conn.commit()
        second = refresh_estimates(conn, SearchConfig())
        stored = read_estimates(conn)
    finally:
        conn.close()
    assert first["estimates_cached"] == 0.0 and first["estimates_written"] == stored.height > 0
    assert first["estimates_unsupported"] == listings.filter(pl.col("area_id") == 6).height
    assert second["estimates_cached"] == 1.0 and len(calls) == 1


def test_refresh_estimates_without_a_price_model_skips_and_writes_nothing(search_db, monkeypatch):
    settings, _, _ = search_db
    monkeypatch.setattr(features_module, "load_price_predictor", lambda uri: None)
    conn = settings.connect()
    try:
        stats = refresh_estimates(conn, SearchConfig())
        stored = read_estimates(conn)
    finally:
        conn.close()
    assert stats["value_features_skipped"] == 1.0 and stored.height == 0


def test_feature_table_covers_every_judgment(search_db, fake_embedder):
    settings, _, _ = search_db
    config = dataclasses.replace(SearchConfig(), n_queries=40, ef_search=100)
    conn = settings.connect()
    try:
        lexicon = load_lexicon(conn)
        queries, judgments = build_query_set(conn, fake_embedder, lexicon, config)
        replace_query_set(conn, queries, judgments)
        conn.commit()
        table = feature_table(conn, lexicon)
    finally:
        conn.close()
    assert table.height == judgments.height
    assert table.columns[:6] == ["query_id", "split", "kind", "listing_id", "grade", "fused_pos"]
    assert table.columns[6:] == list(FEATURES)
    assert table["grade"].null_count() == 0 and table["split"].null_count() == 0
    assert table.select("query_id", "fused_pos").is_duplicated().sum() == 0
    assert table["price_to_estimate"].is_nan().all()  # no estimates were computed here


def test_no_search_module_outside_the_label_readers_names_a_label():
    """Labels, true slots and generator flags must never reach features, retrieval or the engine."""
    package = Path(features_module.__file__).parent
    exempt = {*LABEL_READERS, "config.py"}  # config.py only declares the forbidden list
    pattern = re.compile(r"\b(" + "|".join(SEARCH_FORBIDDEN) + r")\b")
    offenders = {
        path.name: sorted(set(pattern.findall(path.read_text(encoding="utf-8"))))
        for path in package.rglob("*.py")
        if path.name not in exempt and pattern.search(path.read_text(encoding="utf-8"))
    }
    assert offenders == {}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/search/test_search_features.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'search.features'`.

- [ ] **Step 3: Write the features module**

`search/features.py`:

```python
"""The ranker's features for (query, candidate) rows.

Inputs are the PARSED query, the listing's own attributes, the predicted Phase 4 flags and the
Phase 3 price estimates. Nothing here reads a ground-truth label (see the leakage test).
"""

import logging
from datetime import date

import polars as pl

from ingestion.normalize import map_unique, match_key
from listings.fraud import (
    PriceInputError,
    load_fraud_attributes,
    load_price_predictor,
    price_request,
    resolve_price_model_version,
)
from search.config import FEATURES, KIND_SQL, SearchConfig
from search.lexicon import Lexicon
from search.parse import ParsedQuery, parse
from search.store import (
    ESTIMATE_SCHEMA,
    estimates_corpus_run,
    latest_corpus_run,
    read_estimates,
    read_judgments,
    read_queries,
    replace_estimates,
)

LOGGER = logging.getLogger(__name__)
NAN = float("nan")
ATTRIBUTE_SQL = f"""
SELECT l.listing_id, l.area_id, l.area_name, {KIND_SQL} AS kind, l.building_name,
       l.project_name, l.bedrooms, l.size_sqm, l.asking_price_aed, l.posted_at, l.title,
       l.description
FROM listings.listings AS l
ORDER BY l.listing_id
"""
ATTRIBUTE_SCHEMA = {
    "listing_id": pl.Int64,
    "area_id": pl.Int64,
    "area_name": pl.Utf8,
    "kind": pl.Utf8,
    "building_name": pl.Utf8,
    "project_name": pl.Utf8,
    "bedrooms": pl.Int64,
    "size_sqm": pl.Float64,
    "asking_price_aed": pl.Float64,
    "posted_at": pl.Date,
    "title": pl.Utf8,
    "description": pl.Utf8,
}
FLAG_COLUMNS = {
    "bait_price": "flag_bait_price",
    "photo_reuse": "flag_photo_reuse",
    "inconsistent_relist": "flag_inconsistent_relist",
}
FLAGS_SQL = """
SELECT listing_id, flag FROM listings.fraud_flags
WHERE detect_run_id = (SELECT max(detect_run_id) FROM listings.detect_runs)
"""
QUERY_SCHEMA = {
    "query_id": pl.Int64,
    "q_area_ids": pl.List(pl.Int64),
    "q_building_key": pl.Utf8,
    "q_bedrooms": pl.Int64,
    "q_type": pl.Utf8,
    "q_budget_min": pl.Float64,
    "q_budget_max": pl.Float64,
    "q_min_size": pl.Float64,
    "q_amenities": pl.List(pl.Utf8),
}


def load_listing_attributes(conn) -> pl.DataFrame:
    with conn.cursor() as cur:
        cur.execute(ATTRIBUTE_SQL)
        frame = pl.DataFrame(cur.fetchall(), schema=ATTRIBUTE_SCHEMA, orient="row")
    return frame.with_columns(
        map_unique(frame["building_name"], match_key, pl.Utf8).alias("building_key"),
        map_unique(frame["project_name"], match_key, pl.Utf8).alias("project_key"),
    )


def load_predicted_flags(conn) -> pl.DataFrame:
    with conn.cursor() as cur:
        cur.execute(FLAGS_SQL)
        rows = cur.fetchall()
    frame = pl.DataFrame(rows, schema={"listing_id": pl.Int64, "flag": pl.Utf8}, orient="row")
    return (
        frame.filter(pl.col("flag").is_in(list(FLAG_COLUMNS)))
        .group_by("listing_id")
        .agg(
            (pl.col("flag") == flag).any().cast(pl.Float64).alias(column)
            for flag, column in FLAG_COLUMNS.items()
        )
        .sort("listing_id")
    )


def compute_estimates(rows: pl.DataFrame, predictor, log_every: int = 2_000) -> pl.DataFrame:
    """Phase 3 estimates for listing rows shaped like listings.fraud.load_fraud_attributes."""
    out = []
    for index, row in enumerate(rows.iter_rows(named=True), start=1):
        try:
            estimate = predictor.predict_one(price_request(row))
        except PriceInputError as exc:
            LOGGER.debug("listing %s cannot be priced: %s", row["listing_id"], exc)
            continue
        low, high = (float(value) for value in estimate.range_80)
        out.append((row["listing_id"], float(estimate.estimate_aed), low, high))
        if index % log_every == 0:
            LOGGER.info("priced %s of %s listings", index, rows.height)
    return pl.DataFrame(out, schema=ESTIMATE_SCHEMA, orient="row")


def refresh_estimates(conn, config: SearchConfig) -> dict[str, float]:
    """Recompute search.listing_estimates unless they already belong to the latest corpus."""
    corpus_run_id = latest_corpus_run(conn)
    stats = {
        "estimates_cached": 0.0,
        "value_features_skipped": 0.0,
        "estimates_written": 0.0,
        "estimates_unsupported": 0.0,
    }
    if estimates_corpus_run(conn) == corpus_run_id:
        stats["estimates_cached"] = 1.0
        return stats
    predictor = load_price_predictor(config.price_model_uri)
    if predictor is None:
        stats["value_features_skipped"] = 1.0
        return stats
    rows = load_fraud_attributes(conn)
    estimates = compute_estimates(rows, predictor)
    version = resolve_price_model_version(config.price_model_uri)
    written = replace_estimates(conn, estimates, corpus_run_id, version)
    stats["estimates_written"] = float(written)
    stats["estimates_unsupported"] = float(rows.height - estimates.height)
    return stats


def reference_date(attributes: pl.DataFrame) -> date:
    return attributes["posted_at"].max()


def query_frame(parsed: dict[int, ParsedQuery]) -> pl.DataFrame:
    rows = [
        {
            "query_id": query_id,
            "q_area_ids": list(query.area_ids),
            "q_building_key": match_key(query.building) if query.building else None,
            "q_bedrooms": query.bedrooms,
            "q_type": query.property_type,
            "q_budget_min": query.budget_min,
            "q_budget_max": query.budget_max,
            "q_min_size": query.min_size_sqm,
            "q_amenities": list(query.amenities),
        }
        for query_id, query in parsed.items()
    ]
    return pl.DataFrame(rows, schema=QUERY_SCHEMA)


def _flag(expression: pl.Expr) -> pl.Expr:
    return expression.fill_null(False).cast(pl.Float64)


def _when_stated(stated: pl.Expr, value: pl.Expr) -> pl.Expr:
    return pl.when(stated).then(value).otherwise(pl.lit(NAN))


def _amenity_hits(rows: pl.DataFrame) -> pl.DataFrame:
    exploded = (
        rows.select("_row", pl.col("description").str.to_lowercase(), "q_amenities")
        .explode("q_amenities")
        .filter(pl.col("q_amenities").is_not_null())
    )
    hits = exploded.group_by("_row").agg(
        pl.col("description")
        .str.contains(pl.col("q_amenities").str.to_lowercase(), literal=True)
        .sum()
        .cast(pl.Float64)
        .alias("amenity_hits")
    )
    return rows.join(hits, on="_row", how="left", maintain_order="left")


def build_features(
    candidates: pl.DataFrame,
    queries: pl.DataFrame,
    attributes: pl.DataFrame,
    flags: pl.DataFrame,
    estimates: pl.DataFrame,
    reference: date,
) -> pl.DataFrame:
    attribute_columns = [
        "listing_id", "area_id", "kind", "building_key", "project_key", "bedrooms",
        "size_sqm", "asking_price_aed", "posted_at", "description",
    ]  # fmt: skip
    rows = (
        candidates.with_row_index("_row")
        .join(queries, on="query_id", how="left", maintain_order="left")
        .join(attributes.select(attribute_columns), on="listing_id", how="left", maintain_order="left")
        .join(flags, on="listing_id", how="left", maintain_order="left")
        .join(estimates, on="listing_id", how="left", maintain_order="left")
    )
    rows = _amenity_hits(rows)
    price = pl.col("asking_price_aed")
    beds_stated = pl.col("q_bedrooms").is_not_null()
    area_stated = pl.col("q_area_ids").list.len().fill_null(0) > 0
    type_stated = pl.col("q_type").is_not_null()
    building_stated = pl.col("q_building_key").is_not_null()
    has_estimate = pl.col("estimate").is_not_null()
    same_building = (pl.col("building_key") == pl.col("q_building_key")) | (
        pl.col("project_key") == pl.col("q_building_key")
    )
    expressions = {
        "beds_diff": (pl.col("bedrooms") - pl.col("q_bedrooms")).abs().cast(pl.Float64),
        "beds_stated": beds_stated.cast(pl.Float64),
        "price_over_max": price / pl.col("q_budget_max") - 1.0,
        "price_under_min": 1.0 - price / pl.col("q_budget_min"),
        "budget_stated": _flag(
            pl.col("q_budget_min").is_not_null() | pl.col("q_budget_max").is_not_null()
        ),
        "size_ratio": pl.col("size_sqm") / pl.col("q_min_size"),
        "area_match": _when_stated(
            area_stated, _flag(pl.col("q_area_ids").list.contains(pl.col("area_id")))
        ),
        "area_stated": area_stated.cast(pl.Float64),
        "type_match": _when_stated(type_stated, _flag(pl.col("kind") == pl.col("q_type"))),
        "type_stated": type_stated.cast(pl.Float64),
        "building_match": _when_stated(building_stated, _flag(same_building)),
        "amenity_hits": pl.col("amenity_hits").fill_null(0.0),
        "amenity_asked": pl.col("q_amenities").list.len().fill_null(0).cast(pl.Float64),
        "semantic_cos": pl.col("semantic_cos"),
        "fulltext_rank": pl.col("fulltext_rank").fill_null(0.0),
        "semantic_pos": pl.col("semantic_pos").cast(pl.Float64),
        "fulltext_pos": pl.col("fulltext_pos").cast(pl.Float64),
        "rrf_score": pl.col("rrf_score"),
        "price_to_estimate": price / pl.col("estimate"),
        "within_interval": _when_stated(
            has_estimate, _flag((price >= pl.col("low")) & (price <= pl.col("high")))
        ),
        **{column: pl.col(column).fill_null(0.0) for column in FLAG_COLUMNS.values()},
        "cluster_size": pl.col("cluster_size").cast(pl.Float64),
        "days_since_posted": (pl.lit(reference) - pl.col("posted_at"))
        .dt.total_days()
        .cast(pl.Float64),
    }
    assert tuple(expressions) == FEATURES, "feature expressions out of order"
    return rows.sort("_row").select(
        "query_id",
        "listing_id",
        *(
            expression.cast(pl.Float64).fill_null(NAN).alias(name)
            for name, expression in expressions.items()
        ),
    )


def feature_table(conn, lexicon: Lexicon) -> pl.DataFrame:
    """Every stored judgment with its query's split and kind, its grade, and the features."""
    queries = read_queries(conn)
    judgments = read_judgments(conn)
    attributes = load_listing_attributes(conn)
    parsed = {
        query_id: parse(text, lexicon)
        for query_id, text in queries.select("query_id", "text").iter_rows()
    }
    table = build_features(
        judgments.drop("grade"),
        query_frame(parsed),
        attributes,
        load_predicted_flags(conn),
        read_estimates(conn),
        reference_date(attributes),
    )
    labels = judgments.select("query_id", "listing_id", "grade", "fused_pos").join(
        queries.select("query_id", "split", "kind"), on="query_id", how="left"
    )
    return (
        table.join(labels, on=["query_id", "listing_id"], how="left", maintain_order="left")
        .select("query_id", "split", "kind", "listing_id", "grade", "fused_pos", *FEATURES)
        .sort("query_id", "fused_pos")
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -W error tests/search/test_search_features.py`
Expected: all tests PASS.

Two Polars behaviours to check if a test fails:

1. `str.contains(<expression>, literal=True)`. If this Polars version rejects an expression pattern, compute `amenity_hits` another way: in Python, over `rows.select("_row", "description", "q_amenities").iter_rows()`. Keep the same output.
2. `list.contains(<expression>)`. Keep the element-wise semantics: one row's list checked against that same row's `area_id`.

Do not change any expected feature value.

- [ ] **Step 5: Lint, full search tests, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
uv run pytest -q -W error tests/search
git add search/features.py tests/search/test_search_features.py
git commit -m "feat(search): ranker features from the parsed query, flags and price estimates

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 6: Ranking metrics (pure)

**Files:**
- Create: `search/metrics.py`, `tests/search/test_search_metrics.py`

**Interfaces:**
- Consumes: nothing beyond numpy and polars.
- Produces, from `search.metrics`:
  - `ndcg_at_k(ranked_grades: list[int], k: int = 10) -> float`. Gains are `2^g − 1` and the discount is `log2(position + 1)`. Returns NaN when the ideal DCG is 0.
  - `reciprocal_rank(ranked_grades, min_grade: int = 2) -> float`. Returns 0.0 when no result reaches `min_grade`.
  - `precision_at_k(ranked_grades, k: int = 5, grade: int = 3) -> float`. The denominator is always `k`.
  - `mean_top_grade(ranked_grades, k: int = 10) -> float`. Returns NaN for an empty list.
  - `per_query_metrics(frame: pl.DataFrame, score: str, k: int = 10) -> pl.DataFrame`.
    - Input columns: `query_id`, `kind`, `grade`, `fused_pos`, and the column named by `score`.
    - Ranking: rows are ranked within each query by `score` descending, with ties broken by `fused_pos` ascending.
    - Output columns: `query_id, kind, ndcg, rr, p5, mean_grade_top, answerable`.
    - `answerable` means the query's kind is not `no_match` and at least one of its candidates has grade ≥ 1.
    - The output also has a `top_ids` column: the list of listing ids in ranked order, truncated to `k`. This needs `listing_id` in the input.
  - `summarize(per_query: pl.DataFrame) -> dict[str, float]`, with these keys:
    - `ndcg_at_10`, `mrr` and `precision_at_5`: means over answerable queries;
    - `queries`: the number of answerable queries;
    - `no_match.mean_grade_top10` and `no_match.queries`;
    - `kind.<kind>.ndcg_at_10`, for each answerable kind that is present.
  - `bootstrap_ci(values: np.ndarray, n: int = 1000, seed: int = 7, alpha: float = 0.05) -> tuple[float, float]`. Resamples query-level values with replacement and returns percentile bounds of the mean; `(nan, nan)` when `values` is empty.

The key names say `_at_10` and `precision_at_5` because the spec fixes k = 10 and k = 5. `per_query_metrics` still takes `k` so that the tests can use small lists.

- [ ] **Step 1: Write the failing tests**

`tests/search/test_search_metrics.py`:

```python
import math

import numpy as np
import polars as pl
import pytest

from search.metrics import (
    bootstrap_ci,
    mean_top_grade,
    ndcg_at_k,
    per_query_metrics,
    precision_at_k,
    reciprocal_rank,
    summarize,
)


def test_ndcg_by_hand():
    assert ndcg_at_k([3, 2, 0]) == pytest.approx(1.0)
    dcg = 3 / math.log2(3) + 7 / 2
    ideal = 7 + 3 / math.log2(3)
    assert ndcg_at_k([0, 2, 3]) == pytest.approx(dcg / ideal)
    assert ndcg_at_k([0] * 10 + [3], k=10) == 0.0
    assert math.isnan(ndcg_at_k([0, 0, 0]))
    assert math.isnan(ndcg_at_k([]))


def test_reciprocal_rank_precision_and_mean_grade():
    assert reciprocal_rank([0, 1, 2]) == pytest.approx(1 / 3)
    assert reciprocal_rank([0, 1]) == 0.0
    assert reciprocal_rank([1, 0], min_grade=1) == 1.0
    assert precision_at_k([3, 3, 0]) == pytest.approx(0.4)
    assert precision_at_k([3, 2, 3, 3, 3, 3], k=5) == pytest.approx(0.8)
    assert mean_top_grade([3, 1]) == 2.0
    assert mean_top_grade([1] * 12 + [3], k=10) == 1.0
    assert math.isnan(mean_top_grade([]))


FRAME = pl.DataFrame(
    {
        "query_id": [1, 1, 1, 2, 2, 3, 3, 4],
        "kind": ["specified"] * 3 + ["vague"] * 2 + ["no_match"] * 2 + ["specified"],
        "listing_id": [10, 11, 12, 20, 21, 30, 31, 40],
        "grade": [0, 3, 2, 2, 2, 1, 0, 0],
        "fused_pos": [1, 2, 3, 1, 2, 1, 2, 1],
        "score": [0.1, 0.9, 0.9, 0.5, 0.5, 0.2, 0.3, 0.0],
    }
)


def test_per_query_metrics_rank_by_score_then_fused_position():
    out = per_query_metrics(FRAME, "score", k=10).sort("query_id")
    rows = {row["query_id"]: row for row in out.to_dicts()}
    assert rows[1]["top_ids"] == [11, 12, 10]  # tie at 0.9 broken by fused_pos
    assert rows[1]["ndcg"] == pytest.approx(1.0) and rows[1]["rr"] == 1.0
    assert rows[1]["p5"] == pytest.approx(0.2)
    assert rows[2]["top_ids"] == [20, 21] and rows[2]["kind"] == "vague"
    assert rows[3]["top_ids"] == [31, 30] and rows[3]["mean_grade_top"] == 0.5
    assert [rows[q]["answerable"] for q in (1, 2, 3, 4)] == [True, True, False, False]
    assert math.isnan(rows[4]["ndcg"])


def test_summarize_uses_answerable_queries_only():
    summary = summarize(per_query_metrics(FRAME, "score", k=10))
    assert summary["queries"] == 2.0
    assert summary["ndcg_at_10"] == pytest.approx(1.0)
    assert summary["mrr"] == pytest.approx(1.0)
    assert summary["precision_at_5"] == pytest.approx(0.1)
    assert summary["no_match.queries"] == 1.0
    assert summary["no_match.mean_grade_top10"] == pytest.approx(0.5)
    assert summary["kind.specified.ndcg_at_10"] == pytest.approx(1.0)
    assert summary["kind.vague.ndcg_at_10"] == pytest.approx(1.0)
    assert "kind.no_match.ndcg_at_10" not in summary


def test_summarize_on_nothing_answerable_is_nan_not_an_error():
    only_no_match = FRAME.filter(pl.col("kind") == "no_match")
    summary = summarize(per_query_metrics(only_no_match, "score"))
    assert summary["queries"] == 0.0 and math.isnan(summary["ndcg_at_10"])


def test_bootstrap_ci():
    values = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    low, high = bootstrap_ci(values, n=500, seed=3)
    assert low <= values.mean() <= high
    assert (low, high) == bootstrap_ci(values, n=500, seed=3)
    assert bootstrap_ci(np.array([0.5, 0.5]), n=50) == (0.5, 0.5)
    assert all(math.isnan(bound) for bound in bootstrap_ci(np.array([])))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/search/test_search_metrics.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'search.metrics'`.

- [ ] **Step 3: Write the metrics**

`search/metrics.py`:

```python
"""Ranking metrics over per-query result lists. Pure: no database, no MLflow."""

import math

import numpy as np
import polars as pl

NAN = float("nan")
PER_QUERY_SCHEMA = {
    "query_id": pl.Int64,
    "kind": pl.Utf8,
    "top_ids": pl.List(pl.Int64),
    "ndcg": pl.Float64,
    "rr": pl.Float64,
    "p5": pl.Float64,
    "mean_grade_top": pl.Float64,
    "answerable": pl.Boolean,
}


def _dcg(grades, k: int) -> float:
    return sum((2.0**grade - 1.0) / math.log2(index + 2) for index, grade in enumerate(grades[:k]))


def ndcg_at_k(ranked_grades, k: int = 10) -> float:
    ideal = _dcg(sorted(ranked_grades, reverse=True), k)
    return NAN if ideal == 0 else _dcg(list(ranked_grades), k) / ideal


def reciprocal_rank(ranked_grades, min_grade: int = 2) -> float:
    for index, grade in enumerate(ranked_grades):
        if grade >= min_grade:
            return 1.0 / (index + 1)
    return 0.0


def precision_at_k(ranked_grades, k: int = 5, grade: int = 3) -> float:
    return sum(value == grade for value in list(ranked_grades)[:k]) / k


def mean_top_grade(ranked_grades, k: int = 10) -> float:
    top = list(ranked_grades)[:k]
    return float(np.mean(top)) if top else NAN


def per_query_metrics(frame: pl.DataFrame, score: str, k: int = 10) -> pl.DataFrame:
    grouped = (
        frame.sort(["query_id", score, "fused_pos"], descending=[False, True, False])
        .group_by("query_id", maintain_order=True)
        .agg(pl.col("kind").first(), pl.col("grade"), pl.col("listing_id"))
    )
    rows = []
    for query_id, kind, grades, listing_ids in grouped.iter_rows():
        rows.append(
            {
                "query_id": query_id,
                "kind": kind,
                "top_ids": listing_ids[:k],
                "ndcg": ndcg_at_k(grades, k),
                "rr": reciprocal_rank(grades),
                "p5": precision_at_k(grades, 5),
                "mean_grade_top": mean_top_grade(grades, k),
                "answerable": kind != "no_match" and max(grades) >= 1,
            }
        )
    return pl.DataFrame(rows, schema=PER_QUERY_SCHEMA)


def _mean(series: pl.Series) -> float:
    return float(series.mean()) if series.len() else NAN


def summarize(per_query: pl.DataFrame) -> dict[str, float]:
    answerable = per_query.filter(pl.col("answerable"))
    no_match = per_query.filter(pl.col("kind") == "no_match")
    summary = {
        "ndcg_at_10": _mean(answerable["ndcg"]),
        "mrr": _mean(answerable["rr"]),
        "precision_at_5": _mean(answerable["p5"]),
        "queries": float(answerable.height),
        "no_match.mean_grade_top10": _mean(no_match["mean_grade_top"]),
        "no_match.queries": float(no_match.height),
    }
    for (kind,), group in answerable.group_by("kind", maintain_order=True):
        summary[f"kind.{kind}.ndcg_at_10"] = _mean(group["ndcg"])
    return summary


def bootstrap_ci(
    values: np.ndarray, n: int = 1_000, seed: int = 7, alpha: float = 0.05
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return NAN, NAN
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(n, values.size))].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -W error tests/search/test_search_metrics.py`
Expected: all tests PASS.

- [ ] **Step 5: Lint, full search tests, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
uv run pytest -q -W error tests/search
git add search/metrics.py tests/search/test_search_metrics.py
git commit -m "feat(search): NDCG, MRR, precision and bootstrap intervals per query

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 7: Rankers, Optuna tuning, pyfunc, and the registration gate

**Files:**
- Create:
  - `search/ranker.py`
  - `search/train.py`
  - `tests/search/test_search_train.py`

**Interfaces:**

*Consumes:*
- `search.config` (`FEATURES`, `TRUST_FEATURES`, `SearchConfig`)
- `search.metrics` (`per_query_metrics`, `summarize`)
- `models.price.registry.register_champion(model_uri, name) -> str`
- the Task 5 `feature_table` shape: `query_id, split, kind, listing_id, grade, fused_pos, *FEATURES`, sorted by `query_id`

*Produces, from `search.ranker`:*
- `Ranker`: the base class.
  - Attributes: `kind: str`, `features: tuple[str, ...]`, `params: dict`, `best_iteration: int`.
  - Methods:
    - `.score(frame: pl.DataFrame) -> np.ndarray`
    - `.importance() -> pl.DataFrame`, with columns `feature` and `gain`
    - `.save(directory: Path) -> Path`
- `XGBRanker.fit(train, tune, features, params, config) -> XGBRanker`
- `LGBMRanker.fit(train, tune, features, params, config) -> LGBMRanker`
- `RANKER_CLASSES: dict[str, type[Ranker]]`, keyed by `"xgboost"` and `"lightgbm"`
- `load_ranker(directory: Path) -> Ranker`
- `RankerPyfunc`: an `mlflow.pyfunc.PythonModel` whose `.ranker` is loaded in `load_context`
- `log_ranker(ranker, directory: Path) -> str`, which returns the model URI
- `load_champion(uri: str) -> Ranker | None`. It returns `None` and logs a warning naming the tracking URI when the load fails.

*Produces, from `search.train`:*
- `mean_ndcg(ranker, frame, k) -> float`
- `tune_ranker(kind, train, tune, features, config, n_trials) -> tuple[Ranker, float, dict]`, returning the ranker, its tune NDCG and the best parameters
- `TrainingResult(rankers: dict[str, Ranker], tune_ndcg: dict[str, float], best_params: dict[str, dict], winner: str)`
- `train_rankers(table, config, n_trials) -> TrainingResult`
- `fit_ablation(table, result, config) -> Ranker`: the winner's kind and best parameters, trained on `FEATURES` minus `TRUST_FEATURES`
- `gate_passes(ndcg: float, ci_low: float, baseline_ndcg: float) -> bool`
- `register_ranker(ranker, config, directory) -> str`, returning the version

**The "refit" in the spec.**
- Each fit trains on the train split, with early stopping on the tune split.
- The model keeps `best_iteration`, and every prediction uses only the trees up to that iteration.
- That is the same model a refit on the train split with that iteration count would produce, so no second fit is needed.
- `tune_ranker` does refit once more with the best parameters, so that the returned object is the study's best trial and not the last one tried.

**Devices.**
- XGBoost trains on `config.device`: `"cuda"` in the real run, `"cpu"` in tests.
- After training, the booster is switched to CPU. Scoring numpy arrays then raises no device-mismatch warning, which matters under `-W error`.
- LightGBM always runs on the CPU.

- [ ] **Step 1: Write the failing tests**

`tests/search/test_search_train.py`:

```python
import dataclasses
import logging

import numpy as np
import polars as pl
import pytest

from search.config import FEATURES, TRUST_FEATURES, SearchConfig
from search.ranker import (
    RANKER_CLASSES,
    LGBMRanker,
    XGBRanker,
    load_champion,
    load_ranker,
    log_ranker,
)
from search.train import (
    fit_ablation,
    gate_passes,
    mean_ndcg,
    register_ranker,
    train_rankers,
    tune_ranker,
)

CONFIG = dataclasses.replace(
    SearchConfig(), device="cpu", max_rounds=60, early_stopping_rounds=10, n_trials=2
)


def learnable_table(n_queries: int = 90, per_query: int = 12, seed: int = 0) -> pl.DataFrame:
    """Grades follow area_match and beds_diff; fused_pos is shuffled, so order must be learned."""
    rng = np.random.default_rng(seed)
    rows = []
    for query_id in range(1, n_queries + 1):
        split = {3: "tune", 4: "report"}.get(query_id % 5, "train")  # 60 / 20 / 20
        positions = rng.permutation(per_query) + 1
        for index in range(per_query):
            area = float(rng.random() < 0.5)
            beds = float(rng.integers(0, 3))
            grade = int(area * 2 + (beds == 0) * 1) if query_id % 7 else 0
            features = {name: float(rng.random()) for name in FEATURES}
            features.update(area_match=area, beds_diff=beds)
            rows.append(
                {
                    "query_id": query_id,
                    "split": split,
                    "kind": "no_match" if query_id % 7 == 0 else "specified",
                    "listing_id": query_id * 100 + index,
                    "grade": grade,
                    "fused_pos": int(positions[index]),
                    **features,
                }
            )
    return pl.DataFrame(rows).sort("query_id", "fused_pos")


TABLE = learnable_table()
TRAIN = TABLE.filter(pl.col("split") == "train")
TUNE = TABLE.filter(pl.col("split") == "tune")
FUSED = TUNE.with_columns((-pl.col("fused_pos")).alias("score"))


def _fused_ndcg():
    from search.metrics import per_query_metrics, summarize

    return summarize(per_query_metrics(FUSED, "score"))["ndcg_at_10"]


@pytest.mark.parametrize("kind", ["xgboost", "lightgbm"])
def test_each_ranker_learns_the_signal(kind):
    params = {"max_depth": 3} if kind == "xgboost" else {"num_leaves": 7, "min_data_in_leaf": 5}
    ranker = RANKER_CLASSES[kind].fit(TRAIN, TUNE, FEATURES, params, CONFIG)
    assert ranker.kind == kind and ranker.features == FEATURES
    assert ranker.best_iteration >= 0
    scores = ranker.score(TUNE)
    assert scores.shape == (TUNE.height,) and np.isfinite(scores).all()
    assert mean_ndcg(ranker, TUNE, 10) > _fused_ndcg() + 0.1
    importance = ranker.importance()
    assert importance.columns == ["feature", "gain"]
    assert importance.sort("gain", descending=True)["feature"][0] in {"area_match", "beds_diff"}


@pytest.mark.parametrize("cls", [XGBRanker, LGBMRanker])
def test_save_and_load_round_trip(cls, tmp_path):
    ranker = cls.fit(TRAIN, TUNE, FEATURES, {}, CONFIG)
    loaded = load_ranker(ranker.save(tmp_path / "ranker"))
    assert type(loaded) is cls and loaded.best_iteration == ranker.best_iteration
    assert loaded.params == ranker.params
    np.testing.assert_allclose(loaded.score(TUNE), ranker.score(TUNE), rtol=1e-6)


def test_tune_ranker_returns_the_best_trial_deterministically():
    first = tune_ranker("lightgbm", TRAIN, TUNE, FEATURES, CONFIG, n_trials=3)
    again = tune_ranker("lightgbm", TRAIN, TUNE, FEATURES, CONFIG, n_trials=3)
    ranker, ndcg, params = first
    assert ndcg == pytest.approx(mean_ndcg(ranker, TUNE, 10))
    assert params == again[2] and ndcg == pytest.approx(again[1])
    assert {"num_leaves", "learning_rate"} <= set(params)


def test_train_rankers_picks_the_higher_tune_ndcg():
    result = train_rankers(TABLE, CONFIG, n_trials=2)
    assert set(result.rankers) == {"xgboost", "lightgbm"}
    best = max(result.tune_ndcg, key=result.tune_ndcg.get)
    assert result.winner == best
    assert set(result.best_params) == {"xgboost", "lightgbm"}


def test_the_report_split_never_reaches_training(monkeypatch):
    seen = []
    real_fit = XGBRanker.fit.__func__

    def spy(cls, train, tune, features, params, config):
        seen.append(set(train["split"]) | set(tune["split"]))
        return real_fit(cls, train, tune, features, params, config)

    monkeypatch.setattr(XGBRanker, "fit", classmethod(spy))
    train_rankers(TABLE, CONFIG, n_trials=1)
    assert seen and all("report" not in splits for splits in seen)


def test_ablation_drops_the_trust_features():
    result = train_rankers(TABLE, CONFIG, n_trials=1)
    ablation = fit_ablation(TABLE, result, CONFIG)
    assert ablation.kind == result.winner
    assert ablation.features == tuple(f for f in FEATURES if f not in TRUST_FEATURES)
    assert ablation.score(TUNE).shape == (TUNE.height,)


@pytest.mark.parametrize(
    ("ndcg", "ci_low", "baseline", "expected"),
    [(0.80, 0.75, 0.70, True), (0.80, 0.65, 0.70, False), (0.60, 0.55, 0.70, False),
     (0.70, 0.70, 0.70, False), (float("nan"), 0.1, 0.1, False)],
)  # fmt: skip
def test_gate(ndcg, ci_low, baseline, expected):
    assert gate_passes(ndcg, ci_low, baseline) is expected


def test_pyfunc_registration_and_champion_loading(temp_mlflow, tmp_path):
    import mlflow

    ranker = XGBRanker.fit(TRAIN, TUNE, FEATURES, {"max_depth": 3}, CONFIG)
    mlflow.create_experiment(
        "search-ranking-test", artifact_location=temp_mlflow["artifact_location"]
    )  # never the default ./mlruns
    mlflow.set_experiment("search-ranking-test")
    with mlflow.start_run():
        uri = log_ranker(ranker, tmp_path / "ranker")
    loaded = mlflow.pyfunc.load_model(uri)
    predicted = np.asarray(loaded.predict(TUNE.select(FEATURES).to_pandas()))
    np.testing.assert_allclose(predicted, ranker.score(TUNE), rtol=1e-6)

    config = dataclasses.replace(CONFIG, ranker_name="search-ranker-test")
    with mlflow.start_run():
        version = register_ranker(ranker, config, tmp_path / "registered")
    assert version == "1"
    champion = load_champion("models:/search-ranker-test@champion")
    assert champion is not None and champion.kind == "xgboost"
    np.testing.assert_allclose(champion.score(TUNE), ranker.score(TUNE), rtol=1e-6)


def test_missing_champion_is_none_with_a_warning(temp_mlflow, caplog):
    with caplog.at_level(logging.WARNING, logger="search.ranker"):
        assert load_champion("models:/no-such-ranker@champion") is None
    assert temp_mlflow["tracking_uri"] in caplog.text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/search/test_search_train.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'search.ranker'`.

- [ ] **Step 3: Write the rankers**

`search/ranker.py`:

```python
"""Two GBDT rankers behind one interface, and the MLflow pyfunc that serves either one."""

import importlib.metadata
import json
import logging
from pathlib import Path

import mlflow
import numpy as np
import polars as pl

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_PATHS = [str(REPO_ROOT / name) for name in ("search", "listings", "ingestion", "models")]
RUNTIME_PACKAGES = ("mlflow", "xgboost", "lightgbm", "polars", "pandas", "pyarrow", "numpy")
META_FILE = "ranker.json"


def _matrix(frame: pl.DataFrame, features) -> np.ndarray:
    return frame.select(list(features)).to_numpy().astype(np.float64)


def _group_sizes(frame: pl.DataFrame) -> np.ndarray:
    return (
        frame.group_by("query_id", maintain_order=True).len()["len"].to_numpy().astype(np.int64)
    )


class Ranker:
    kind = "base"
    model_file = ""

    def __init__(self, booster, features, params: dict, best_iteration: int):
        self.booster = booster
        self.features = tuple(features)
        self.params = dict(params)
        self.best_iteration = int(best_iteration)

    def score(self, frame: pl.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def importance(self) -> pl.DataFrame:
        raise NotImplementedError

    def _save_model(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    def _load_model(cls, path: Path):
        raise NotImplementedError

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._save_model(directory / self.model_file)
        meta = {
            "kind": self.kind,
            "features": list(self.features),
            "params": self.params,
            "best_iteration": self.best_iteration,
        }
        (directory / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return directory


class XGBRanker(Ranker):
    kind = "xgboost"
    model_file = "model.json"

    @classmethod
    def fit(cls, train, tune, features, params, config) -> "XGBRanker":
        import xgboost as xgb

        def dmatrix(frame):
            return xgb.DMatrix(
                _matrix(frame, features),
                label=frame["grade"].to_numpy(),
                qid=frame["query_id"].to_numpy(),
                feature_names=list(features),
            )

        booster_params = {
            "objective": "rank:ndcg",
            "eval_metric": f"ndcg@{config.ndcg_k}",
            "lambdarank_pair_method": "topk",
            "tree_method": "hist",
            "device": config.device,
            "seed": config.seed,
            **params,
        }
        booster = xgb.train(
            booster_params,
            dmatrix(train),
            num_boost_round=config.max_rounds,
            evals=[(dmatrix(tune), "tune")],
            early_stopping_rounds=config.early_stopping_rounds,
            verbose_eval=False,
        )
        booster.set_param({"device": "cpu"})
        return cls(booster, features, params, booster.best_iteration)

    def score(self, frame: pl.DataFrame) -> np.ndarray:
        import xgboost as xgb

        matrix = xgb.DMatrix(_matrix(frame, self.features), feature_names=list(self.features))
        return self.booster.predict(matrix, iteration_range=(0, self.best_iteration + 1))

    def importance(self) -> pl.DataFrame:
        gains = self.booster.get_score(importance_type="gain")
        return pl.DataFrame(
            {"feature": list(self.features), "gain": [gains.get(f, 0.0) for f in self.features]}
        )

    def _save_model(self, path: Path) -> None:
        self.booster.save_model(str(path))

    @classmethod
    def _load_model(cls, path: Path):
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(str(path))
        booster.set_param({"device": "cpu"})
        return booster


class LGBMRanker(Ranker):
    kind = "lightgbm"
    model_file = "model.txt"

    @classmethod
    def fit(cls, train, tune, features, params, config) -> "LGBMRanker":
        import lightgbm as lgb

        def dataset(frame, reference=None):
            return lgb.Dataset(
                _matrix(frame, features),
                label=frame["grade"].to_numpy(),
                group=_group_sizes(frame),
                feature_name=list(features),
                reference=reference,
                free_raw_data=False,
            )

        train_set = dataset(train)
        booster_params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [config.ndcg_k],
            "verbosity": -1,
            "seed": config.seed,
            "deterministic": True,
            "force_row_wise": True,
            **params,
        }
        booster = lgb.train(
            booster_params,
            train_set,
            num_boost_round=config.max_rounds,
            valid_sets=[dataset(tune, reference=train_set)],
            callbacks=[lgb.early_stopping(config.early_stopping_rounds, verbose=False)],
        )
        return cls(booster, features, params, booster.best_iteration)

    def score(self, frame: pl.DataFrame) -> np.ndarray:
        iterations = self.best_iteration if self.best_iteration > 0 else None
        return self.booster.predict(_matrix(frame, self.features), num_iteration=iterations)

    def importance(self) -> pl.DataFrame:
        gains = self.booster.feature_importance(importance_type="gain")
        return pl.DataFrame({"feature": list(self.features), "gain": [float(g) for g in gains]})

    def _save_model(self, path: Path) -> None:
        self.booster.save_model(str(path))

    @classmethod
    def _load_model(cls, path: Path):
        import lightgbm as lgb

        return lgb.Booster(model_file=str(path))


RANKER_CLASSES: dict[str, type[Ranker]] = {"xgboost": XGBRanker, "lightgbm": LGBMRanker}


def load_ranker(directory: Path) -> Ranker:
    directory = Path(directory)
    meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
    cls = RANKER_CLASSES[meta["kind"]]
    booster = cls._load_model(directory / cls.model_file)
    return cls(booster, meta["features"], meta["params"], meta["best_iteration"])


class RankerPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.ranker = load_ranker(Path(context.artifacts["ranker_dir"]))

    def predict(self, context, model_input, params=None):
        return self.ranker.score(pl.from_pandas(model_input))


def _pip_requirements() -> list[str]:
    return [f"{name}=={importlib.metadata.version(name)}" for name in RUNTIME_PACKAGES]


def log_ranker(ranker: Ranker, directory: Path) -> str:
    """Log the ranker as a pyfunc under the active run; returns its model URI."""
    info = mlflow.pyfunc.log_model(
        artifact_path="ranker",
        python_model=RankerPyfunc(),
        artifacts={"ranker_dir": str(ranker.save(directory))},
        code_paths=CODE_PATHS,
        pip_requirements=_pip_requirements(),
    )
    return info.model_uri


def load_champion(uri: str) -> Ranker | None:
    tracking_uri = mlflow.get_tracking_uri()
    try:
        return mlflow.pyfunc.load_model(uri).unwrap_python_model().ranker
    except Exception as exc:  # noqa: BLE001 — no champion is an expected state; callers fall back
        LOGGER.warning(
            "search ranker %s unavailable via MLflow tracking URI %s (%s)", uri, tracking_uri, exc
        )
        return None
```

- [ ] **Step 4: Write training**

`search/train.py`:

```python
"""Tune both rankers on the tune split, pick the winner, and gate its registration."""

import math
from dataclasses import dataclass
from pathlib import Path

import optuna
import polars as pl

from models.price.registry import register_champion
from search.config import FEATURES, TRUST_FEATURES, SearchConfig
from search.metrics import per_query_metrics, summarize
from search.ranker import RANKER_CLASSES, Ranker, log_ranker


def _xgboost_space(trial: optuna.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
    }


def _lightgbm_space(trial: optuna.Trial) -> dict:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 7, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 200, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": 1,
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
    }


SPACES = {"xgboost": _xgboost_space, "lightgbm": _lightgbm_space}


@dataclass(frozen=True)
class TrainingResult:
    rankers: dict[str, Ranker]
    tune_ndcg: dict[str, float]
    best_params: dict[str, dict]
    winner: str


def mean_ndcg(ranker: Ranker, frame: pl.DataFrame, k: int) -> float:
    scored = frame.with_columns(pl.Series("score", ranker.score(frame)))
    return summarize(per_query_metrics(scored, "score", k))["ndcg_at_10"]


def tune_ranker(kind: str, train, tune, features, config: SearchConfig, n_trials: int):
    cls, space = RANKER_CLASSES[kind], SPACES[kind]

    def objective(trial: optuna.Trial) -> float:
        params = space(trial)
        score = mean_ndcg(cls.fit(train, tune, features, params, config), tune, config.ndcg_k)
        return score if math.isfinite(score) else 0.0

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.seed)
    )
    study.optimize(objective, n_trials=n_trials)
    best_params = dict(study.best_params)
    if kind == "lightgbm":
        best_params["bagging_freq"] = 1
    ranker = cls.fit(train, tune, features, best_params, config)
    return ranker, mean_ndcg(ranker, tune, config.ndcg_k), best_params


def _splits(table: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    train = table.filter(pl.col("split") == "train").sort("query_id", "fused_pos")
    tune = table.filter(pl.col("split") == "tune").sort("query_id", "fused_pos")
    if train.height == 0 or tune.height == 0:
        raise ValueError("the feature table needs judged queries in both train and tune splits")
    return train, tune


def train_rankers(table: pl.DataFrame, config: SearchConfig, n_trials: int) -> TrainingResult:
    train, tune = _splits(table)
    rankers, tune_ndcg, best_params = {}, {}, {}
    for kind in RANKER_CLASSES:  # xgboost first: it wins ties
        rankers[kind], tune_ndcg[kind], best_params[kind] = tune_ranker(
            kind, train, tune, FEATURES, config, n_trials
        )
    winner = max(RANKER_CLASSES, key=lambda kind: tune_ndcg[kind])
    return TrainingResult(rankers, tune_ndcg, best_params, winner)


def fit_ablation(table: pl.DataFrame, result: TrainingResult, config: SearchConfig) -> Ranker:
    train, tune = _splits(table)
    features = tuple(name for name in FEATURES if name not in TRUST_FEATURES)
    cls = RANKER_CLASSES[result.winner]
    return cls.fit(train, tune, features, result.best_params[result.winner], config)


def gate_passes(ndcg: float, ci_low: float, baseline_ndcg: float) -> bool:
    values = (ndcg, ci_low, baseline_ndcg)
    if not all(math.isfinite(value) for value in values):
        return False
    return ndcg > baseline_ndcg and ci_low > baseline_ndcg


def register_ranker(ranker: Ranker, config: SearchConfig, directory: Path) -> str:
    """Log under the active MLflow run and move the champion alias to the new version."""
    return register_champion(log_ranker(ranker, Path(directory)), config.ranker_name)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q -W error tests/search/test_search_train.py`
Expected: all tests PASS.

If a library emits a warning that `-W error` turns into a failure, do not suppress it globally:
- An XGBoost device warning means a scoring path skipped `set_param({"device": "cpu"})`. Fix that path.
- An MLflow warning about a missing input example or signature should be filtered in that one test, with `pytest.mark.filterwarnings` naming the exact message. Report it.

- [ ] **Step 6: Lint, full search tests, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
uv run pytest -q -W error tests/search
git add search/ranker.py search/train.py tests/search/test_search_train.py
git commit -m "feat(search): XGBoost and LightGBM rankers tuned with Optuna, pyfunc and gated registry

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 8: Report-split evaluation, Phase 4 effects, parser accuracy, artifacts, MLflow

**Files:**
- Create:
  - `search/evaluate.py`
  - `tests/search/test_search_evaluate.py`

**Interfaces:**
- **Consumes:**
  - `search.metrics`: `per_query_metrics`, `summarize`, `bootstrap_ci`
  - `search.ranker.Ranker`
  - `search.train`: `train_rankers`, `fit_ablation`
  - `search.features.feature_table`
  - `search.store.read_queries`
  - `search.parse.parse`
  - `search.lexicon.Lexicon`
  - `search.config`: `SLOTS`, `SearchConfig`
  - `models.price.registry`: `configure`, `log_metrics`
  - `ingestion.normalize.match_key`
- **Produces (`search.evaluate`):**
  - `CONTENDERS`: `("xgboost", "lightgbm", "baseline_newest", "baseline_semantic", "baseline_fused")`
  - `baseline_scores(frame) -> dict[str, np.ndarray]`
  - `contender_scores(frame, rankers: dict[str, Ranker]) -> dict[str, np.ndarray]`
  - `evaluate_contenders(frame, scores, config) -> tuple[dict[str, float], dict[str, pl.DataFrame]]`
  - `top_k_effects(frame, scores, fraud_ids: set[int], k: int, prefix: str = "") -> dict[str, float]`
  - `load_fraud_listing_ids(conn) -> set[int]`
  - `load_true_slots(conn, split: str = "report") -> dict[int, dict]`
  - `slot_checks(parsed, truth) -> dict[str, tuple[bool, object, object]]`
    - Keys are exactly `SLOTS`.
  - `parser_accuracy(queries, slots, lexicon) -> tuple[dict[str, float], pl.DataFrame]`
    - The second item is the error table.
  - `retrieval_recall(queries, report_rows) -> float`
  - `Evaluation(metrics, per_query, parse_errors, report_queries)`, a frozen dataclass.
  - `evaluate_report(conn, table, rankers, lexicon, config, champion=None, ablation=None) -> Evaluation`
  - `write_artifacts(directory, evaluation, importance, timings) -> list[Path]`
    - Writes `ndcg_comparison.png`, `feature_importance.csv`, `per_query_report.csv`, `parse_errors.csv` and `timings.json`.
  - `search_run(config, run_name, tracking_uri=None, artifact_location=None)`, a context manager that yields the active MLflow run.
  - `log_results(metrics: dict, params: dict, artifacts: list[Path]) -> None`
    - Non-finite metrics are dropped.

**Metric names.**

| Group | Names |
|---|---|
| Per contender `<c>` | `<c>.ndcg_at_10`, `<c>.mrr`, `<c>.precision_at_5`, `<c>.queries`, `<c>.ci_low`, `<c>.ci_high` |
| By query kind | `kind.<kind>.<c>.ndcg_at_10`, `kind.no_match.<c>.mean_grade_top10`, `kind.no_match.<c>.queries` |
| Phase 4 effects (champion) | `dup.top10_removed`, `fraud.top10_share` |
| No-trust ablation | `ablation.no_trust.ndcg_at_10` (plus its CI, MRR and kind keys), `ablation.no_trust.dup.top10_removed`, `ablation.no_trust.fraud.top10_share` |
| Parser | `parse.<slot>.accuracy` |
| Retrieval | `retrieval.recall_at_200` |
| Report | `report.queries` |

**Baselines.**
- `baseline_newest` = −`days_since_posted`, so the newest listing comes first.
- `baseline_semantic` = −`semantic_pos`. Listings that came only from full text sort last.
- `baseline_fused` = −`fused_pos`.
- Ties are always broken by `fused_pos`, inside `per_query_metrics`.

**What each Phase 4 effect measures.**
- `dup.top10_removed`: the mean, over report queries, of Σ(`cluster_size` − 1) across the champion's top 10. It counts the duplicate listings the collapse kept out of the top 10.
- `fraud.top10_share`: the share of all top-10 slots, across report queries, held by a listing with a ground-truth fraud label.

**Slot checks.**
- **An unstated slot** counts as correct only if the parser also left it empty. For `area`, that means the parser left `area_name` empty; `area_ids` may still be set from a building.
- **Money and size** are correct within 1%.
- **Building** is compared with `match_key` on both sides.
- **Amenities** are compared as sets.

- [ ] **Step 1: Write the failing tests**

`tests/search/test_search_evaluate.py`:

```python
import dataclasses
import json
import math

import mlflow
import numpy as np
import polars as pl
import pytest
from search_fixtures import ALIASES, AREAS, BAIT_IDS, BUILDINGS

from search.config import FEATURES, SLOTS, SearchConfig
from search.evaluate import (
    CONTENDERS,
    Evaluation,
    baseline_scores,
    evaluate_contenders,
    evaluate_report,
    load_fraud_listing_ids,
    load_true_slots,
    log_results,
    parser_accuracy,
    retrieval_recall,
    search_run,
    slot_checks,
    top_k_effects,
    write_artifacts,
)
from search.features import feature_table
from search.lexicon import Lexicon, load_lexicon
from search.parse import parse
from search.queries import build_query_set
from search.store import replace_query_set
from search.train import fit_ablation, train_rankers

CONFIG = dataclasses.replace(
    SearchConfig(),
    device="cpu",
    max_rounds=30,
    early_stopping_rounds=5,
    n_bootstrap=200,
    ef_search=100,
)
LEXICON = Lexicon.from_rows(
    areas=AREAS,
    aliases=ALIASES,
    buildings=[(name, area) for area, names in BUILDINGS.items() for name in names if name],
    projects=[(f"Project {area}", area) for area, _ in AREAS],
)


def _frame() -> pl.DataFrame:
    base = {name: 0.0 for name in FEATURES}
    rows = [
        # query 1: two listings, the fresher one is worse
        {**base, "query_id": 1, "kind": "specified", "listing_id": 1, "grade": 3, "fused_pos": 2,
         "days_since_posted": 30.0, "semantic_pos": float("nan"), "cluster_size": 3.0},
        {**base, "query_id": 1, "kind": "specified", "listing_id": 11, "grade": 0, "fused_pos": 1,
         "days_since_posted": 1.0, "semantic_pos": 1.0, "cluster_size": 1.0},
        # query 2: no_match
        {**base, "query_id": 2, "kind": "no_match", "listing_id": 12, "grade": 1, "fused_pos": 1,
         "days_since_posted": 5.0, "semantic_pos": 1.0, "cluster_size": 1.0},
    ]  # fmt: skip
    return pl.DataFrame(rows)


class FixedRanker:
    def __init__(self, scores):
        self._scores = np.asarray(scores, dtype=float)

    def score(self, frame):
        return self._scores


def test_baselines_order_by_freshness_semantic_rank_and_fused_rank():
    scores = baseline_scores(_frame())
    assert list(scores) == ["baseline_newest", "baseline_semantic", "baseline_fused"]
    assert scores["baseline_newest"][1] > scores["baseline_newest"][0]
    assert scores["baseline_semantic"][0] < scores["baseline_semantic"][1]  # missing goes last
    assert list(scores["baseline_fused"]) == [-2.0, -1.0, -1.0]


def test_evaluate_contenders_names_every_metric():
    frame = _frame()
    scores = {"xgboost": np.array([1.0, 0.0, 0.0]), **baseline_scores(frame)}
    metrics, per_query = evaluate_contenders(frame, scores, CONFIG)
    assert metrics["xgboost.ndcg_at_10"] == pytest.approx(1.0)
    assert metrics["baseline_fused.ndcg_at_10"] == pytest.approx((2**0 - 1 + 7 / math.log2(3)) / 7)
    assert metrics["baseline_newest.mrr"] == pytest.approx(0.5)
    assert metrics["xgboost.ci_low"] == metrics["xgboost.ci_high"] == pytest.approx(1.0)
    assert metrics["kind.specified.xgboost.ndcg_at_10"] == pytest.approx(1.0)
    assert metrics["kind.no_match.xgboost.mean_grade_top10"] == pytest.approx(1.0)
    assert metrics["kind.no_match.xgboost.queries"] == 1.0
    assert metrics["xgboost.queries"] == 1.0
    assert set(per_query) == set(scores)


def test_top_k_effects_count_hidden_duplicates_and_fraud_slots():
    frame = _frame()
    effects = top_k_effects(frame, np.array([1.0, 0.0, 0.0]), fraud_ids={12}, k=10)
    assert effects["dup.top10_removed"] == pytest.approx((2 + 0) / 2)
    assert effects["fraud.top10_share"] == pytest.approx(1 / 3)
    prefixed = top_k_effects(frame, np.zeros(3), fraud_ids=set(), k=1, prefix="ablation.x.")
    assert prefixed["ablation.x.fraud.top10_share"] == 0.0


def _truth(**overrides):
    truth = {
        "area_id": 1, "area_name": "Dubai Marina", "building": None, "bedrooms": 2,
        "property_type": "flat", "budget_min": None, "budget_max": 1_500_000.0,
        "min_size_sqm": None, "amenities": ["balcony"],
    }  # fmt: skip
    return {**truth, **overrides}


def test_slot_checks():
    parsed = parse("2BR apartment in Dubai Marina under 1.5M with balcony", LEXICON)
    checks = slot_checks(parsed, _truth())
    assert tuple(checks) == SLOTS
    assert all(ok for ok, _, _ in checks.values())
    wrong = slot_checks(parsed, _truth(bedrooms=3, budget_max=1_600_000.0, amenities=[]))
    assert [slot for slot, (ok, _, _) in wrong.items() if not ok] == [
        "bedrooms", "budget_max", "amenities",
    ]  # fmt: skip
    near = slot_checks(parsed, _truth(budget_max=1_510_000.0))
    assert near["budget_max"][0]


def test_parser_accuracy_and_error_table():
    queries = pl.DataFrame(
        {"query_id": [1, 2], "text": ["2BR flat in Dubai Marina under 1.5M with balcony",
                                      "3BR flat in Dubai Marina under 1.5M with balcony"]}
    )  # fmt: skip
    accuracy, errors = parser_accuracy(queries, {1: _truth(), 2: _truth()}, LEXICON)
    assert accuracy["parse.bedrooms.accuracy"] == 0.5
    assert accuracy["parse.area.accuracy"] == 1.0
    assert set(accuracy) == {f"parse.{slot}.accuracy" for slot in SLOTS}
    assert errors.rows() == [(2, queries["text"][1], "bedrooms", "2", "3")]


def test_retrieval_recall():
    queries = pl.DataFrame({"query_id": [1, 2, 3], "n_grade3": [2, 0, 4]})
    rows = pl.DataFrame({"query_id": [1, 1, 3, 3], "grade": [3, 1, 3, 0]})
    assert retrieval_recall(queries, rows) == pytest.approx((1 / 2 + 1 / 4) / 2)
    assert math.isnan(retrieval_recall(queries.filter(pl.col("n_grade3") == 0), rows))


def test_the_database_evaluation_end_to_end(search_db, fake_embedder, tmp_path):
    settings, _, _ = search_db
    config = dataclasses.replace(CONFIG, n_queries=150)
    conn = settings.connect()
    try:
        lexicon = load_lexicon(conn)
        queries, judgments = build_query_set(conn, fake_embedder, lexicon, config)
        replace_query_set(conn, queries, judgments)
        conn.commit()
        assert load_fraud_listing_ids(conn) == set(BAIT_IDS)
        assert set(load_true_slots(conn)) == set(
            queries.filter(pl.col("split") == "report")["query_id"]
        )
        table = feature_table(conn, lexicon)
        result = train_rankers(table, config, n_trials=1)
        ablation = fit_ablation(table, result, config)
        evaluation = evaluate_report(
            conn, table, result.rankers, lexicon, config, champion=result.winner, ablation=ablation
        )
    finally:
        conn.close()
    metrics = evaluation.metrics
    for contender in CONTENDERS:
        assert 0.0 <= metrics[f"{contender}.ndcg_at_10"] <= 1.0
    assert 0.0 <= metrics["retrieval.recall_at_200"] <= 1.0
    assert metrics["parse.bedrooms.accuracy"] >= 0.95
    assert {"dup.top10_removed", "fraud.top10_share", "ablation.no_trust.ndcg_at_10"} <= set(
        metrics
    )
    assert metrics["report.queries"] == queries.filter(pl.col("split") == "report").height

    importance = result.rankers[result.winner].importance()
    paths = write_artifacts(tmp_path, evaluation, importance, {"queries": {"seconds": 1.5}})
    assert sorted(path.name for path in paths) == [
        "feature_importance.csv", "ndcg_comparison.png", "parse_errors.csv",
        "per_query_report.csv", "timings.json",
    ]  # fmt: skip
    report = pl.read_csv(tmp_path / "per_query_report.csv")
    assert {"query_id", "text", "kind", "parsed", *(f"ndcg.{c}" for c in CONTENDERS)} <= set(
        report.columns
    )
    assert json.loads((tmp_path / "timings.json").read_text()) == {"queries": {"seconds": 1.5}}
    assert isinstance(evaluation, Evaluation)


def test_search_run_logs_finite_metrics_and_artifacts(temp_mlflow, tmp_path):
    artifact = tmp_path / "note.txt"
    artifact.write_text("hello", encoding="utf-8")
    config = dataclasses.replace(CONFIG, experiment="search-ranking-test")
    with search_run(
        config, "unit", temp_mlflow["tracking_uri"], temp_mlflow["artifact_location"]
    ) as run:
        log_results({"a": 1.0, "b": float("nan")}, {"seed": 7}, [artifact])
    stored = mlflow.get_run(run.info.run_id)
    assert stored.data.metrics == {"a": 1.0}
    assert stored.data.params == {"seed": "7"}
    names = [item.path for item in mlflow.MlflowClient().list_artifacts(run.info.run_id)]
    assert names == ["note.txt"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/search/test_search_evaluate.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'search.evaluate'`.

- [ ] **Step 3: Write evaluation**

`search/evaluate.py`:

```python
"""Report-split evaluation: every contender, the Phase 4 effects, parser accuracy and retrieval
recall, plus artifacts and MLflow logging.

This module reads ground truth (fraud labels, true slots) on purpose: it is where answers are
compared with predictions.
"""

import json
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import polars as pl

from ingestion.normalize import match_key
from models.price.registry import configure, log_metrics
from search.config import SLOTS, SearchConfig
from search.lexicon import Lexicon
from search.metrics import bootstrap_ci, per_query_metrics, summarize
from search.parse import ParsedQuery, parse
from search.store import read_queries

CONTENDERS = ("xgboost", "lightgbm", "baseline_newest", "baseline_semantic", "baseline_fused")
MISSING_POSITION = 1e9
MAX_PARSE_ERRORS = 200
MONEY_TOLERANCE = 0.01


def baseline_scores(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    semantic = frame["semantic_pos"].fill_nan(None).fill_null(MISSING_POSITION).to_numpy()
    return {
        "baseline_newest": -frame["days_since_posted"].to_numpy().astype(np.float64),
        "baseline_semantic": -semantic.astype(np.float64),
        "baseline_fused": -frame["fused_pos"].to_numpy().astype(np.float64),
    }


def contender_scores(frame: pl.DataFrame, rankers: dict) -> dict[str, np.ndarray]:
    scores = {name: np.asarray(ranker.score(frame)) for name, ranker in rankers.items()}
    return {**scores, **baseline_scores(frame)}


def _scored(frame: pl.DataFrame, values: np.ndarray) -> pl.DataFrame:
    return frame.with_columns(pl.Series("score", np.asarray(values, dtype=np.float64)))


def evaluate_contenders(frame, scores, config: SearchConfig):
    metrics: dict[str, float] = {}
    per_query: dict[str, pl.DataFrame] = {}
    for name, values in scores.items():
        table = per_query_metrics(_scored(frame, values), "score", config.ndcg_k)
        per_query[name] = table
        for key, value in summarize(table).items():
            if key.startswith("kind."):
                kind = key.split(".")[1]
                metrics[f"kind.{kind}.{name}.ndcg_at_10"] = value
            elif key.startswith("no_match."):
                metrics[f"kind.no_match.{name}.{key.split('.', 1)[1]}"] = value
            else:
                metrics[f"{name}.{key}"] = value
        answerable = table.filter(pl.col("answerable"))["ndcg"].to_numpy()
        low, high = bootstrap_ci(answerable, config.n_bootstrap, config.seed)
        metrics[f"{name}.ci_low"], metrics[f"{name}.ci_high"] = low, high
    return metrics, per_query


def top_k_effects(frame, scores, fraud_ids: set[int], k: int, prefix: str = "") -> dict[str, float]:
    table = per_query_metrics(_scored(frame, scores), "score", k)
    sizes = dict(zip(frame["listing_id"].to_list(), frame["cluster_size"].to_list()))
    tops = table["top_ids"].to_list()
    hidden = [sum(int(sizes[listing]) - 1 for listing in top) for top in tops]
    slots = sum(len(top) for top in tops)
    flagged = sum(listing in fraud_ids for top in tops for listing in top)
    return {
        f"{prefix}dup.top10_removed": float(np.mean(hidden)) if hidden else math.nan,
        f"{prefix}fraud.top10_share": flagged / slots if slots else math.nan,
    }


def load_fraud_listing_ids(conn) -> set[int]:
    with conn.cursor() as cur:
        cur.execute("SELECT listing_id FROM listings.listings WHERE fraud_label IS NOT NULL")
        return {row[0] for row in cur.fetchall()}


def load_true_slots(conn, split: str = "report") -> dict[int, dict]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT query_id, true_slots FROM search.queries WHERE split = %s ORDER BY query_id",
            (split,),
        )
        return {query_id: dict(slots) for query_id, slots in cur.fetchall()}


def _close(parsed: float | None, truth: float | None) -> bool:
    if parsed is None or truth is None:
        return parsed is None and truth is None
    return abs(parsed - truth) <= MONEY_TOLERANCE * max(abs(truth), 1.0)


def _key(name: str | None) -> str | None:
    return match_key(name) if name else None


def slot_checks(parsed: ParsedQuery, truth: dict) -> dict[str, tuple[bool, object, object]]:
    area_ok = (
        truth["area_id"] in parsed.area_ids
        if truth["area_id"] is not None
        else parsed.area_name is None
    )
    return {
        "area": (area_ok, truth["area_id"], list(parsed.area_ids)),
        "building": (
            _key(parsed.building) == _key(truth["building"]),
            truth["building"],
            parsed.building,
        ),
        "bedrooms": (parsed.bedrooms == truth["bedrooms"], truth["bedrooms"], parsed.bedrooms),
        "property_type": (
            parsed.property_type == truth["property_type"],
            truth["property_type"],
            parsed.property_type,
        ),
        "budget_min": (
            _close(parsed.budget_min, truth["budget_min"]),
            truth["budget_min"],
            parsed.budget_min,
        ),
        "budget_max": (
            _close(parsed.budget_max, truth["budget_max"]),
            truth["budget_max"],
            parsed.budget_max,
        ),
        "min_size_sqm": (
            _close(parsed.min_size_sqm, truth["min_size_sqm"]),
            truth["min_size_sqm"],
            parsed.min_size_sqm,
        ),
        "amenities": (
            set(parsed.amenities) == set(truth["amenities"]),
            sorted(truth["amenities"]),
            sorted(parsed.amenities),
        ),
    }


def parser_accuracy(queries: pl.DataFrame, slots: dict[int, dict], lexicon: Lexicon):
    right = dict.fromkeys(SLOTS, 0)
    errors = []
    rows = queries.select("query_id", "text").iter_rows()
    for query_id, text in rows:
        for slot, (ok, expected, got) in slot_checks(parse(text, lexicon), slots[query_id]).items():
            right[slot] += ok
            if not ok and len(errors) < MAX_PARSE_ERRORS:
                errors.append((query_id, text, slot, json.dumps(expected), json.dumps(got)))
    total = queries.height
    accuracy = {
        f"parse.{slot}.accuracy": (right[slot] / total if total else math.nan) for slot in SLOTS
    }
    schema = {
        "query_id": pl.Int64,
        "text": pl.Utf8,
        "slot": pl.Utf8,
        "expected": pl.Utf8,
        "parsed": pl.Utf8,
    }
    return accuracy, pl.DataFrame(errors, schema=schema, orient="row")


def retrieval_recall(queries: pl.DataFrame, report_rows: pl.DataFrame) -> float:
    found = report_rows.group_by("query_id").agg((pl.col("grade") == 3).sum().alias("found"))
    answerable = (
        queries.filter(pl.col("n_grade3") > 0)
        .join(found, on="query_id", how="left")
        .with_columns(pl.col("found").fill_null(0))
    )
    if answerable.height == 0:
        return math.nan
    return float((answerable["found"] / answerable["n_grade3"]).mean())


@dataclass(frozen=True)
class Evaluation:
    metrics: dict[str, float]
    per_query: dict[str, pl.DataFrame]
    parse_errors: pl.DataFrame
    report_queries: pl.DataFrame  # query_id, text, kind, parsed (JSON)


def evaluate_report(
    conn,
    table: pl.DataFrame,
    rankers: dict,
    lexicon: Lexicon,
    config: SearchConfig,
    champion: str | None = None,
    ablation=None,
) -> Evaluation:
    report = table.filter(pl.col("split") == "report").sort("query_id", "fused_pos")
    scores = contender_scores(report, rankers)
    metrics, per_query = evaluate_contenders(report, scores, config)
    fraud_ids = load_fraud_listing_ids(conn)
    if champion is not None:
        metrics |= top_k_effects(report, scores[champion], fraud_ids, config.ndcg_k)
    if ablation is not None:
        ablation_scores = ablation.score(report)
        extra, _ = evaluate_contenders(report, {"ablation.no_trust": ablation_scores}, config)
        metrics |= extra
        metrics |= top_k_effects(
            report, ablation_scores, fraud_ids, config.ndcg_k, "ablation.no_trust."
        )
    queries = read_queries(conn, splits=("report",))
    accuracy, errors = parser_accuracy(queries, load_true_slots(conn), lexicon)
    metrics |= accuracy
    metrics["retrieval.recall_at_200"] = retrieval_recall(queries, report)
    metrics["report.queries"] = float(queries.height)
    parsed = [json.dumps(parse(text, lexicon).to_dict()) for text in queries["text"].to_list()]
    report_queries = queries.select("query_id", "text", "kind").with_columns(
        pl.Series("parsed", parsed, dtype=pl.Utf8)
    )
    return Evaluation(metrics, per_query, errors, report_queries)


def _comparison_chart(metrics: dict[str, float], path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [name for name in CONTENDERS if f"{name}.ndcg_at_10" in metrics]
    values = [metrics[f"{name}.ndcg_at_10"] for name in names]
    errors = np.array(
        [
            [value - metrics[f"{name}.ci_low"], metrics[f"{name}.ci_high"] - value]
            for name, value in zip(names, values)
        ]
    ).T
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(names, values, yerr=np.nan_to_num(errors), capsize=4, color="#0e6e7e")
    axis.set_ylabel("NDCG@10 (report split, 95% CI)")
    axis.set_ylim(0, 1)
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def write_artifacts(
    directory: Path, evaluation: Evaluation, importance: pl.DataFrame, timings: dict
) -> list[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report = evaluation.report_queries
    for name, table in evaluation.per_query.items():
        report = report.join(
            table.select("query_id", pl.col("ndcg").alias(f"ndcg.{name}")),
            on="query_id",
            how="left",
        )
    paths = [
        _comparison_chart(evaluation.metrics, directory / "ndcg_comparison.png"),
        directory / "feature_importance.csv",
        directory / "per_query_report.csv",
        directory / "parse_errors.csv",
        directory / "timings.json",
    ]
    importance.sort("gain", descending=True).write_csv(paths[1])
    report.sort("query_id").write_csv(paths[2])
    evaluation.parse_errors.write_csv(paths[3])
    paths[4].write_text(json.dumps(timings, indent=2, allow_nan=False), encoding="utf-8")
    return paths


@contextmanager
def search_run(
    config: SearchConfig,
    run_name: str,
    tracking_uri: str | None = None,
    artifact_location: str | None = None,
) -> Iterator:
    configure(config.experiment, tracking_uri, artifact_location)
    with mlflow.start_run(run_name=run_name) as run:
        yield run


def log_results(metrics: dict[str, float], params: dict, artifacts: list[Path]) -> None:
    mlflow.log_params({key: str(value) for key, value in params.items()})
    log_metrics(metrics)
    for artifact in artifacts:
        mlflow.log_artifact(str(artifact))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -W error tests/search/test_search_evaluate.py`
Expected: all tests PASS.

A check on `test_evaluate_contenders_names_every_metric`:
- In query 1, fused order puts `listing 11` (grade 0) first and `listing 1` (grade 3) second.
- DCG is therefore `0 + 7/log2(3)`, and the ideal is `7`.
- The expected value in the test is written exactly that way. If it fails, fix the code, not the number.

- [ ] **Step 5: Lint, full search tests, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
uv run pytest -q -W error tests/search
git add search/evaluate.py tests/search/test_search_evaluate.py
git commit -m "feat(search): report-split evaluation with baselines, CIs, Phase 4 effects and parser accuracy

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 9: The search engine

**Files:**
- Create: `search/engine.py`, `tests/search/test_search_engine.py`

**Interfaces:**
- **Consumes:**
  - Task 2: `load_lexicon`, `parse`
  - Task 4: `load_clusters`, `retrieve`
  - Task 5: `load_listing_attributes`, `load_predicted_flags`, `reference_date`, `query_frame`, `build_features`
  - Task 1: `read_estimates`
  - Task 7: `load_champion`, `register_ranker`, `XGBRanker`
  - From other packages: `listings.fraud.resolve_price_model_version`, `listings.embed.SentenceTransformerEmbedder`, `listings.embed.resolve_device`
- **Produces (`search.engine`):**
  - `Hit`, a frozen dataclass with fields `listing_id, score, title, area_name, bedrooms, asking_price_aed, size_sqm, reasons: tuple[str, ...], duplicates_hidden: int`
  - `SearchResult`, a frozen dataclass with fields `parsed: ParsedQuery, results: tuple[Hit, ...], ranker: str, notes: tuple[str, ...], timings_ms: dict[str, float]`
  - `FALLBACK = "fallback_fused"`
  - `reasons(parsed: ParsedQuery, row: dict) -> tuple[str, ...]`
    - `row` is one feature row joined to its listing attributes.
  - `query_notes(parsed: ParsedQuery) -> list[str]`
  - `SearchEngine(conn, embedder, config: SearchConfig = SearchConfig(), ranker=<load champion>)`
    - `.search(text: str, k: int = 10) -> SearchResult`
    - `.ranker_label: str`
  - `search(conn, text: str, k: int = 10) -> SearchResult`
    - The spec's module-level entry point.
    - It builds one real-embedder `SearchEngine` per connection object and reuses it.

**Engine rules:**
- **Ranker label:**
  - Loaded from the registry: `"<ranker_name>/v<version>"`.
  - Injected by a caller: `"<kind> (injected)"`.
  - No ranker: `FALLBACK`. In that case, results keep fused order, and the engine adds the note `"no ranking model is registered; showing retrieval order"`.
- **Empty query:** no results, plus the note `"empty query: add an area, a budget, a property type or a few words"`.
- **No candidates:** the note `"no listings match; try widening the budget or removing a filter"`.
- **Parser notes:**
  - Each `("place", text)` → `"area not recognised: <text>"`.
  - Each `("type", text)` → `"property type not listed: <text>"`.
  - `budget_min_exceeds_max` → `"the minimum budget is above the maximum, so the budget was ignored"`.
- **Ordering:** score descending, ties broken by `fused_pos`. Return the top `k`.
- **`duplicates_hidden`:** equals `cluster_size − 1`.
- **Reasons**, in this order, each only when relevant:

| Reason | When it appears | Text |
|---|---|---|
| Area | an area was parsed | `"area ✓"` or `"different area"` |
| Type | a type was parsed | `"type ✓"` or `"different type"` |
| Building | the building matched | `"<building> ✓"` |
| Bedrooms | bedrooms were parsed | `"studio ✓"` or `"N bedroom(s) ✓"` when they match; `"N bedroom(s) (asked M)"` when they don't; `"bedrooms not listed"` when unknown |
| Budget maximum | a maximum was parsed | `"within budget"`, or `"P% over budget"` when `price_over_max > 0`, as a whole percentage |
| Budget minimum | a minimum was parsed and `price_under_min > 0` | `"P% under your minimum"` |
| Size | a minimum size was parsed | `"size ✓"`, or `"P% smaller than asked"` |
| Amenities | once per parsed amenity | `"<amenity> ✓"` or `"no <amenity>"` |
| Value | a finite `price_to_estimate` below 0.95 or above 1.05 | `"priced P% below estimate"` or `"priced P% above estimate"` |
| Flags | each predicted flag | `"flagged: bait price"`, `"flagged: photo reuse"`, `"flagged: inconsistent relist"` |

- [ ] **Step 1: Write the failing tests**

`tests/search/test_search_engine.py`:

```python
import dataclasses

import numpy as np
import polars as pl
import pytest
from search_fixtures import N_CLONES

import search.engine as engine_module
from search.config import FEATURES, SearchConfig
from search.engine import FALLBACK, SearchEngine, query_notes, reasons
from search.parse import ParsedQuery
from search.ranker import XGBRanker
from search.train import register_ranker

CONFIG = dataclasses.replace(SearchConfig(), ef_search=100, device="cpu")
NAN = float("nan")


class LargestFirst:
    kind = "test"

    def score(self, frame):
        # size dominates; the fusion score (at most ~0.033) only breaks exact size ties
        return frame["size_ratio"].fill_nan(0.0).to_numpy() * 1000 + frame["rrf_score"].to_numpy()


def _row(**overrides):
    row = {name: NAN for name in FEATURES}
    row.update(
        area_match=1.0, type_match=0.0, building_match=NAN, beds_diff=1.0, bedrooms=3,
        price_over_max=0.043, price_under_min=NAN, size_ratio=0.9,
        description="Features include balcony.", price_to_estimate=0.8,
        flag_bait_price=1.0, flag_photo_reuse=0.0, flag_inconsistent_relist=0.0,
    )  # fmt: skip
    row.update(overrides)
    return row


def test_reasons_cover_every_stated_slot_in_order():
    parsed = ParsedQuery(
        area_ids=(1,), area_name="Dubai Marina", property_type="flat", bedrooms=2,
        budget_max=1e6, min_size_sqm=100.0, amenities=("balcony", "shared pool"),
    )  # fmt: skip
    assert reasons(parsed, _row()) == (
        "area ✓",
        "different type",
        "3 bedrooms (asked 2)",
        "4% over budget",
        "10% smaller than asked",
        "balcony ✓",
        "no shared pool",
        "priced 20% below estimate",
        "flagged: bait price",
    )


def test_reasons_for_exact_matches_and_quiet_values():
    parsed = ParsedQuery(bedrooms=0, budget_max=1e6, building="Marina Gate", min_size_sqm=50.0)
    row = _row(
        beds_diff=0.0, bedrooms=0, price_over_max=-0.2, size_ratio=1.5, building_match=1.0,
        price_to_estimate=1.02, flag_bait_price=0.0, flag_inconsistent_relist=1.0,
    )  # fmt: skip
    assert reasons(parsed, row) == (
        "Marina Gate ✓",
        "studio ✓",
        "within budget",
        "size ✓",
        "flagged: inconsistent relist",
    )
    assert reasons(ParsedQuery(bedrooms=1), _row(beds_diff=0.0, bedrooms=1)) == ("1 bedroom ✓",)
    assert reasons(ParsedQuery(bedrooms=2), _row(beds_diff=NAN, bedrooms=None))[0] == (
        "bedrooms not listed"
    )
    under = reasons(ParsedQuery(budget_min=2e6), _row(price_under_min=0.25))
    assert under[0] == "25% under your minimum"


def test_query_notes():
    parsed = ParsedQuery(
        unrecognised=(("place", "al barsha"), ("type", "land")),
        errors=("budget_min_exceeds_max",),
    )
    assert query_notes(parsed) == [
        "area not recognised: al barsha",
        "property type not listed: land",
        "the minimum budget is above the maximum, so the budget was ignored",
    ]


def _engine(settings, embedder, ranker, config=CONFIG):
    conn = settings.connect()
    return conn, SearchEngine(conn, embedder, config, ranker=ranker)


def test_fallback_keeps_retrieval_order_and_explains_itself(search_db, fake_embedder):
    settings, listings, _ = search_db
    conn, engine = _engine(settings, fake_embedder, None)
    try:
        result = engine.search("2 bed apartment in Dubai Marina with balcony", k=5)
    finally:
        conn.close()
    assert result.ranker == FALLBACK == engine.ranker_label
    assert "no ranking model is registered; showing retrieval order" in result.notes
    assert 0 < len(result.results) <= 5
    area_one = set(listings.filter(pl.col("area_id") == 1)["listing_id"])
    assert all(hit.listing_id in area_one for hit in result.results)
    assert all(hit.area_name == "Dubai Marina" and hit.title for hit in result.results)
    assert all(hit.reasons[:2] == ("area ✓", "type ✓") for hit in result.results)
    scores = [hit.score for hit in result.results]
    assert scores == sorted(scores, reverse=True)
    assert result.parsed.bedrooms == 2
    assert set(result.timings_ms) == {"parse", "retrieve", "rank", "total"}


def test_an_injected_ranker_orders_the_results(search_db, fake_embedder):
    settings, _, _ = search_db
    conn, engine = _engine(settings, fake_embedder, LargestFirst())
    try:
        result = engine.search("apartment over 60 sqm", k=8)
    finally:
        conn.close()
    assert result.ranker == "test (injected)"
    sizes = [hit.size_sqm for hit in result.results]
    assert sizes == sorted(sizes, reverse=True)


def test_duplicates_are_hidden_and_counted(search_db, fake_embedder):
    settings, listings, _ = search_db
    conn, engine = _engine(settings, fake_embedder, None)
    source = listings.filter(pl.col("listing_id") == 1).row(0, named=True)
    try:
        result = engine.search(source["title"], k=50)
    finally:
        conn.close()
    ids = [hit.listing_id for hit in result.results]
    clone_id = listings.height - N_CLONES + 1
    assert 1 in ids and clone_id not in ids
    assert next(hit for hit in result.results if hit.listing_id == 1).duplicates_hidden == 1


@pytest.mark.parametrize(
    ("text", "note"),
    [
        ("under 5", "empty query: add an area, a budget, a property type or a few words"),
        ("villa in Dubai Marina", "no listings match; try widening the budget or removing a filter"),
        ("flat in Al Barsha", "area not recognised: al barsha"),
    ],
)
def test_notes_for_unhelpful_queries(search_db, fake_embedder, text, note):
    settings, _, _ = search_db
    conn, engine = _engine(settings, fake_embedder, LargestFirst())
    try:
        result = engine.search(text)
    finally:
        conn.close()
    assert note in result.notes
    if note.startswith(("empty", "no listings")):
        assert result.results == ()


def test_the_registered_champion_is_loaded_and_labelled(search_db, fake_embedder, temp_mlflow, tmp_path):
    import mlflow

    rng = np.random.default_rng(0)
    rows = []
    for query_id in range(1, 31):
        for index in range(8):
            features = {name: float(rng.random()) for name in FEATURES}
            rows.append({"query_id": query_id, "grade": int(features["size_ratio"] * 3),
                         "fused_pos": index + 1, **features})  # fmt: skip
    frame = pl.DataFrame(rows)
    config = dataclasses.replace(
        CONFIG, ranker_name="search-engine-test",
        ranker_uri="models:/search-engine-test@champion", max_rounds=20, early_stopping_rounds=5,
    )  # fmt: skip
    ranker = XGBRanker.fit(frame, frame, FEATURES, {"max_depth": 2}, config)
    mlflow.create_experiment("engine-test", artifact_location=temp_mlflow["artifact_location"])
    mlflow.set_experiment("engine-test")
    with mlflow.start_run():
        register_ranker(ranker, config, tmp_path / "ranker")
    settings, _, _ = search_db
    conn = settings.connect()
    try:
        engine = SearchEngine(conn, fake_embedder, config)
        result = engine.search("2 bed apartment in Dubai Marina")
    finally:
        conn.close()
    assert engine.ranker_label == "search-engine-test/v1" == result.ranker
    assert result.results and FALLBACK not in result.ranker


def test_module_search_reuses_one_engine_per_connection(search_db, monkeypatch):
    from listings.embed import FakeEmbedder

    settings, _, _ = search_db
    built = []

    class CountingEngine(SearchEngine):
        def __init__(self, conn, embedder, config=SearchConfig(), ranker=None):
            built.append(type(embedder).__name__)
            super().__init__(conn, embedder, CONFIG, ranker=None)

    monkeypatch.setattr(engine_module, "SearchEngine", CountingEngine)
    monkeypatch.setattr(engine_module, "SentenceTransformerEmbedder", lambda device: FakeEmbedder())
    monkeypatch.setattr(engine_module, "_ENGINES", {})
    conn = settings.connect()
    try:
        first = engine_module.search(conn, "apartment in JVC")
        second = engine_module.search(conn, "villa in Arabian Ranches", k=3)
    finally:
        conn.close()
    assert built == ["FakeEmbedder"]
    assert first.results and len(second.results) <= 3
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/search/test_search_engine.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'search.engine'`.

- [ ] **Step 3: Write the engine**

`search/engine.py`:

```python
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
    if parsed.budget_min is not None and _finite(row["price_under_min"]):
        if row["price_under_min"] > 0:
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
    def __init__(self, conn, embedder, config: SearchConfig = SearchConfig(), ranker=_LOAD_CHAMPION):
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
                    "listing_id", "title", "area_name", "bedrooms",
                    "asking_price_aed", "size_sqm", "description",
                ),  # fmt: skip
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
    """One engine per connection object, with the real MiniLM embedder on the best device."""
    engine = _ENGINES.get(id(conn))
    if engine is None:
        embedder = SentenceTransformerEmbedder(device=resolve_device("auto"))
        engine = _ENGINES[id(conn)] = SearchEngine(conn, embedder)
    return engine.search(text, k)
```

Two notes on this code:
- `resolve_price_model_version` works for any `models:/name@alias` URI, including the ranker's, despite its name. Importing it avoids duplicating the registry lookup.
- `test_the_registered_champion_is_loaded_and_labelled` fits on the same frame twice (train = tune). That is fine for a loading test; the test does not measure quality.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -W error tests/search/test_search_engine.py`
Expected: all tests PASS.

Also run the leakage scan: `uv run pytest -q -W error tests/search/test_search_features.py -k label`. `engine.py` must pass it.

- [ ] **Step 5: Lint, full search tests, commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
uv run pytest -q -W error tests/search
git add search/engine.py tests/search/test_search_engine.py
git commit -m "feat(search): SearchEngine with reasons, notes and a retrieval-order fallback

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 10: CLI, end-to-end test, the real run, README and spec amendments

**Files:**
- Create:
  - `search/__main__.py`
  - `tests/search/test_search_integration.py`
- Modify:
  - `README.md`: add a new "## Property search" section and update the status line, module layout, Mermaid diagram and cost table
  - `docs/superpowers/specs/2026-09-16-phase5-search-ranking-design.md`: add a dated amendments section

**Interfaces:**
- **Consumes:** every earlier task.
- **Produces:** `search.__main__.main(argv: list[str] | None = None) -> int`, with these subcommands:
  - `queries [--n N] [--seed S] [--device auto|cuda|cpu] [--fake] [--data-dir DIR]`
  - `train [--trials T] [--device auto|cuda|cpu] [--data-dir DIR]`
  - `evaluate [--data-dir DIR]`
  - `query TEXT [--k K] [--fake] [--device ...]`

**How each stage behaves:**
- **`queries`**
  1. Applies the search schema.
  2. Refreshes the price estimates, and prints a stderr WARNING if the value features were skipped.
  3. Generates, retrieves and grades the queries, then replaces the query set.
  4. Writes `queries_stats.json` (the estimate stats plus per-split and per-kind counts) to the data dir.
  5. Prints the counts.
- **`train`**
  1. Refuses to run (a `RuntimeError`, which makes the exit code 1) unless `queries_corpus_run(conn) == latest_corpus_run(conn)`.
  2. Builds the feature table and trains both rankers, plus the no-trust ablation.
  3. Evaluates on the report split.
  4. Applies the registration gate.
  5. Inside one MLflow run named `search-train`, logs everything and registers the winner **only if** the gate passes.
  6. Prints the contender table and the gate verdict.
  7. The gate failing is a legitimate result, so `train` still exits 0 in that case.
- **`evaluate`**
  1. Loads the champion; if none is registered, fails with `no champion registered — run python -m search train first`.
  2. Re-scores it and the three baselines on the report split.
  3. Logs an MLflow run named `search-evaluate`.
- **`query`** prints the parsed slots, the notes, the ranker label and the hits with their reasons.

**Shared behaviour:**
- **Settings:** `load_dotenv()` and `logging.basicConfig` run before dispatch.
- **Timings:** each successful stage records its wall-clock seconds in `<data-dir>/stage_timings.json`. `queries` starts a fresh table, and `train` logs the table as `timings.json`.
- **Errors:** any exception prints `<command> failed: <Type>: <message>` to stderr and exits 1.
- **`--fake`:** uses `FakeEmbedder`, for tests only.
- **MLflow params logged by `train`:**
  - `seed`, `n_queries`, `trials`, `device`
  - `winner` and `winner_params` (JSON)
  - `price_model_uri` and `price_model_version` (or `"unavailable"`)
  - `corpus_run_id`, `detect_run_id`
  - `registered_version` (or `"none"`)
  - `value_features_skipped`

  It also logs `tune.<kind>.ndcg_at_10` for both rankers, plus `gate.passed`.

- [ ] **Step 1: Write the failing end-to-end test**

`tests/search/test_search_integration.py`:

```python
import json

import mlflow
import pytest

from ingestion.config import DbSettings
from search.__main__ import main


@pytest.fixture
def cli_env(search_db, temp_mlflow, monkeypatch, tmp_path):
    """Point DbSettings.from_env() at the seeded test database, and MLflow at a temp store."""
    settings, _, _ = search_db
    for name, value in {
        "POSTGRES_HOST": settings.host,
        "POSTGRES_PORT": str(settings.port),
        "POSTGRES_USER": settings.user,
        "POSTGRES_PASSWORD": settings.password,
        "POSTGRES_DB": settings.dbname,
    }.items():
        monkeypatch.setenv(name, value)
    assert DbSettings.from_env().dbname == "zestimator_test"
    # create the experiment up front so the CLI never falls back to ./mlruns for artifacts
    mlflow.create_experiment("search-ranking", artifact_location=temp_mlflow["artifact_location"])
    return tmp_path / "data"


def test_the_cli_drives_the_whole_pipeline(cli_env, capsys):
    data = str(cli_env)
    assert main(["queries", "--n", "150", "--fake", "--data-dir", data]) == 0
    out = capsys.readouterr()
    assert "150 queries" in out.out
    assert "value features were SKIPPED" in out.err  # no price model in the temp registry
    stats = json.loads((cli_env / "queries_stats.json").read_text(encoding="utf-8"))
    assert stats["queries"] == 150 and stats["value_features_skipped"] == 1.0

    assert main(["train", "--trials", "1", "--device", "cpu", "--data-dir", data]) == 0
    out = capsys.readouterr().out
    assert "baseline_fused" in out and "gate:" in out
    runs = mlflow.search_runs(experiment_names=["search-ranking"], output_format="list")
    assert len(runs) == 1
    train_run = runs[0]
    assert train_run.info.run_name == "search-train"
    assert {"xgboost.ndcg_at_10", "baseline_fused.ndcg_at_10", "gate.passed"} <= set(
        train_run.data.metrics
    )
    assert train_run.data.params["winner"] in {"xgboost", "lightgbm"}
    artifacts = {a.path for a in mlflow.MlflowClient().list_artifacts(train_run.info.run_id)}
    assert {"ndcg_comparison.png", "per_query_report.csv", "timings.json"} <= artifacts
    passed = train_run.data.metrics["gate.passed"] == 1.0
    assert (train_run.data.params["registered_version"] != "none") is passed

    assert main(["query", "2 bed apartment in Dubai Marina", "--fake"]) == 0
    out = capsys.readouterr().out
    assert "Dubai Marina" in out and "area ✓" in out

    status = main(["evaluate", "--data-dir", data])
    captured = capsys.readouterr()
    if passed:
        assert status == 0
        runs = mlflow.search_runs(experiment_names=["search-ranking"], output_format="list")
        assert {run.info.run_name for run in runs} == {"search-train", "search-evaluate"}
    else:
        assert status == 1 and "no champion registered" in captured.err

    timings = json.loads((cli_env / "stage_timings.json").read_text(encoding="utf-8"))
    assert {"queries", "train"} <= set(timings)


def test_train_refuses_a_stale_query_set(cli_env, capsys):
    data = str(cli_env)
    assert main(["queries", "--n", "60", "--fake", "--data-dir", data]) == 0
    settings = DbSettings.from_env()
    conn = settings.connect()
    try:
        with conn.cursor() as cur:  # a newer corpus run makes the stored queries stale
            cur.execute(
                "INSERT INTO listings.corpus_runs (seed, photo_dataset_sha, counts) "
                "VALUES (1, 'x', '{}'::jsonb)"
            )
        conn.commit()
    finally:
        conn.close()
    capsys.readouterr()
    assert main(["train", "--trials", "1", "--device", "cpu", "--data-dir", data]) == 1
    assert "python -m search queries" in capsys.readouterr().err


def test_a_bad_command_line_is_an_argparse_error():
    with pytest.raises(SystemExit):
        main(["train", "--trials", "many"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest -q tests/search/test_search_integration.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'search.__main__'`.

- [ ] **Step 3: Write the CLI**

`search/__main__.py`:

```python
"""Command line: python -m search queries|train|evaluate|query."""

import argparse
import dataclasses
import json
import logging
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from ingestion.config import DbSettings
from listings.embed import FakeEmbedder, SentenceTransformerEmbedder, resolve_device
from listings.fraud import resolve_price_model_version
from search.config import DATA_DIR, SearchConfig
from search.engine import SearchEngine
from search.evaluate import (
    CONTENDERS,
    evaluate_report,
    log_results,
    search_run,
    write_artifacts,
)
from search.features import feature_table, refresh_estimates
from search.lexicon import load_lexicon
from search.queries import build_query_set
from search.ranker import load_champion
from search.store import (
    apply_schema,
    latest_corpus_run,
    queries_corpus_run,
    replace_query_set,
)
from search.train import fit_ablation, gate_passes, register_ranker, train_rankers

TIMINGS_FILE = "stage_timings.json"
STATS_FILE = "queries_stats.json"
DEVICES = ("auto", "cuda", "cpu")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m search", description="Synthetic queries, ranker training and search."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    queries = commands.add_parser("queries", help="generate, retrieve and grade the query set")
    queries.add_argument("--n", type=int, default=SearchConfig.n_queries)
    queries.add_argument("--seed", type=int, default=SearchConfig.seed)
    queries.add_argument("--device", choices=DEVICES, default="auto")
    queries.add_argument("--fake", action="store_true", help="use the deterministic test embedder")
    queries.add_argument("--data-dir", type=Path, default=DATA_DIR)

    train = commands.add_parser("train", help="fit, evaluate, gate and register the ranker")
    train.add_argument("--trials", type=int, default=SearchConfig.n_trials)
    train.add_argument("--device", choices=DEVICES, default="auto")
    train.add_argument("--data-dir", type=Path, default=DATA_DIR)

    evaluate = commands.add_parser("evaluate", help="re-score the registered champion")
    evaluate.add_argument("--data-dir", type=Path, default=DATA_DIR)

    query = commands.add_parser("query", help="search the corpus and explain the results")
    query.add_argument("text")
    query.add_argument("--k", type=int, default=10)
    query.add_argument("--fake", action="store_true")
    query.add_argument("--device", choices=DEVICES, default="auto")
    query.add_argument("--data-dir", type=Path, default=DATA_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parser().parse_args(argv)
    load_dotenv()  # before any settings are read
    commands = {"queries": _queries, "train": _train, "evaluate": _evaluate, "query": _query}
    args.started = time.perf_counter()
    try:
        status = commands[args.command](args)
    except Exception as exc:  # noqa: BLE001 — CLI boundary: report and exit 1
        print(f"{args.command} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if status == 0 and args.command != "query":
        seconds = time.perf_counter() - args.started
        _record_timing(args.data_dir, args.command, seconds)
        print(f"{args.command} took {seconds:.1f}s")
    return status


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _record_timing(data_dir: Path, stage: str, seconds: float) -> None:
    path = Path(data_dir) / TIMINGS_FILE
    timings = {} if stage == "queries" else _read_json(path)
    timings[stage] = {
        "seconds": round(seconds, 3),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(timings, indent=2, allow_nan=False), encoding="utf-8")


def _embedder(args):
    if args.fake:
        return FakeEmbedder(), "cpu"
    device = resolve_device(args.device)
    return SentenceTransformerEmbedder(device=device), device


def _latest_detect_run(conn) -> int | None:
    with conn.cursor() as cur:
        cur.execute("SELECT max(detect_run_id) FROM listings.detect_runs")
        return cur.fetchone()[0]


def _queries(args) -> int:
    config = dataclasses.replace(SearchConfig(), n_queries=args.n, seed=args.seed)
    embedder, device = _embedder(args)
    conn = DbSettings.from_env().connect()
    try:
        apply_schema(conn)
        conn.commit()
        stats = refresh_estimates(conn, config)
        conn.commit()
        print(f"building {config.n_queries} queries (embedding on {device})")
        queries, judgments = build_query_set(conn, embedder, load_lexicon(conn), config)
        replace_query_set(conn, queries, judgments)
        conn.commit()
    finally:
        conn.close()
    if stats["value_features_skipped"]:
        print(
            "WARNING: value features were SKIPPED — the price model "
            f"({config.price_model_uri}) could not be loaded; price_to_estimate and "
            "within_interval will be missing for every listing.",
            file=sys.stderr,
        )
    counts = queries.group_by("split", "kind").len().sort("split", "kind")
    summary = {
        **stats,
        "queries": queries.height,
        "judgments": judgments.height,
        "counts": {f"{split}.{kind}": n for split, kind, n in counts.iter_rows()},
    }
    path = Path(args.data_dir) / STATS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"{queries.height} queries, {judgments.height:,} graded candidates")
    for key, value in summary["counts"].items():
        print(f"  {key:<24}{value:>6}")
    return 0


def _require_fresh_queries(conn) -> int:
    corpus_run_id = latest_corpus_run(conn)
    if queries_corpus_run(conn) != corpus_run_id:
        raise RuntimeError(
            "the stored query set was built on an older corpus — run `python -m search queries`"
        )
    return corpus_run_id


def _print_contenders(metrics: dict[str, float], names) -> None:
    print(f"{'contender':<22}{'NDCG@10':>9}{'95% CI':>18}{'MRR':>8}{'P@5':>8}")
    for name in names:
        if f"{name}.ndcg_at_10" not in metrics:
            continue
        ci = f"{metrics[f'{name}.ci_low']:.3f}-{metrics[f'{name}.ci_high']:.3f}"
        print(
            f"{name:<22}{metrics[f'{name}.ndcg_at_10']:>9.3f}{ci:>18}"
            f"{metrics[f'{name}.mrr']:>8.3f}{metrics[f'{name}.precision_at_5']:>8.3f}"
        )


def _train(args) -> int:
    config = dataclasses.replace(
        SearchConfig(), device=resolve_device(args.device), n_trials=args.trials
    )
    conn = DbSettings.from_env().connect()
    try:
        corpus_run_id = _require_fresh_queries(conn)
        detect_run_id = _latest_detect_run(conn)
        lexicon = load_lexicon(conn)
        table = feature_table(conn, lexicon)
        result = train_rankers(table, config, config.n_trials)
        ablation = fit_ablation(table, result, config)
        evaluation = evaluate_report(
            conn, table, result.rankers, lexicon, config, champion=result.winner, ablation=ablation
        )
    finally:
        conn.close()
    stats = _read_json(Path(args.data_dir) / STATS_FILE)
    metrics = dict(evaluation.metrics)
    for kind, value in result.tune_ndcg.items():
        metrics[f"tune.{kind}.ndcg_at_10"] = value
    winner = result.winner
    passed = gate_passes(
        metrics[f"{winner}.ndcg_at_10"],
        metrics[f"{winner}.ci_low"],
        metrics["baseline_fused.ndcg_at_10"],
    )
    metrics["gate.passed"] = float(passed)
    timings = _read_json(Path(args.data_dir) / TIMINGS_FILE)
    timings["train_before_logging"] = {"seconds": round(time.perf_counter() - args.started, 3)}
    for stage, entry in timings.items():
        metrics[f"timing.{stage}_seconds"] = float(entry["seconds"])

    with tempfile.TemporaryDirectory() as tmp, search_run(config, "search-train"):
        version = register_ranker(result.rankers[winner], config, Path(tmp) / "ranker") if passed else None
        artifacts = write_artifacts(
            Path(tmp) / "artifacts", evaluation, result.rankers[winner].importance(), timings
        )
        params = {
            "seed": config.seed,
            "n_queries": stats.get("queries", "unknown"),
            "trials": config.n_trials,
            "device": config.device,
            "winner": winner,
            "winner_params": json.dumps(result.best_params[winner], sort_keys=True),
            "price_model_uri": config.price_model_uri,
            "price_model_version": resolve_price_model_version(config.price_model_uri)
            or "unavailable",
            "value_features_skipped": stats.get("value_features_skipped", "unknown"),
            "corpus_run_id": corpus_run_id,
            "detect_run_id": detect_run_id,
            "registered_version": version or "none",
        }
        log_results(metrics, params, artifacts)

    _print_contenders(metrics, [*CONTENDERS, "ablation.no_trust"])
    print("tune NDCG@10: " + ", ".join(f"{k} {v:.3f}" for k, v in result.tune_ndcg.items()))
    verdict = f"registered {config.ranker_name} v{version}" if passed else "not registered"
    print(f"gate: winner {winner} {'PASSED' if passed else 'FAILED'} — {verdict}")
    return 0


def _evaluate(args) -> int:
    config = SearchConfig()
    champion = load_champion(config.ranker_uri)
    if champion is None:
        raise RuntimeError("no champion registered — run python -m search train first")
    conn = DbSettings.from_env().connect()
    try:
        corpus_run_id = _require_fresh_queries(conn)
        lexicon = load_lexicon(conn)
        table = feature_table(conn, lexicon)
        evaluation = evaluate_report(
            conn, table, {champion.kind: champion}, lexicon, config, champion=champion.kind
        )
    finally:
        conn.close()
    with tempfile.TemporaryDirectory() as tmp, search_run(config, "search-evaluate"):
        artifacts = write_artifacts(
            Path(tmp), evaluation, champion.importance(), _read_json(Path(args.data_dir) / TIMINGS_FILE)
        )
        log_results(
            evaluation.metrics,
            {
                "ranker_uri": config.ranker_uri,
                "ranker_version": resolve_price_model_version(config.ranker_uri) or "unknown",
                "ranker_kind": champion.kind,
                "corpus_run_id": corpus_run_id,
            },
            artifacts,
        )
    _print_contenders(evaluation.metrics, [champion.kind, *CONTENDERS[2:]])
    return 0


def _query(args) -> int:
    embedder, _ = _embedder(args)
    conn = DbSettings.from_env().connect()
    try:
        result = SearchEngine(conn, embedder).search(args.text, args.k)
    finally:
        conn.close()
    understood = {k: v for k, v in result.parsed.to_dict().items() if v not in (None, [], "")}
    print(f"understood: {json.dumps(understood)}")
    print(f"ranker: {result.ranker}")
    for note in result.notes:
        print(f"note: {note}")
    for position, hit in enumerate(result.results, start=1):
        beds = "?" if hit.bedrooms is None else hit.bedrooms
        print(
            f"{position:>2}. #{hit.listing_id} {hit.title} | {hit.area_name} | {beds} bed | "
            f"{hit.size_sqm:.0f} sqm | AED {hit.asking_price_aed:,.0f}"
        )
        hidden = f" (+{hit.duplicates_hidden} duplicate)" if hit.duplicates_hidden else ""
        print(f"    {', '.join(hit.reasons)}{hidden}")
    print(f"took {result.timings_ms['total']:.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Notes on this module:
- The "Phase 4 detect run" lookup uses plain SQL here; `__main__.py` must still pass the leakage scan.
- The line `version = register_ranker(...) if passed else None` may exceed 100 characters. Let `ruff format` wrap it.

- [ ] **Step 4: Run the end-to-end tests to verify they pass**

Run: `uv run pytest -q -W error tests/search/test_search_integration.py`
Expected: all tests PASS. The pipeline test takes about a minute.

- [ ] **Step 5: Full suite, lint, commit the code**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check .
uv run pytest -q -W error
git add search/__main__.py tests/search/test_search_integration.py
git commit -m "feat(search): python -m search queries|train|evaluate|query with an end-to-end test

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

Run the whole suite (not just `tests/search`) exactly once, in the foreground. It must be green apart from the one known skip.

- [ ] **Step 6: The real run**

Run from the repo root. Tee every stage's output to `.superpowers/sdd/2026-09-16-phase5-search-ranking/task-10-logs/<stage>.log`, and record each stage's wall-clock seconds.
- **Short stages:** use a Bash timeout of 600000 ms.
- **Long stages (`queries` and `train`):** run them in the background with their output going to the log file. Then check the log with short foreground `tail` commands until the process exits. Never sit idle waiting on a background job.

```bash
mkdir -p .superpowers/sdd/2026-09-16-phase5-search-ranking/task-10-logs
uv run python -m search queries 2>&1 | tee .superpowers/sdd/2026-09-16-phase5-search-ranking/task-10-logs/queries.log
uv run python -m search train 2>&1 | tee .superpowers/sdd/2026-09-16-phase5-search-ranking/task-10-logs/train.log
uv run python -m search evaluate 2>&1 | tee .superpowers/sdd/2026-09-16-phase5-search-ranking/task-10-logs/evaluate.log
uv run python -m search query "2BR in Dubai Marina under 1.5M" 2>&1 | tee .superpowers/sdd/2026-09-16-phase5-search-ranking/task-10-logs/query-marina.log
uv run python -m search query "family villa with pool in Arabian Ranches" 2>&1 | tee .superpowers/sdd/2026-09-16-phase5-search-ranking/task-10-logs/query-villa.log
uv run python -m search query "studio in JVC max 600k" 2>&1 | tee .superpowers/sdd/2026-09-16-phase5-search-ranking/task-10-logs/query-jvc.log
```

**Gates.** Stop and report BLOCKED if any of these fail:
- `queries.log` says it embedded on `cuda`, and it contains **no** "value features were SKIPPED" warning. The Phase 3 champion (`models:/zestimator-price@champion`, v2) is registered on the server.
- `train.log` ends with a gate line.
- The MLflow run named `search-train` exists in experiment `search-ranking` at http://127.0.0.1:5000, with its artifacts.

**If the gate fails,** `train` registers nothing and `evaluate` exits 1 with "no champion registered". That is a legitimate result, not a failure of the task:
- Record the result in the README.
- Still run the `query` commands, which will show `fallback_fused`.

**If `train` runs longer than 90 minutes,** stop it and report DONE_WITH_CONCERNS with the log tail and the trial count reached. Do not lower `--trials` on your own.

- [ ] **Step 7: README section**

Add `## Property search` after the duplicate-and-fraud section, and fill every number from the Step 6 logs or the `search-train` MLflow run. It must contain:

1. **What it does and how**, in one paragraph: parse, then two-channel retrieval with RRF, duplicate collapse, the 25 features and the GBDT ranker. Include the command block with real stage timings.
2. **Data provenance.** Listings are synthetic over real DLD sales (Phase 4). The queries and their 0–3 grades are **synthetic and rule-based**, so the ranker partly learns the grading rules. The numbers measure the pipeline against those rules, not real user satisfaction.
3. **Contender table** (report split): NDCG@10 with its 95% CI, MRR and P@5, for all five contenders plus the no-trust ablation. Name the winner, its tune NDCG, and the gate verdict. If nothing was registered, say so plainly.
4. **By query kind:** NDCG@10 for `specified` and `vague` queries, and the mean top-10 grade for `no_match` queries, each for the winner and for `baseline_fused`.
5. **Retrieval and parsing:** `retrieval.recall_at_200` and all eight `parse.<slot>.accuracy` values, each marked as measured on report-split queries.
6. **Phase 4 effects:** `dup.top10_removed`, and `fraud.top10_share` with and without the trust features. Explain what each measures.
7. **Caveats:**
   - The value features use the Phase 3 champion, which was fit on every source sale, so they are in-sample.
   - The template split holds out whole phrasing frames, but per-query randomness (aliases, number formats) is shared.
   - There is no geographic proximity.
   - The fake-embedder tests are no evidence of real semantic quality.
8. **Examples:** two or three example queries from the `query-*.log` files, with their top 3 results and reasons.
9. **Honest framing.** If a result is weaker than the framing implies, change the framing, not the number.

Also update:
- the README status line (Phase 5 done);
- the module layout, which gains `search/`;
- the Mermaid diagram: add the search flow from the listings tables, through the `search` schema, to the ranker registry;
- the cost table: $0 for GPU time.

- [ ] **Step 8: Spec amendments**

Append `## Amendments during implementation (2026-09-16)` to the spec, recording:
- Planning rulings 1–4, 6, 7 and 8 from this plan's header.
- The reordered parser passes: places before bedrooms and type, and why.
- The `store.py` exemption to the label-name scan.
- The headline numbers and the date of the real run.
- Anything else the implementation did differently from the spec, and why.

- [ ] **Step 9: Commit the docs**

```bash
uv run pytest -q -W error tests/search
git add README.md docs/superpowers/specs/2026-09-16-phase5-search-ranking-design.md
git commit -m "docs: Phase 5 property search results, write-up and spec amendments

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

Never commit `.superpowers/`, `data/` or `mlruns/`.
