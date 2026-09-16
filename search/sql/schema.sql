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
