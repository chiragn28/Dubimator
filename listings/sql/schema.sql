CREATE SCHEMA IF NOT EXISTS listings;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS listings.corpus_runs (
    corpus_run_id serial PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now(),
    seed integer NOT NULL,
    photo_dataset_sha text,
    counts jsonb NOT NULL
);

CREATE TABLE IF NOT EXISTS listings.listings (
    listing_id bigint PRIMARY KEY,
    source_transaction_id text NOT NULL,
    agent_id integer NOT NULL,
    posted_at date NOT NULL,
    title text NOT NULL,
    description text NOT NULL,
    asking_price_aed double precision NOT NULL,
    area_id integer NOT NULL,
    area_name text,
    building_name text,
    project_name text,
    property_type text NOT NULL,
    property_sub_type text,
    reg_type text NOT NULL,
    size_sqm double precision NOT NULL,
    bedrooms smallint,
    photo_set_id integer NOT NULL,
    dup_group_id bigint,
    control_group_id text,
    fraud_label text,
    is_synthetic boolean NOT NULL DEFAULT true,
    corpus_run_id integer NOT NULL REFERENCES listings.corpus_runs (corpus_run_id)
);

CREATE TABLE IF NOT EXISTS listings.photos (
    photo_id bigint PRIMARY KEY,
    set_id integer NOT NULL,
    room text NOT NULL,
    path text NOT NULL,
    variant_of bigint,
    variant_kind text,
    is_stock boolean NOT NULL,
    embedding vector(512),
    corpus_run_id integer NOT NULL REFERENCES listings.corpus_runs (corpus_run_id)
);

CREATE TABLE IF NOT EXISTS listings.listing_photos (
    listing_id bigint NOT NULL REFERENCES listings.listings (listing_id),
    photo_id bigint NOT NULL REFERENCES listings.photos (photo_id),
    position smallint NOT NULL,
    PRIMARY KEY (listing_id, position)
);

CREATE TABLE IF NOT EXISTS listings.listing_embeddings (
    listing_id bigint PRIMARY KEY REFERENCES listings.listings (listing_id),
    text_embedding vector(384),
    image_embedding vector(512)
);

CREATE TABLE IF NOT EXISTS listings.detect_runs (
    detect_run_id serial PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now(),
    corpus_run_id integer NOT NULL REFERENCES listings.corpus_runs (corpus_run_id),
    threshold double precision,
    metrics jsonb
);

CREATE TABLE IF NOT EXISTS listings.duplicate_pairs (
    detect_run_id integer NOT NULL REFERENCES listings.detect_runs (detect_run_id),
    listing_a bigint NOT NULL REFERENCES listings.listings (listing_id),
    listing_b bigint NOT NULL REFERENCES listings.listings (listing_id),
    score double precision NOT NULL,
    decision boolean NOT NULL,
    signals jsonb NOT NULL,
    PRIMARY KEY (detect_run_id, listing_a, listing_b),
    CONSTRAINT duplicate_pairs_canonical CHECK (listing_a < listing_b)
);

CREATE TABLE IF NOT EXISTS listings.fraud_flags (
    detect_run_id integer NOT NULL REFERENCES listings.detect_runs (detect_run_id),
    listing_id bigint NOT NULL REFERENCES listings.listings (listing_id),
    flag text NOT NULL,
    detail jsonb NOT NULL,
    PRIMARY KEY (detect_run_id, listing_id, flag)
);

CREATE INDEX IF NOT EXISTS listings_posted_at_idx ON listings.listings (posted_at);
CREATE INDEX IF NOT EXISTS listings_area_building_idx
    ON listings.listings (area_id, building_name);
CREATE INDEX IF NOT EXISTS listing_photos_photo_idx ON listings.listing_photos (photo_id);

COMMENT ON TABLE listings.listings IS
    'Synthetic listings generated over real DLD sales (Phase 4). Text, agents, posting '
    'dates, duplicates and fraud cases are invented; area, building, size and price level '
    'come from dld.market_sales.';
COMMENT ON COLUMN listings.listings.dup_group_id IS
    'Ground truth: the listing this one was cloned from. Never a detection input.';
COMMENT ON COLUMN listings.listings.control_group_id IS
    'Ground truth: same-building group that must NOT be flagged. Never a detection input.';
COMMENT ON COLUMN listings.listings.fraud_label IS
    'Ground truth fraud case. Never a detection input.';
