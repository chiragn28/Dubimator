CREATE SCHEMA IF NOT EXISTS dld;

CREATE TABLE IF NOT EXISTS dld.ingestion_runs (
    run_id serial PRIMARY KEY,
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    source_path text NOT NULL,
    source_sha256 text NOT NULL,
    rows_read integer NOT NULL,
    rows_loaded integer,
    rows_market_sale integer,
    status text NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
    error text,
    details jsonb
);

CREATE TABLE IF NOT EXISTS dld.transactions (
    transaction_id text,
    source_row integer NOT NULL,
    trans_group text NOT NULL,
    procedure_name text,
    instance_date date,
    year smallint,
    property_type text NOT NULL,
    property_sub_type text,
    property_usage text,
    reg_type text NOT NULL,
    area_id integer,
    area_name text,
    area_name_ar text,
    building_name text,
    project_number integer,
    project_name text,
    master_project text,
    nearest_landmark text,
    nearest_metro text,
    nearest_mall text,
    rooms text,
    bedrooms smallint,
    has_parking boolean,
    area_sqm double precision,
    price_aed double precision,
    price_per_sqm_aed double precision,
    parties_role_1 smallint,
    parties_role_2 smallint,
    parties_role_3 smallint,
    exclusion_reason text,
    peer_tier smallint,
    price_robust_z double precision,
    ingest_run_id integer NOT NULL REFERENCES dld.ingestion_runs (run_id),
    PRIMARY KEY (ingest_run_id, source_row)
);

CREATE INDEX IF NOT EXISTS transactions_exclusion_reason_idx
    ON dld.transactions (exclusion_reason);
CREATE INDEX IF NOT EXISTS transactions_area_type_date_idx
    ON dld.transactions (area_id, property_type, instance_date);

CREATE TABLE IF NOT EXISTS dld.areas (
    area_id integer PRIMARY KEY,
    name_en text NOT NULL,
    name_ar text,
    match_key text,
    market_sales integer NOT NULL
);

CREATE TABLE IF NOT EXISTS dld.area_aliases (
    alias_key text NOT NULL,
    alias text NOT NULL,
    area_id integer NOT NULL REFERENCES dld.areas (area_id),
    source text NOT NULL CHECK (source IN ('official', 'master_project', 'curated')),
    PRIMARY KEY (alias_key, area_id)
);

CREATE OR REPLACE VIEW dld.market_sales AS
    SELECT * FROM dld.transactions WHERE exclusion_reason IS NULL;
