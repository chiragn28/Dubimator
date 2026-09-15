import dataclasses
from datetime import date
from pathlib import Path

from ingestion.pipeline import run_pipeline
from models.price.config import FORBIDDEN_FEATURES, HOME_UNIT_SUB_TYPES, TrainConfig
from models.price.data import HOMES_SQL, RAW_SCHEMA, load_homes, prepare_homes

SMALL = dataclasses.replace(TrainConfig(), min_segment_rows=1)


def test_sql_never_selects_target_derived_columns():
    for name in ("price_per_sqm_aed", "price_robust_z", "peer_tier", "source_row"):
        assert name not in HOMES_SQL


def test_only_lineage_and_target_columns_overlap_the_forbidden_list():
    assert set(RAW_SCHEMA) & set(FORBIDDEN_FEATURES) == {"ingest_run_id", "price_aed"}


def test_size_bounds_drop_and_count_per_segment(raw_homes):
    raw = raw_homes(
        {"area_sqm": 11.9},
        {"area_sqm": 12.0},
        {"area_sqm": 2_000.0},
        {"area_sqm": 2_000.5},
        {"property_type": "villa", "property_sub_type": None, "area_sqm": 59.0},
        {"property_type": "villa", "property_sub_type": None, "area_sqm": 60.0},
        {"property_type": "villa", "property_sub_type": "Villa", "area_sqm": 40.0},
    )
    rows, drops, _ = prepare_homes(raw, SMALL)
    assert rows["area_sqm"].to_list() == [12.0, 2_000.0, 60.0, 40.0]
    assert drops == {"size_bounds.unit_ready_built_up": 2, "size_bounds.villa_ready_plot": 1}


def test_unsupported_segments_are_dropped_and_counted(raw_homes):
    config = dataclasses.replace(TrainConfig(), min_segment_rows=2)
    plot_villa = {
        "property_type": "villa", "property_sub_type": None, "reg_type": "off_plan",
        "area_sqm": 400.0,
    }  # fmt: skip
    rows, drops, supported = prepare_homes(raw_homes({}, {}, plot_villa), config)
    assert supported == ("unit_ready_built_up",)
    assert drops == {"unsupported_segment.villa_off_plan_plot": 1}
    assert rows.height == 2


def test_support_counts_only_clean_train_period_rows(raw_homes):
    config = dataclasses.replace(TrainConfig(), min_segment_rows=2)
    raw = raw_homes(
        {"instance_date": date(2016, 5, 1)},
        {"instance_date": date(2022, 8, 1)},  # val period: doesn't count
        {"instance_date": date(2016, 6, 1), "is_clean": False},  # not clean: doesn't count
    )
    _, drops, supported = prepare_homes(raw, config)
    assert supported == ()
    assert drops == {"unsupported_segment.unit_ready_built_up": 3}


def test_load_homes_reads_scope_lineage_and_reference_tables(pg_test_db):
    summary = run_pipeline(Path("tests/fixtures/dld_sample.csv"), pg_test_db)
    config = dataclasses.replace(SMALL, index_start=date(1990, 1, 1))
    homes = load_homes(pg_test_db, config)

    assert homes.rows.height > 0
    assert set(homes.rows["property_type"].unique()) <= {"unit", "villa"}
    units = homes.rows.filter(homes.rows["property_type"] == "unit")
    assert set(units["property_sub_type"].unique()) <= set(HOME_UNIT_SUB_TYPES)
    assert homes.lineage["ingest_run_id"] == summary.run_id
    assert len(homes.lineage["source_sha256"]) == 64
    clean = homes.rows.filter(homes.rows["is_clean"])
    assert homes.data_end == clean["instance_date"].max()
    stat = homes.rows.filter(~homes.rows["is_clean"])
    assert stat.height == 0 or stat["instance_date"].min() >= config.test_start
    assert homes.areas.columns == ["area_id", "name_en"] and homes.areas.height > 0
    assert homes.aliases.columns == ["alias_key", "area_id"] and homes.aliases.height > 0
