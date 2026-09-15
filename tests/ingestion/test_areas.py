import polars as pl

from ingestion.areas import CURATED_ALIASES_PATH, build_area_aliases, build_areas

COLUMNS = {
    "area_id": pl.Int64,
    "area_name": pl.Utf8,
    "area_name_ar": pl.Utf8,
    "master_project": pl.Utf8,
    "exclusion_reason": pl.Utf8,
}


def frame(rows):
    return pl.DataFrame(rows, schema=COLUMNS, orient="row")


def rows_for(area_id, name, n, master=None, reason=None, name_ar=None):
    return [(area_id, name, name_ar, master, reason)] * n


def aliases_of(aliases, area_id):
    return set(aliases.filter(pl.col("area_id") == area_id)["alias_key"].to_list())


def write_curated(tmp_path, text):
    path = tmp_path / "curated.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_build_areas_counts_market_sales_and_picks_most_common_name():
    data = frame(
        rows_for(364, "Marsa Dubai", 3, name_ar="مرسى دبي")
        + rows_for(364, "Marsa Dubai", 2, reason="mortgage", name_ar="مرسى دبي")
        + rows_for(364, "Marsa  Dubai Old", 1, name_ar="مرسى")
        + rows_for(7, "Al-Nahdah", 1)
    )
    areas = build_areas(data)
    assert areas.columns == ["area_id", "name_en", "name_ar", "match_key", "market_sales"]
    assert areas.to_dicts() == [
        {
            "area_id": 7,
            "name_en": "Al-Nahdah",
            "name_ar": None,
            "match_key": "nahdah",
            "market_sales": 1,
        },
        {
            "area_id": 364,
            "name_en": "Marsa Dubai",
            "name_ar": "مرسى دبي",
            "match_key": "marsa dubai",
            "market_sales": 4,
        },
    ]


def test_build_areas_skips_null_ids_and_fills_missing_names():
    areas = build_areas(frame(rows_for(None, "Ghost", 2) + rows_for(9, None, 1)))
    assert areas.to_dicts() == [
        {
            "area_id": 9,
            "name_en": "Area 9",
            "name_ar": None,
            "match_key": "area 9",
            "market_sales": 1,
        }
    ]


def test_official_alias_can_map_to_several_areas(tmp_path):
    data = frame(rows_for(404, "Mushrif", 1) + rows_for(420, "Mushrif", 1))
    areas = build_areas(data)
    aliases, _ = build_area_aliases(data, areas, write_curated(tmp_path, "alias,area_name_en\n"))
    mushrif = aliases.filter(pl.col("alias_key") == "mushrif")
    assert sorted(mushrif["area_id"].to_list()) == [404, 420]
    assert set(mushrif["source"].to_list()) == {"official"}


def test_master_project_alias_needs_50_rows_and_80_percent_share(tmp_path):
    data = frame(
        rows_for(1, "Marsa Dubai", 80, master="Dubai Marina")
        + rows_for(2, "Al Thanyah Fifth", 20, master="Dubai Marina")
        + rows_for(3, "Burj Khalifa", 49, master="Downtown Towers")
        + rows_for(4, "Business Bay", 60, master="Split Project")
        + rows_for(5, "Al Barsha", 60, master="Split Project")
    )
    areas = build_areas(data)
    aliases, _ = build_area_aliases(data, areas, write_curated(tmp_path, "alias,area_name_en\n"))
    master = aliases.filter(pl.col("source") == "master_project")
    assert master.select("alias_key", "area_id").rows() == [("dubai marina", 1)]


def test_curated_aliases_resolve_and_report_unresolved(tmp_path):
    data = frame(rows_for(1, "Marsa Dubai", 5))
    areas = build_areas(data)
    curated = write_curated(tmp_path, "alias,area_name_en\nJBR,Marsa Dubai\nJLT,Al Thanyah Fifth\n")
    aliases, unresolved = build_area_aliases(data, areas, curated)
    assert ("jbr", 1, "curated") in aliases.select("alias_key", "area_id", "source").rows()
    assert unresolved == ["JLT"]


def test_duplicate_alias_keeps_first_source(tmp_path):
    data = frame(rows_for(1, "Business Bay", 60, master="Business Bay"))
    areas = build_areas(data)
    curated = write_curated(tmp_path, "alias,area_name_en\nBusiness Bay,Business Bay\n")
    aliases, _ = build_area_aliases(data, areas, curated)
    assert aliases.rows() == [("business bay", "Business Bay", 1, "official")]


def test_shipped_curated_file_is_well_formed():
    curated = pl.read_csv(CURATED_ALIASES_PATH, infer_schema=False)
    assert curated.columns == ["alias", "area_name_en"]
    assert curated.height == 11
    assert curated.null_count().sum_horizontal()[0] == 0
