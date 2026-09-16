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
    ("3+ bedroom villa", {"bedrooms": 3, "property_type": "villa"}),
    ("2+ br apartment", {"bedrooms": 2, "property_type": "flat"}),
    # budget
    ("under 1.5M", {"budget_max": 1_500_000.0, "budget_min": None}),
    ("below AED 900k", {"budget_max": 900_000.0}),
    ("max 2,000,000", {"budget_max": 2_000_000.0}),
    ("up to 1.2 million", {"budget_max": 1_200_000.0}),
    ("budget 850k", {"budget_max": 850_000.0}),
    ("900k-1.2M", {"budget_min": 900_000.0, "budget_max": 1_200_000.0}),
    ("budget 900k—2M", {"budget_min": 900_000.0, "budget_max": 2_000_000.0}),
    ("1-1.5M", {"budget_min": 1_000_000.0, "budget_max": 1_500_000.0}),
    (
        "970,000-1.5M",
        {"budget_min": 970_000.0, "budget_max": 1_500_000.0},
    ),  # a comma-grouped first number is a complete amount; it must not inherit the "M"
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


def test_lexicon_is_hashable():
    assert isinstance(hash(LEXICON), int)


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
