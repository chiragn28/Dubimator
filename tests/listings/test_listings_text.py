import random

from listings.text import ListingFacts, make_description, make_title

FLAT = ListingFacts(
    bedrooms=2,
    size_sqm=96.5,
    area_name="Marsa Dubai",
    area_name_ar="مرسى دبي",
    building_name="Marina Gate 1",
    project_name="Marina Gate",
    property_type="unit",
    sub_kind="flat",
)
VILLA = ListingFacts(
    bedrooms=None,
    size_sqm=520.0,
    area_name="Hadaeq Sheikh Mohammed Bin Rashid",
    area_name_ar=None,
    building_name=None,
    project_name="Dubai Hills",
    property_type="villa",
    sub_kind="villa",
)


def test_text_is_deterministic_for_a_seed():
    first = make_description(random.Random(7), FLAT)
    second = make_description(random.Random(7), FLAT)
    assert first == second
    assert make_description(random.Random(8), FLAT) != first


def test_description_states_the_facts():
    text = make_description(random.Random(1), FLAT)
    assert "2" in text and "Marsa Dubai" in text
    assert "97 sqm" in text or "96" in text
    assert len(text) > 80


def test_title_mentions_kind_and_location():
    title = make_title(random.Random(2), FLAT)
    assert "Marsa Dubai" in title or "Marina Gate 1" in title
    assert "2" in title
    assert len(title) <= 120


def test_villa_without_bedrooms_or_building_still_reads_correctly():
    title, text = make_title(random.Random(3), VILLA), make_description(random.Random(3), VILLA)
    assert "None" not in title and "None" not in text
    assert "villa" in title.lower() or "villa" in text.lower()


def test_arabic_area_names_appear_in_some_descriptions():
    texts = [make_description(random.Random(seed), FLAT) for seed in range(40)]
    assert any("مرسى دبي" in text for text in texts)
    assert all(text.strip() for text in texts)
