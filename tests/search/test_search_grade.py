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
    slots = {"area_id": 2, "property_type": "flat", "bedrooms": 1, "budget_max": 1_500_000}
    grades = grade_frame(slots, frame)
    assert grades.name == "grade" and grades.len() == frame.height
    one_by_one = [grade(slots, row) for row in frame.to_dicts()]
    assert grades.to_list() == one_by_one
    assert set(one_by_one) >= {0, 1, 3}
