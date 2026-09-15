from models.price.features import derive_segments


def test_size_basis_and_sub_kind(raw_homes):
    frame = derive_segments(
        raw_homes(
            {"property_type": "unit", "property_sub_type": "Flat"},
            {"property_type": "unit", "property_sub_type": "Hotel Apartment"},
            {"property_type": "unit", "property_sub_type": "Stacked Townhouses"},
            {"property_type": "villa", "property_sub_type": "Villa"},
            {"property_type": "villa", "property_sub_type": None},
        )
    )
    assert frame["size_basis"].to_list() == ["built_up", "built_up", "built_up", "built_up", "plot"]
    assert frame["sub_kind"].to_list() == ["flat", "hotel_apartment", "townhouse", "villa", "villa"]


def test_room_kind_and_bedrooms_for_every_rooms_value(raw_homes):
    rooms = ["Studio", "2 B/R", "Penthouse", "Single Room", None, "Office"]
    frame = derive_segments(raw_homes(*({"rooms": value} for value in rooms)))
    assert frame["room_kind"].to_list() == [
        "studio", "bedrooms", "penthouse", "single_room", "unknown", "unknown",
    ]  # fmt: skip
    assert frame["bedrooms"].to_list() == [0.0, 2.0, None, None, None, None]


def test_segment_name(raw_homes):
    frame = derive_segments(
        raw_homes(
            {"property_type": "unit", "reg_type": "ready"},
            {"property_type": "villa", "property_sub_type": None, "reg_type": "off_plan"},
        )
    )
    assert frame["segment"].to_list() == ["unit_ready_built_up", "villa_off_plan_plot"]
