"""Seeded listing titles and descriptions. No text is copied from any real listing."""

import random
from dataclasses import dataclass

OPENERS = (
    "Available now",
    "Just listed",
    "Exclusive listing",
    "Priced to sell",
    "New to the market",
    "Rare opportunity",
)
AMENITIES = (
    "covered parking", "shared pool", "fully fitted kitchen", "built-in wardrobes",
    "balcony", "gym access", "24/7 security", "children's play area", "maid's room",
    "sea view", "community view", "central air conditioning",
)  # fmt: skip
BLURBS = (
    "Managed by an experienced local agent.",
    "Viewings available seven days a week.",
    "Handover documents ready for a quick transfer.",
    "Tenanted until the end of the quarter; investor friendly.",
    "Vacant on transfer.",
)
CALLS = (
    "Call today to arrange a viewing.",
    "Message us for the full photo set.",
    "Contact the listing agent for floor plans.",
    "Book a viewing this week.",
)
KIND_WORDS = {
    "flat": ("apartment", "flat"),
    "hotel_apartment": ("hotel apartment", "serviced apartment"),
    "townhouse": ("townhouse", "stacked townhouse"),
    "villa": ("villa", "family villa"),
}


@dataclass(frozen=True)
class ListingFacts:
    bedrooms: int | None
    size_sqm: float
    area_name: str
    area_name_ar: str | None
    building_name: str | None
    project_name: str | None
    property_type: str
    sub_kind: str


def _bedroom_phrase(bedrooms: int | None) -> str:
    if bedrooms is None:
        return ""
    if bedrooms == 0:
        return "Studio "
    return f"{bedrooms}-bedroom "


def _where(facts: ListingFacts) -> str:
    return facts.building_name or facts.project_name or facts.area_name


def make_title(rng: random.Random, facts: ListingFacts) -> str:
    kind = rng.choice(KIND_WORDS[facts.sub_kind])
    return f"{_bedroom_phrase(facts.bedrooms)}{kind} in {_where(facts)}, {facts.area_name}".strip()


def make_description(rng: random.Random, facts: ListingFacts) -> str:
    kind = rng.choice(KIND_WORDS[facts.sub_kind])
    area = facts.area_name
    if facts.area_name_ar and rng.random() < 0.25:
        area = f"{facts.area_name} ({facts.area_name_ar})"
    sentences = [
        f"{rng.choice(OPENERS)}: {_bedroom_phrase(facts.bedrooms).lower()}{kind} in {area}.",
        f"The property offers {facts.size_sqm:.0f} sqm of space"
        + (f" in {facts.building_name}." if facts.building_name else "."),
    ]
    amenities = rng.sample(AMENITIES, rng.randint(2, 5))
    sentences.append("Features include " + ", ".join(amenities) + ".")
    if facts.project_name and rng.random() < 0.6:
        sentences.append(f"Part of the {facts.project_name} development.")
    sentences.append(rng.choice(BLURBS))
    sentences.append(rng.choice(CALLS))
    return " ".join(sentences)
