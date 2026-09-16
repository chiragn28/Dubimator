"""Synthetic search queries with their true slots.

Each query is seeded from a real (non-fraud) listing, then rendered through one of N_TEMPLATES
frames: a prefix, a slot order, and a phrasing style for bedrooms and budget. Queries are split
by frame, so the report split uses frames the ranker never saw. This module reads ground truth
(fraud labels) only to choose seeds and to grade.
"""

import json
import math
import random

import polars as pl

from ingestion.normalize import map_unique, match_key
from listings.text import AMENITIES
from search.config import KIND_SQL, QUERY_KINDS, SPLITS, SQFT_TO_SQM, SearchConfig, listing_kind
from search.grade import GRADE_SCHEMA, grade_frame
from search.parse import AMENITY_SYNONYMS

TRUE_SLOT_KEYS = (
    "area_id", "area_name", "building", "bedrooms", "property_type",
    "budget_min", "budget_max", "min_size_sqm", "amenities",
)  # fmt: skip
PREFIXES = ("", "looking for", "need a", "searching for", "want to buy", "show me")
ORDERS = (
    ("beds_type", "area", "budget", "size", "amenities"),
    ("area", "beds_type", "budget", "amenities", "size"),
    ("beds_type", "budget", "area", "size", "amenities"),
    ("budget", "beds_type", "area", "amenities", "size"),
    ("beds_type", "amenities", "area", "budget", "size"),
)
N_TEMPLATES = len(PREFIXES) * len(ORDERS)
VAGUE_WORDS = ("family", "investment", "quiet", "modern", "spacious", "affordable", "luxury")
TYPE_WORDS = {
    "flat": ("apartment", "flat"),
    "hotel_apartment": ("hotel apartment", "serviced apartment"),
    "townhouse": ("townhouse",),
    "villa": ("villa",),
}
NUMBER_NAMES = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}
MAX_PHRASED_BEDROOMS = 7
MIN_BUDGET = 10_000.0
NO_MATCH_BUDGET_SHARE = 0.3
PHRASES_BY_AMENITY: dict[str, list[str]] = {}
for _phrase, _canonical in sorted(AMENITY_SYNONYMS.items()):
    PHRASES_BY_AMENITY.setdefault(_canonical, []).append(_phrase)

GRADING_SQL = f"""
SELECT l.listing_id, l.area_id, l.area_name, {KIND_SQL} AS kind, l.building_name,
       l.project_name, l.bedrooms, l.size_sqm, l.asking_price_aed, l.description, l.fraud_label
FROM listings.listings AS l
ORDER BY l.listing_id
"""
GRADING_READ_SCHEMA = {
    "listing_id": pl.Int64,
    "area_id": pl.Int64,
    "area_name": pl.Utf8,
    "kind": pl.Utf8,
    "building_name": pl.Utf8,
    "project_name": pl.Utf8,
    "bedrooms": pl.Int64,
    "size_sqm": pl.Float64,
    "asking_price_aed": pl.Float64,
    "description": pl.Utf8,
    "fraud_label": pl.Utf8,
}


def _with_keys(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        map_unique(frame["building_name"], match_key, pl.Utf8).alias("building_key"),
        map_unique(frame["project_name"], match_key, pl.Utf8).alias("project_key"),
    ).select(*GRADE_SCHEMA, "area_name", "building_name", "project_name")


def grading_frame(listings: pl.DataFrame) -> pl.DataFrame:
    """Phase 4 listing columns -> the frame grade_frame and generate_queries read."""
    kinds = [
        listing_kind(property_type, sub_type)
        for property_type, sub_type in listings.select("property_type", "property_sub_type").rows()
    ]
    frame = listings.with_columns(pl.Series("kind", kinds, dtype=pl.Utf8)).with_columns(
        pl.col("bedrooms").cast(pl.Int64), pl.col("area_id").cast(pl.Int64)
    )
    return _with_keys(frame)


def load_grading_listings(conn) -> pl.DataFrame:
    with conn.cursor() as cur:
        cur.execute(GRADING_SQL)
        frame = pl.DataFrame(cur.fetchall(), schema=GRADING_READ_SCHEMA, orient="row")
    return _with_keys(frame)


def load_area_aliases(conn) -> dict[int, list[str]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT area_id, alias FROM dld.area_aliases WHERE source <> 'official' "
            "ORDER BY area_id, alias"
        )
        rows = cur.fetchall()
    aliases: dict[int, list[str]] = {}
    for area_id, alias in rows:
        aliases.setdefault(area_id, []).append(alias)
    return aliases


def assign_template_splits(seed: int, shares: tuple[float, float, float]) -> dict[int, str]:
    template_ids = list(range(N_TEMPLATES))
    random.Random(seed).shuffle(template_ids)
    n_train = round(N_TEMPLATES * shares[0])
    n_tune = round(N_TEMPLATES * shares[1])
    splits = {}
    for position, template_id in enumerate(template_ids):
        if position < n_train:
            splits[template_id] = SPLITS[0]
        elif position < n_train + n_tune:
            splits[template_id] = SPLITS[1]
        else:
            splits[template_id] = SPLITS[2]
    return splits


def true_slots(row_json: str) -> dict:
    return json.loads(row_json)


def _ceil_nice(amount: float) -> float:
    step = 100_000.0 if amount >= 1_000_000 else 10_000.0
    return max(MIN_BUDGET, math.ceil(amount / step) * step)


def _floor_nice(amount: float) -> float:
    step = 100_000.0 if amount >= 1_000_000 else 10_000.0
    return max(MIN_BUDGET, math.floor(amount / step) * step)


def _money_text(rng: random.Random, amount: float) -> str:
    if amount >= 1_000_000:
        millions = f"{amount / 1e6:g}"
        options = (f"{millions}M", f"{amount:,.0f}", f"AED {millions} million", f"{millions}m AED")
    else:
        thousands = f"{amount / 1e3:g}"
        options = (f"{thousands}k", f"{amount:,.0f}", f"AED {thousands}k")
    return rng.choice(options)


def _budget_text(rng: random.Random, low: float | None, high: float, style: int) -> str:
    top = _money_text(rng, high)
    if low is None:
        return ("under {}", "below {}", "max {}", "up to {}", "budget {}")[style].format(top)
    bottom = _money_text(rng, low)
    template = ("{}-{}", "between {} and {}", "from {} to {}", "{} to {}", "{} - {}")[style]
    return template.format(bottom, top)


def _beds_text(bedrooms: int, style: int) -> str:
    if bedrooms == 0:
        return "studio"
    return (f"{bedrooms}BR", f"{bedrooms} bed", f"{NUMBER_NAMES[bedrooms]} bedroom")[style]


def _choose_slots(
    rng: random.Random,
    listing: dict,
    kind: str,
    aliases_by_area: dict[int, list[str]],
    min_price: dict[tuple[int, str], float],
    config: SearchConfig,
) -> dict:
    slots: dict = dict.fromkeys(TRUE_SLOT_KEYS)
    slots["amenities"] = []
    render = {"area_text": None, "amenity_words": [], "vague": None, "size_text": None}
    description = (listing["description"] or "").lower()
    present = [amenity for amenity in AMENITIES if amenity in description]
    bedrooms = listing["bedrooms"]
    available = ["area", "property_type"]
    if bedrooms is not None and 0 <= bedrooms <= MAX_PHRASED_BEDROOMS:
        available.append("bedrooms")
    available += ["budget", "min_size_sqm"]
    if present:
        available.append("amenities")

    if kind == "vague":
        pool = [slot for slot in ("area", "property_type", "amenities") if slot in available]
        chosen = rng.sample(pool, rng.randint(1, min(2, len(pool))))
        render["vague"] = rng.choice(VAGUE_WORDS)
    elif kind == "specified":
        chosen = rng.sample(available, rng.randint(3, len(available)))
    else:  # no_match: area + type + an impossible budget, plus any other slots
        rest = [slot for slot in available if slot not in ("area", "property_type", "budget")]
        chosen = ["area", "property_type", "budget", *rng.sample(rest, rng.randint(0, len(rest)))]

    if "area" in chosen:
        slots["area_id"], slots["area_name"] = listing["area_id"], listing["area_name"]
        aliases = aliases_by_area.get(listing["area_id"], [])
        use_alias = aliases and rng.random() < config.alias_probability
        render["area_text"] = rng.choice(aliases) if use_alias else listing["area_name"]
        named = listing["building_name"] or listing["project_name"]
        if kind == "specified" and named and rng.random() < config.building_probability:
            slots["building"] = named
    if "property_type" in chosen:
        slots["property_type"] = listing["kind"]
    if "bedrooms" in chosen:
        slots["bedrooms"] = bedrooms
    if "budget" in chosen:
        price = listing["asking_price_aed"]
        if kind == "no_match":
            cheapest = min_price[(listing["area_id"], listing["kind"])]
            slots["budget_max"] = _floor_nice(NO_MATCH_BUDGET_SHARE * cheapest)
        else:
            slots["budget_max"] = _ceil_nice(price * (1 + rng.uniform(*config.budget_headroom)))
            if rng.random() < 0.5:
                low = _floor_nice(price * (1 - rng.uniform(*config.budget_headroom)))
                if low < slots["budget_max"]:
                    slots["budget_min"] = low
    if "min_size_sqm" in chosen:
        target = listing["size_sqm"] * (1 - rng.uniform(*config.size_slack))
        if rng.random() < 0.3:
            square_feet = max(10, int(target / SQFT_TO_SQM // 10 * 10))
            slots["min_size_sqm"] = round(square_feet * SQFT_TO_SQM, 2)
            render["size_text"] = rng.choice(
                (f"at least {square_feet:,} sq ft", f"over {square_feet} sqft")
            )
        else:
            square_metres = max(10, int(target // 10 * 10))
            slots["min_size_sqm"] = float(square_metres)
            render["size_text"] = rng.choice(
                (
                    f"over {square_metres} sqm",
                    f"{square_metres}+ sqm",
                    f"min {square_metres} sqm",
                    f"at least {square_metres} m2",
                )
            )
    if "amenities" in chosen:
        picked = rng.sample(present, min(len(present), rng.randint(1, 2)))
        slots["amenities"] = picked
        render["amenity_words"] = [rng.choice(PHRASES_BY_AMENITY[amenity]) for amenity in picked]
    return {**slots, **render}


def render_query(rng: random.Random, slots: dict, template_id: int) -> str:
    prefix = PREFIXES[template_id // len(ORDERS)]
    order = ORDERS[template_id % len(ORDERS)]
    beds_style, budget_style = template_id % 3, (template_id // 3) % 5
    words = []
    if slots["vague"]:
        words.append(slots["vague"])
    if slots["bedrooms"] is not None:
        words.append(_beds_text(slots["bedrooms"], beds_style))
    if slots["property_type"]:
        words.append(rng.choice(TYPE_WORDS[slots["property_type"]]))
    pieces = {"beds_type": " ".join(words), "area": "", "budget": "", "size": "", "amenities": ""}
    if slots["area_text"]:
        if slots["building"]:
            pieces["area"] = f"at {slots['building']}, {slots['area_text']}"
        else:
            connector = rng.choice(("in", "at", "near", ""))
            pieces["area"] = f"{connector} {slots['area_text']}".strip()
    if slots["budget_max"] is not None:
        pieces["budget"] = _budget_text(rng, slots["budget_min"], slots["budget_max"], budget_style)
    if slots["size_text"]:
        pieces["size"] = slots["size_text"]
    if slots["amenity_words"]:
        pieces["amenities"] = "with " + " and ".join(slots["amenity_words"])
    parts = [prefix, *(pieces[name] for name in order)]
    return " ".join(" ".join(part for part in parts if part).split())


def generate_queries(
    grading: pl.DataFrame, aliases_by_area: dict[int, list[str]], config: SearchConfig
) -> pl.DataFrame:
    rng = random.Random(config.seed)
    splits = assign_template_splits(config.seed, config.split_shares)
    seeds = grading.filter(pl.col("fraud_label").is_null() & pl.col("kind").is_not_null())
    seed_rows = seeds.sort("listing_id").to_dicts()
    if not seed_rows:
        raise ValueError("no unlabelled listings to seed queries from")
    cheapest = (
        grading.filter(pl.col("kind").is_not_null())
        .group_by("area_id", "kind")
        .agg(pl.col("asking_price_aed").min())
    )
    min_price = {(a, k): p for a, k, p in cheapest.iter_rows()}
    rows, seen = [], set()
    attempts = 0
    while len(rows) < config.n_queries:
        attempts += 1
        if attempts > 20 * config.n_queries:
            raise RuntimeError(
                f"only {len(rows)} distinct queries after {attempts} attempts; "
                "the corpus is too small for n_queries"
            )
        listing = rng.choice(seed_rows)
        kind = rng.choices(QUERY_KINDS, weights=config.kind_shares)[0]
        template_id = rng.randrange(N_TEMPLATES)
        slots = _choose_slots(rng, listing, kind, aliases_by_area, min_price, config)
        text = render_query(rng, slots, template_id)
        if text in seen:
            continue
        seen.add(text)
        truth = {key: slots[key] for key in TRUE_SLOT_KEYS}
        grades = grade_frame(truth, grading, config.near_margin)
        rows.append(
            {
                "query_id": len(rows) + 1,
                "text": text,
                "template_id": template_id,
                "kind": kind,
                "split": splits[template_id],
                "true_slots": json.dumps(truth, allow_nan=False),
                "seed_listing_id": listing["listing_id"],
                "n_grade3": int((grades == 3).sum()),
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "query_id": pl.Int64,
            "text": pl.Utf8,
            "template_id": pl.Int64,
            "kind": pl.Utf8,
            "split": pl.Utf8,
            "true_slots": pl.Utf8,
            "seed_listing_id": pl.Int64,
            "n_grade3": pl.Int64,
        },
    )
