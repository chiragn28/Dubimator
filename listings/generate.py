"""Generate the synthetic listings corpus from real DLD sales.

Real: area, building, project, size, bedrooms, price level, from dld.market_sales.
Invented: text, agents, posting dates, planted duplicates, fraud cases.
"""

import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from ingestion.config import DbSettings
from listings.config import HOME_UNIT_SUB_TYPES, PHOTO_ROOMS, CorpusConfig
from listings.photos import VARIANT_DIR, PhotoPool, make_variant
from listings.text import ListingFacts, make_description, make_title

SALES_SCHEMA = {
    "transaction_id": pl.Utf8,
    "instance_date": pl.Date,
    "property_type": pl.Utf8,
    "property_sub_type": pl.Utf8,
    "reg_type": pl.Utf8,
    "area_id": pl.Int64,
    "area_name": pl.Utf8,
    "building_name": pl.Utf8,
    "project_name": pl.Utf8,
    "bedrooms": pl.Int64,
    "area_sqm": pl.Float64,
    "price_aed": pl.Float64,
}
SALES_SQL = f"""
SELECT {", ".join(SALES_SCHEMA)}
FROM dld.market_sales
WHERE instance_date >= %(sales_from)s
  AND (property_type = 'villa'
       OR (property_type = 'unit' AND property_sub_type IN %(unit_sub_types)s))
  AND price_aed > 0 AND area_sqm > 0 AND area_id IS NOT NULL
ORDER BY transaction_id
"""
AREAS_SQL = "SELECT area_id, name_en, name_ar FROM dld.areas ORDER BY area_id"
AREAS_SCHEMA = {"area_id": pl.Int64, "name_en": pl.Utf8, "name_ar": pl.Utf8}

LISTING_SCHEMA = {
    "listing_id": pl.Int64,
    "source_transaction_id": pl.Utf8,
    "agent_id": pl.Int64,
    "posted_at": pl.Date,
    "title": pl.Utf8,
    "description": pl.Utf8,
    "asking_price_aed": pl.Float64,
    "area_id": pl.Int64,
    "area_name": pl.Utf8,
    "building_name": pl.Utf8,
    "project_name": pl.Utf8,
    "property_type": pl.Utf8,
    "property_sub_type": pl.Utf8,
    "reg_type": pl.Utf8,
    "size_sqm": pl.Float64,
    "bedrooms": pl.Int64,
    "photo_set_id": pl.Int64,
    "dup_group_id": pl.Int64,
    "control_group_id": pl.Utf8,
    "fraud_label": pl.Utf8,
    "is_synthetic": pl.Boolean,
}
PHOTO_SCHEMA = {
    "photo_id": pl.Int64,
    "set_id": pl.Int64,
    "room": pl.Utf8,
    "path": pl.Utf8,
    "variant_of": pl.Int64,
    "variant_kind": pl.Utf8,
    "is_stock": pl.Boolean,
}
LISTING_PHOTO_SCHEMA = {"listing_id": pl.Int64, "photo_id": pl.Int64, "position": pl.Int64}
SUB_KINDS = {
    "Flat": "flat",
    "Hotel Apartment": "hotel_apartment",
    "Stacked Townhouses": "townhouse",
}
ROOM_POSITION = {room: index for index, room in enumerate(PHOTO_ROOMS)}


@dataclass(frozen=True)
class Corpus:
    listings: pl.DataFrame
    photos: pl.DataFrame
    listing_photos: pl.DataFrame
    counts: dict[str, int]


def load_sales(settings: DbSettings, config: CorpusConfig) -> pl.DataFrame:
    params = {"sales_from": config.sales_from, "unit_sub_types": HOME_UNIT_SUB_TYPES}
    conn = settings.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(SALES_SQL, params)
            return pl.DataFrame(cur.fetchall(), schema=SALES_SCHEMA, orient="row")
    finally:
        conn.close()


def load_areas(settings: DbSettings) -> pl.DataFrame:
    conn = settings.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(AREAS_SQL)
            return pl.DataFrame(cur.fetchall(), schema=AREAS_SCHEMA, orient="row")
    finally:
        conn.close()


def _sub_kind(property_type: str, sub_type: str | None) -> str:
    if property_type == "villa":
        return "villa"
    kind = SUB_KINDS.get(sub_type or "")
    if kind is None:
        raise ValueError(f"unexpected unit sub-type {sub_type!r}")
    return kind


def _sample_base_sales(sales: pl.DataFrame, config: CorpusConfig) -> pl.DataFrame:
    """Sales for the base listings: whole groups from busy buildings, then the rest."""
    key = ["area_id", "building_name"]
    per_building = 3
    with_building = sales.filter(pl.col("building_name").is_not_null())
    busy = (
        with_building.group_by(key)
        .len()
        .filter(pl.col("len") >= config.min_building_sales)
        .select(key)
    )
    ranked = (
        with_building.join(busy, on=key, how="inner", maintain_order="left")
        .with_columns(pl.col("transaction_id").hash(seed=config.seed).alias("__h"))
        .sort([*key, "__h"])
        .with_columns(pl.int_range(pl.len()).over(key).alias("__rank"))
        .filter(pl.col("__rank") < per_building)
        .drop("__h", "__rank")
    )
    # Take the busy buildings in seeded-hash order, not area/building order: sorted order would
    # fill the control groups from the lowest area ids only, and a control false-positive rate
    # measured on three neighbourhoods would generalise to nothing.
    group_key = pl.concat_str(
        [pl.col("area_id").cast(pl.Utf8), pl.lit(":"), pl.col("building_name")]
    )
    sizes = (
        ranked.group_by(key, maintain_order=True)
        .len()
        .with_columns(group_key.hash(seed=config.seed + 3).alias("__gh"))
        .sort("__gh")
        .with_columns(pl.col("len").cum_sum().alias("__cum"))
    )
    wanted = sizes.filter(pl.col("__cum") - pl.col("len") < config.n_from_busy_buildings).select(
        key
    )
    busy_rows = ranked.join(wanted, on=key, how="inner", maintain_order="left").with_columns(
        pl.concat_str([pl.lit("bldg:"), group_key]).alias("control_group_id")
    )
    remaining = config.n_base - busy_rows.height
    rest = sales.join(
        busy_rows.select("transaction_id"), on="transaction_id", how="anti", maintain_order="left"
    )
    if remaining < 0:
        raise ValueError(
            f"busy-building sample is {busy_rows.height} rows, more than n_base={config.n_base}"
        )
    if rest.height < remaining:
        raise ValueError(
            f"not enough sales: need {remaining} more for n_base={config.n_base}, have {rest.height}"
        )
    others = rest.sample(n=remaining, seed=config.seed, shuffle=True).with_columns(
        pl.lit(None, pl.Utf8).alias("control_group_id")
    )
    columns = [*sales.columns, "control_group_id"]
    return pl.concat([busy_rows.select(columns), others.select(columns)]).sample(
        fraction=1.0, shuffle=True, seed=config.seed + 1
    )


def _base_photo_rows(set_ids, stock_sets: set[int]) -> pl.DataFrame:
    rows = []
    for set_id in sorted(set_ids):
        for room in PHOTO_ROOMS:
            rows.append(
                {
                    "photo_id": len(rows) + 1,
                    "set_id": set_id,
                    "room": room,
                    "path": f"photos/{set_id}_{room}.jpg",
                    "variant_of": None,
                    "variant_kind": None,
                    "is_stock": set_id in stock_sets,
                }
            )
    return pl.DataFrame(rows, schema=PHOTO_SCHEMA)


def _plain_sets_by_area(plain: list[int], area_ids: list[int], config: CorpusConfig) -> dict:
    """Give every plain photo set one or two home areas.

    Without this a plain set is drawn uniformly across the corpus, so at production scale each
    one spans far more than stock_min_areas areas: the photo_reuse rule would fire on nearly
    every listing and tens of thousands of unrelated pairs would share byte-identical photos.
    Round-robin over a seeded shuffle of the areas, so every area is covered before any area
    gets a second set, and no set ever reaches more than two areas.
    """
    if len(plain) < len(area_ids):
        # Fewer local sets than areas makes the invariant unsatisfiable: some set would have to
        # serve enough areas to trip the photo_reuse rule. Say so here rather than letting
        # _validate report a confusing spread failure much later.
        raise ValueError(
            f"photo pool has {len(plain)} non-stock sets for {len(area_ids)} areas; each area "
            f"needs at least one local set, so use a larger pool or fewer areas"
        )
    rng = random.Random(config.seed + 5)
    order = sorted(area_ids)
    rng.shuffle(order)
    by_area: dict[int, list[int]] = {area_id: [] for area_id in sorted(area_ids)}
    cursor = 0
    for set_id in plain:
        homes = set()
        for _ in range(rng.randint(1, 2)):
            homes.add(order[cursor % len(order)])
            cursor += 1
        for area_id in sorted(homes):
            by_area[area_id].append(set_id)
    return by_area


def _build_base_listings(
    base: pl.DataFrame,
    areas: pl.DataFrame,
    config: CorpusConfig,
    stock: list[int],
    plain: list[int],
) -> pl.DataFrame:
    rng = random.Random(config.seed)
    arabic = dict(zip(areas["area_id"].to_list(), areas["name_ar"].to_list()))
    start = date.fromisoformat(config.posted_from)
    span = (date.fromisoformat(config.posted_to) - start).days - config.repost_days[1]
    if span <= 0:
        raise ValueError("posting window is shorter than the repost delay")
    if config.n_bait_price > base.height:
        raise ValueError(
            f"n_bait_price={config.n_bait_price} exceeds the {base.height} base listings"
        )
    plain_by_area = _plain_sets_by_area(plain, base["area_id"].unique().to_list(), config)
    # Bait multiplies the DLD sale price, per the spec — not the asking price, which would
    # compound the two factors into an effective 0.40-0.70x. Chosen up front so the factor can
    # be applied where the sale price is still in hand; the sale price is never stored.
    bait_rng = random.Random(config.seed + 17)
    bait_factors = {
        listing_id: bait_rng.uniform(*config.bait_factor)
        for listing_id in bait_rng.sample(range(1, base.height + 1), config.n_bait_price)
    }
    rows = []
    for index, sale in enumerate(base.iter_rows(named=True)):
        listing_id = index + 1
        if rng.random() < config.stock_share:
            photo_set = rng.choice(stock)
        else:
            # Falls back to the whole plain pool only if an area got no home set at all;
            # _validate then fails loudly rather than letting the spread go unnoticed.
            photo_set = rng.choice(plain_by_area.get(sale["area_id"]) or plain)
        # Always drawn, so which listings are bait does not perturb the rest of the stream.
        factor = rng.uniform(*config.asking_factor)
        bait = bait_factors.get(listing_id)
        asking = round(sale["price_aed"] * (bait if bait is not None else factor) / 10_000) * 10_000
        facts = ListingFacts(
            bedrooms=sale["bedrooms"],
            size_sqm=sale["area_sqm"],
            area_name=sale["area_name"],
            area_name_ar=arabic.get(sale["area_id"]),
            building_name=sale["building_name"],
            project_name=sale["project_name"],
            property_type=sale["property_type"],
            sub_kind=_sub_kind(sale["property_type"], sale["property_sub_type"]),
        )
        text_rng = random.Random(config.seed * 7919 + listing_id)
        rows.append(
            {
                "listing_id": listing_id,
                "source_transaction_id": sale["transaction_id"],
                "agent_id": rng.randrange(1, config.n_agents + 1),
                "posted_at": start + timedelta(days=rng.randint(0, span)),
                "title": make_title(text_rng, facts),
                "description": make_description(text_rng, facts),
                "asking_price_aed": float(asking),
                "area_id": sale["area_id"],
                "area_name": sale["area_name"],
                "building_name": sale["building_name"],
                "project_name": sale["project_name"],
                "property_type": sale["property_type"],
                "property_sub_type": sale["property_sub_type"],
                "reg_type": sale["reg_type"],
                "size_sqm": sale["area_sqm"],
                "bedrooms": sale["bedrooms"],
                "photo_set_id": photo_set,
                "dup_group_id": None,
                "control_group_id": sale["control_group_id"],
                "fraud_label": "bait_price" if bait is not None else None,
                "is_synthetic": True,
            }
        )
    return pl.DataFrame(rows, schema=LISTING_SCHEMA)


def _area_spread(listings: pl.DataFrame, set_ids: list[int]) -> pl.DataFrame:
    return (
        listings.filter(pl.col("photo_set_id").is_in(set_ids))
        .group_by("photo_set_id")
        .agg(pl.col("area_id").n_unique().alias("areas"))
    )


def _base_listing_photos(listings: pl.DataFrame, photos: pl.DataFrame) -> pl.DataFrame:
    base = (
        photos.filter(pl.col("variant_of").is_null())
        .select(pl.col("set_id").alias("photo_set_id"), "photo_id", "room")
        .with_columns(
            pl.col("room").replace_strict(ROOM_POSITION, return_dtype=pl.Int64).alias("position")
        )
    )
    return (
        listings.select("listing_id", "photo_set_id")
        .join(base, on="photo_set_id", how="inner")
        .select("listing_id", "photo_id", "position")
        .sort(["listing_id", "position"])
        .cast(LISTING_PHOTO_SCHEMA)
    )


def _plant_clones(
    listings: pl.DataFrame,
    photos: pl.DataFrame,
    listing_photos: pl.DataFrame,
    arabic: dict[int, str | None],
    config: CorpusConfig,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Clone base listings into exact reposts, reworded copies and edited-photo copies."""
    rng = random.Random(config.seed + 99)
    photos_by_listing: dict[int, list[int]] = {}
    for row in listing_photos.sort(["listing_id", "position"]).iter_rows(named=True):
        photos_by_listing.setdefault(row["listing_id"], []).append(row["photo_id"])
    photo_meta = {row["photo_id"]: row for row in photos.iter_rows(named=True)}
    by_id = {row["listing_id"]: row for row in listings.iter_rows(named=True)}

    kinds = (
        ["exact"] * config.n_exact_repost
        + ["reworded"] * config.n_reworded
        + ["edited"] * config.n_edited_photo
    )
    sources = rng.sample(sorted(by_id), len(kinds))
    shifted = set(rng.sample(range(config.n_exact_repost), config.n_price_shifted_reposts))
    next_listing_id = max(by_id) + 1
    next_photo_id = int(photos["photo_id"].max()) + 1

    clone_rows, clone_photo_rows, variant_rows = [], [], []
    for position, (source_id, kind) in enumerate(zip(sources, kinds)):
        source = by_id[source_id]
        listing_id = next_listing_id + position
        agent = rng.randrange(1, config.n_agents + 1)
        while agent == source["agent_id"] and config.n_agents > 1:
            agent = rng.randrange(1, config.n_agents + 1)

        asking = source["asking_price_aed"]
        if kind == "exact" and position in shifted:
            direction = rng.choice((-1.0, 1.0))
            moved = asking * (1.0 + direction * rng.uniform(*config.price_shift))
            asking = float(round(moved / 10_000) * 10_000)

        title, description = source["title"], source["description"]
        if kind == "reworded":
            facts = ListingFacts(
                bedrooms=source["bedrooms"],
                size_sqm=source["size_sqm"],
                area_name=source["area_name"],
                area_name_ar=arabic.get(source["area_id"]),
                building_name=source["building_name"],
                project_name=source["project_name"],
                property_type=source["property_type"],
                sub_kind=_sub_kind(source["property_type"], source["property_sub_type"]),
            )
            for attempt in range(5):
                text_rng = random.Random(config.seed * 7919 + listing_id + 500_000 + attempt * 7)
                title, description = make_title(text_rng, facts), make_description(text_rng, facts)
                if description != source["description"]:
                    break
            else:
                raise ValueError(f"could not reword listing {source_id} into different text")

        source_photos = photos_by_listing[source_id]
        if kind == "edited":
            photo_ids = []
            for photo_id in source_photos:
                meta = photo_meta[photo_id]
                variant_rows.append(
                    {
                        "photo_id": next_photo_id,
                        "set_id": meta["set_id"],
                        "room": meta["room"],
                        "path": f"variants/{next_photo_id}.jpg",
                        "variant_of": photo_id,
                        "variant_kind": "edited",
                        "is_stock": meta["is_stock"],
                    }
                )
                photo_ids.append(next_photo_id)
                next_photo_id += 1
        else:
            photo_ids = source_photos
        for slot, photo_id in enumerate(photo_ids):
            clone_photo_rows.append(
                {"listing_id": listing_id, "photo_id": photo_id, "position": slot}
            )

        clone_rows.append(
            {
                **source,
                "listing_id": listing_id,
                "agent_id": agent,
                "posted_at": source["posted_at"] + timedelta(days=rng.randint(*config.repost_days)),
                "title": title,
                "description": description,
                "asking_price_aed": asking,
                "dup_group_id": source_id,
                "control_group_id": None,
                # fraud_label is inherited from **source on purpose: a repost of a bait-priced
                # listing carries the same cheap asking price, so it is bait too and a detector
                # that flags it is right. Only control_group_id is cleared — a clone is never a
                # member of a must-not-flag group.
            }
        )
    return (
        pl.DataFrame(clone_rows, schema=LISTING_SCHEMA),
        pl.DataFrame(clone_photo_rows, schema=LISTING_PHOTO_SCHEMA),
        pl.DataFrame(variant_rows, schema=PHOTO_SCHEMA),
    )


def _validate(listings: pl.DataFrame, photos: pl.DataFrame, config: CorpusConfig) -> None:
    if listings.height != config.n_listings:
        raise ValueError(f"generated {listings.height} listings, expected {config.n_listings}")
    stock_sets = photos.filter(pl.col("is_stock"))["set_id"].unique().to_list()
    spread = _area_spread(listings, stock_sets)
    thin = spread.filter(pl.col("areas") < config.stock_min_areas)
    if spread.height < len(stock_sets) or thin.height:
        raise ValueError(
            f"every stock photo set must span >= {config.stock_min_areas} areas; "
            f"unused or thin sets: {thin.to_dicts()}"
        )
    # The mirror of the check above, and just as load-bearing: if a plain set also went
    # citywide then photo_reuse would fire on nearly every listing and unrelated listings
    # would share byte-identical photos, which is what the photos-only baseline is measured
    # against. A plain set must stay local.
    plain_sets = photos.filter(~pl.col("is_stock"))["set_id"].unique().to_list()
    wide = _area_spread(listings, plain_sets).filter(pl.col("areas") >= config.stock_min_areas)
    if wide.height:
        raise ValueError(
            f"every non-stock photo set must stay under {config.stock_min_areas} areas; "
            f"citywide sets: {wide.to_dicts()}"
        )


def generate_corpus(
    sales: pl.DataFrame, areas: pl.DataFrame, pool_set_ids, config: CorpusConfig
) -> Corpus:
    if len(pool_set_ids) <= config.n_stock_sets:
        raise ValueError(
            f"photo pool has {len(pool_set_ids)} sets; need more than n_stock_sets="
            f"{config.n_stock_sets}"
        )
    shuffled = sorted(pool_set_ids)
    random.Random(config.seed).shuffle(shuffled)
    stock = sorted(shuffled[: config.n_stock_sets])
    plain = sorted(shuffled[config.n_stock_sets :])

    photos = _base_photo_rows(pool_set_ids, set(stock))
    base = _build_base_listings(_sample_base_sales(sales, config), areas, config, stock, plain)
    base_photos = _base_listing_photos(base, photos)

    arabic = dict(zip(areas["area_id"].to_list(), areas["name_ar"].to_list()))
    clones, clone_photos, variants = _plant_clones(base, photos, base_photos, arabic, config)

    listings = pl.concat([base, clones])
    all_photos = pl.concat([photos, variants])
    listing_photos = pl.concat([base_photos, clone_photos])
    _validate(listings, all_photos, config)
    counts = {
        "listings": listings.height,
        "base": base.height,
        "exact_repost": config.n_exact_repost,
        "reworded": config.n_reworded,
        "edited_photo": config.n_edited_photo,
        # bait_price counts the planted base listings; clones that inherited the label are
        # counted separately so the corpus total stays visible without conflating the two.
        "bait_price": int(
            listings.filter(
                (pl.col("fraud_label") == "bait_price") & pl.col("dup_group_id").is_null()
            ).height
        ),
        "bait_price_clones": int(
            listings.filter(
                (pl.col("fraud_label") == "bait_price") & pl.col("dup_group_id").is_not_null()
            ).height
        ),
        "price_shifted": config.n_price_shifted_reposts,
        "photos": all_photos.height,
        "variants": variants.height,
    }
    return Corpus(listings, all_photos, listing_photos, counts)


def render_variants(
    corpus: Corpus, pool: PhotoPool, config: CorpusConfig, variant_dir: Path = VARIANT_DIR
) -> int:
    """Write one edited image per variant row. Returns how many files were written."""
    variant_dir = Path(variant_dir)
    sources = {
        row["photo_id"]: (row["set_id"], row["room"]) for row in corpus.photos.iter_rows(named=True)
    }
    written = 0
    for row in corpus.photos.filter(pl.col("variant_kind") == "edited").iter_rows(named=True):
        set_id, room = sources[row["variant_of"]]
        make_variant(pool.path(set_id, room), variant_dir / Path(row["path"]).name, config)
        written += 1
    return written
