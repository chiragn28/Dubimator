"""Shared rendering helpers for the Streamlit demo pages.

Every page imports from here instead of duplicating formatting or error
handling, so the "every page shows its data-as-of date, never shows a
number without its range, and shows a synthetic-listings note" rules in
the Phase 8 global constraints stay true in one place.
"""

from __future__ import annotations

import math

import streamlit as st

from demo import client as api_client
from demo.client import ApiProblem

DATA_NOTE = "Data as of 2023-03-17 (Dubai Land Department)."
SYNTHETIC_NOTE = "Listings are synthetic, generated over real DLD sales, and labelled."

# Listing IDs run 1-20000, but which ones are interesting depends on the detect run, so these
# are picked from the current corpus: one per outcome, so the listing check shows something on
# the first click instead of an empty result. Refresh them after rebuilding the corpus:
#   SELECT listing_id FROM listings.fraud_flags GROUP BY listing_id
#     ORDER BY count(DISTINCT flag) DESC LIMIT 1;                          -- the flagged one
#   SELECT listing_a FROM listings.duplicate_pairs WHERE decision LIMIT 1; -- the duplicate
EXAMPLE_LISTING_IDS: tuple[tuple[str, str], ...] = (
    ("8249", "flags"),
    ("19722", "duplicate"),
    ("1", "clean"),
)

_SETUP_CODES = frozenset({"no_key", "unreachable", "unauthorized"})
_SETUP_HINT = (
    "Set DEMO_API_URL and DEMO_API_KEY (in a .env file or your shell) and reload the page."
)


def show_problem(problem: ApiProblem) -> None:
    """Render an `ApiProblem` as a friendly error, with its request id.

    `no_key`, `unreachable` and `unauthorized` are the problems whoever runs
    the demo can fix themselves (a missing or wrong `DEMO_API_URL` /
    `DEMO_API_KEY`), so those get an extra setup hint.
    """
    st.error(problem.message)
    st.caption(f"request id: {problem.request_id or 'n/a'}")
    if problem.code in _SETUP_CODES:
        st.info(_SETUP_HINT)


def money(aed: float) -> str:
    """Format an AED amount, e.g. `money(1455490)` -> `"AED 1,455,490"`."""
    return f"AED {aed:,.0f}"


def forecast_rows(forecast: dict) -> list[dict]:
    """One row per horizon: horizon, point, low, high, confidence, note.

    The note is the horizon's `message` (e.g. a low-confidence caveat) when
    it is deployed, or its `reason` for not being deployed otherwise.
    """
    rows = []
    for horizon, key in (("3m", "forecast_3m"), ("1y", "forecast_1y"), ("3y", "forecast_3y")):
        data = forecast.get(key) or {}
        if data.get("status") == "not_deployed":
            rows.append(
                {
                    "horizon": horizon,
                    "point": None,
                    "low": None,
                    "high": None,
                    "confidence": None,
                    "note": data.get("reason", "not deployed"),
                }
            )
        else:
            rows.append(
                {
                    "horizon": horizon,
                    "point": data.get("point"),
                    "low": data.get("ci_low"),
                    "high": data.get("ci_high"),
                    "confidence": data.get("confidence"),
                    "note": data.get("message", ""),
                }
            )
    return rows


def render_component_status(ready: dict) -> None:
    """Render each API component's up/down state and version.

    `/v1/ready` nests them under `"components"`.
    """
    components = ready.get("components")
    if not isinstance(components, dict):
        return
    for name, info in components.items():
        if not isinstance(info, dict):
            continue
        up = info.get("up")
        version = info.get("version")
        icon = "🟢" if up else "🔴"
        label = f"{icon} **{name}**"
        if version:
            label += f" ({version})"
        st.markdown(label)
        if not up and info.get("error"):
            st.caption(info["error"])


@st.cache_data(ttl=600)
def cached_areas() -> dict:
    """`GET /v1/areas`, cached for 10 minutes (the spec's caching decision)."""
    return api_client.get_client().areas()


@st.cache_data(ttl=600)
def cached_area_history(area_id: int) -> dict:
    """`GET /v1/areas/{id}/history`, cached for 10 minutes."""
    return api_client.get_client().area_history(area_id)


@st.cache_data(ttl=600)
def cached_ready() -> dict:
    """`GET /v1/ready`, cached for 10 minutes."""
    return api_client.get_client().ready()


# The corpus images are American houses from the Houses-dataset, assigned to synthetic listings
# at random (listings/generate.py picks a set with rng.choice, never consulting property type),
# so no page shows them: beside a Dubai listing they read as a picture of the property and
# contradict it. The detector still embeds and compares them, which this note explains.
PHOTO_COMPARISON_NOTE = (
    "The detector also compared the listings' images; those are stand-ins from a public photo "
    "dataset, so they are scored but not shown."
)


# Streamlit's own colour-background markdown, so chips need no raw HTML.
_CHIP_COLOURS = {"neutral": "gray", "good": "green", "warn": "orange", "bad": "red"}


def chips(items, tone: str = "neutral") -> None:
    """A wrapped row of small labels, e.g. `chips(["3 bed", "423 m²"])`.

    `items` may be strings or `(text, tone)` pairs; a bare string takes `tone`.
    """
    parts = []
    for item in items:
        text, item_tone = item if isinstance(item, tuple) else (item, tone)
        if not text:
            continue
        colour = _CHIP_COLOURS.get(item_tone, "gray")
        # Escape the markdown delimiters that appear in real values (e.g. "1_bedroom").
        safe = str(text).replace("[", "(").replace("]", ")").replace("_", r"\_")
        parts.append(f":{colour}-background[{safe}]")
    if parts:
        st.markdown(" ".join(parts))


def _bedrooms_text(value) -> str | None:
    if value is None:
        return None
    return "studio" if value == 0 else f"{value} bed"


def parsed_chips(parsed: dict) -> list[tuple[str, str]]:
    """What the parser understood, as chips — the slots it filled, then what it couldn't.

    Replaces dumping the parsed-query JSON: a visitor wants to read "villa, 3 bed, in Dubai
    Marina, up to AED 3,000,000", and wants ignored words called out in amber rather than
    buried in a list.
    """
    out: list[tuple[str, str]] = []
    if parsed.get("property_type"):
        out.append((str(parsed["property_type"]).replace("_", " "), "good"))
    bedrooms = _bedrooms_text(parsed.get("bedrooms"))
    if bedrooms:
        out.append((bedrooms, "good"))
    if parsed.get("area_name"):
        out.append((f"in {parsed['area_name']}", "good"))
    if parsed.get("building"):
        out.append((f"building {parsed['building']}", "good"))
    low, high = parsed.get("budget_min"), parsed.get("budget_max")
    if low and high:
        out.append((f"{money(low)} to {money(high)}", "good"))
    elif high:
        out.append((f"up to {money(high)}", "good"))
    elif low:
        out.append((f"over {money(low)}", "good"))
    if parsed.get("min_size_sqm"):
        out.append((f"at least {parsed['min_size_sqm']:,.0f} m²", "good"))
    for amenity in parsed.get("amenities") or []:
        out.append((str(amenity).replace("_", " "), "good"))
    free_text = (parsed.get("free_text") or "").strip()
    if free_text:
        out.append((f"text: {free_text}", "neutral"))
    for word in parsed.get("unrecognised") or []:
        out.append((f"ignored: {word}", "warn"))
    return out


def _reason_tone(reason: str) -> str:
    """A ranking reason is either a match ("area OK") or a miss ("89% over budget")."""
    return "good" if "✓" in reason else "warn"


# The corpus photos are American houses from the Houses-dataset (listings/photos.py), assigned to
# listings at random with no regard for property type, so a 52 m2 studio can get a detached house
# with a lawn. Showing one beside "Studio apartment in Business Bay" contradicts the text, so
# search results get a drawn tile built from the listing's own facts instead. The real photos are
# still used where they carry meaning: the listing check, where reused photos are the finding.
_TILE_GLYPHS = {
    "apartment": "&#9974;",
    "hotel_apartment": "&#9974;",
    "townhouse": "&#8962;",
    "villa": "&#8962;",
}
_TILE_TINTS = (
    ("#0f3b46", "#15505e"),
    ("#12313f", "#1b4557"),
    ("#123a38", "#1a524d"),
    ("#1a2f45", "#24415f"),
)


def _tile_kind(hit: dict) -> str:
    """The property kind implied by a hit, from its title (the API sends no explicit kind)."""
    title = (hit.get("title") or "").lower()
    for kind in ("hotel apartment", "townhouse", "villa", "apartment"):
        if kind in title:
            return kind.replace(" ", "_")
    return "apartment"


def listing_tile(hit: dict) -> str:
    """An SVG stand-in for a listing photo, drawn from the listing's own facts.

    Honest by construction: it can only show what the listing actually says, so it can never
    contradict the text next to it the way a random stock photo does.
    """
    kind = _tile_kind(hit)
    glyph = _TILE_GLYPHS.get(kind, "&#9974;")
    seed = int(hit.get("listing_id") or 0)
    top, bottom = _TILE_TINTS[seed % len(_TILE_TINTS)]
    label = kind.replace("_", " ")
    size = f"{hit['size_sqm']:,.0f} m&#178;" if hit.get("size_sqm") else ""
    area = (hit.get("area_name") or "").replace("&", "&amp;").replace("<", "&lt;")
    return f"""
<div style="aspect-ratio:4/3;border-radius:8px;overflow:hidden;position:relative;
     background:linear-gradient(160deg,{top},{bottom});display:flex;flex-direction:column;
     align-items:center;justify-content:center;gap:4px;color:rgba(255,255,255,0.92)">
  <div style="font-size:2.4rem;line-height:1">{glyph}</div>
  <div style="font-size:0.78rem;letter-spacing:0.04em;text-transform:uppercase;
       color:rgba(255,255,255,0.75)">{label}</div>
  <div style="font-size:0.72rem;color:rgba(255,255,255,0.6)">{area}{" &middot; " if area and size else ""}{size}</div>
  <div style="position:absolute;bottom:6px;font-size:0.6rem;color:rgba(255,255,255,0.4)">
    no listing photo</div>
</div>
"""


def listing_card(hit: dict) -> None:
    """One search result, laid out like a property listing rather than a row of fields.

    The thumbnail is drawn from the listing's own facts rather than a corpus photo: see
    `listing_tile` for why.
    """
    with st.container(border=True):
        image_col, text_col = st.columns([1, 2.6], vertical_alignment="top")
        with image_col:
            st.markdown(listing_tile(hit), unsafe_allow_html=True)
        with text_col:
            price = hit.get("asking_price_aed")
            if price is not None:
                st.markdown(f"### {money(price)}")
            st.markdown(f"**{hit.get('title') or 'Untitled listing'}**")
            facts = [
                _bedrooms_text(hit.get("bedrooms")),
                f"{hit['size_sqm']:,.0f} m²" if hit.get("size_sqm") else None,
                hit.get("area_name"),
                f"#{hit['listing_id']}" if hit.get("listing_id") is not None else None,
            ]
            chips([fact for fact in facts if fact])
            reasons = hit.get("reasons") or []
            if reasons:
                chips([(reason, _reason_tone(reason)) for reason in reasons])
            hidden = hit.get("duplicates_hidden") or 0
            if hidden:
                plural = "" if hidden == 1 else "s"
                st.caption(f"{hidden} likely repost{plural} of this listing hidden")


def _pct(value, digits: int = 1) -> str:
    return f"{float(value) * 100:.{digits}f}%"


def flag_summary(flag: str, detail: dict) -> tuple[str, str, list[tuple[str, str]]]:
    """A fraud flag as (heading, sentence, chips) instead of its raw detail dict.

    An unknown flag falls back to its name with no sentence, so a detector added later still
    renders instead of disappearing, and a detail that is plain text (rather than the
    detector's dict) is shown as the sentence rather than crashing the page.
    """
    heading = flag.replace("_", " ").capitalize()
    if not isinstance(detail, dict):
        return heading, "" if detail is None else str(detail), []
    if flag == "bait_price":
        asking, low = detail.get("asking_price_aed"), detail.get("range_80_low")
        estimate, below = detail.get("estimate_aed"), detail.get("below_low_pct")
        if None not in (asking, low, estimate, below):
            sentence = (
                f"Asking {money(asking)} is {below:.1f}% below {money(low)}, the bottom of "
                f"the estimated range. The model values this home at {money(estimate)}."
            )
        else:
            sentence = "The asking price sits below the estimated range."
        tags = []
        if asking:
            tags.append((f"asking {money(asking)}", "bad"))
        if estimate:
            tags.append((f"estimate {money(estimate)}", "neutral"))
        return "Bait price", sentence, tags
    if flag == "inconsistent_relist":
        low, high = detail.get("min_price_aed"), detail.get("max_price_aed")
        spread, size = detail.get("spread"), detail.get("cluster_size")
        if None not in (low, high, spread, size):
            sentence = (
                f"The same home appears {size} times, priced from {money(low)} to "
                f"{money(high)} - a {_pct(spread, 0)} spread."
            )
        else:
            sentence = "The same home is listed more than once at inconsistent prices."
        tags = []
        if size:
            tags.append((f"{size} listings", "warn"))
        if spread is not None:
            tags.append((f"{_pct(spread, 0)} spread", "bad"))
        return "Inconsistent relist", sentence, tags
    if flag == "photo_reuse":
        listings, areas = detail.get("listings"), detail.get("areas")
        if None not in (listings, areas):
            sentence = (
                f"These photos also appear in {listings:,} other listings across {areas} "
                "areas, so they are agency stock rather than this home."
            )
        else:
            sentence = "These photos appear in many other listings."
        tags = []
        if listings:
            tags.append((f"{listings:,} listings", "bad"))
        if areas:
            tags.append((f"{areas} areas", "warn"))
        return "Reused photos", sentence, tags
    return heading, "", []


# The duplicate-pair signals worth showing, as (key, chip text when the signal is set).
_MATCH_SIGNALS = (
    ("same_building", "same building"),
    ("same_project", "same project"),
    ("same_area", "same area"),
    ("bedrooms_equal", "same bedrooms"),
    ("same_agent", "same agent"),
)


def duplicate_chips(signals: dict) -> list[tuple[str, str]]:
    """A duplicate pair's signals as readable chips instead of a dozen raw floats."""
    if not isinstance(signals, dict):
        # Older or hand-written payloads sometimes carry a list of signal names.
        return [(str(name), "neutral") for name in (signals or [])]
    out: list[tuple[str, str]] = []
    for key, text in _MATCH_SIGNALS:
        if signals.get(key):
            out.append((text, "good"))
    text_cosine = signals.get("text_cosine")
    if text_cosine is not None:
        tone = "good" if float(text_cosine) > 0.9 else "neutral"
        out.append((f"text {_pct(text_cosine, 0)} alike", tone))
    image_cosine = signals.get("image_max_cosine")
    if image_cosine is not None:
        # Cosines can round just past 1.0; a chip must never read "100.1% alike".
        capped = min(float(image_cosine), 1.0)
        out.append((f"photos {_pct(capped, 0)} alike", "good" if capped > 0.9 else "neutral"))
    days = signals.get("days_apart")
    if days is not None:
        out.append((f"{float(days):.0f} days apart", "neutral"))
    price_ratio = signals.get("abs_log_price_ratio")
    if price_ratio is not None:
        # exp(|log ratio|) - 1 turns the model's feature back into a price gap.
        gap = math.exp(abs(float(price_ratio))) - 1
        if gap < 0.1:
            out.append((f"price within {_pct(gap, 0)}", "good"))
        else:
            out.append((f"price {_pct(gap, 0)} apart", "warn"))
    shared = signals.get("shared_photo_count")
    if shared:
        out.append((f"{int(shared)} identical photos", "good"))
    return out
