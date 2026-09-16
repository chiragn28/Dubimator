"""Query text -> structured slots. Pure: the place lexicon is passed in.

Each pass blanks what it consumed with BLANK, so later passes cannot reuse it, and the
unrecognised-place scan can tell that a preposition was followed by an already-matched place.
"""

import re
from dataclasses import asdict, dataclass

from listings.text import AMENITIES
from search.config import SQFT_TO_SQM
from search.lexicon import Lexicon

BLANK = "\x00"
MIN_BARE_AMOUNT = 10_000.0  # a number with no suffix and no "AED" is money only from here up
SCALE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "mn": 1e6, "million": 1e6}
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
AMENITY_SYNONYMS = {
    **{amenity: amenity for amenity in AMENITIES},
    "pool": "shared pool",
    "swimming pool": "shared pool",
    "parking": "covered parking",
    "gym": "gym access",
    "security": "24/7 security",
    "fitted kitchen": "fully fitted kitchen",
    "wardrobes": "built-in wardrobes",
    "maids room": "maid's room",
    "maid room": "maid's room",
    "play area": "children's play area",
    "kids play area": "children's play area",
    "central ac": "central air conditioning",
}
STOPWORDS = frozenset(
    [
        "a", "an", "the", "i", "we", "me", "my", "for", "in", "at", "near", "around", "with",
        "and", "or", "vs", "looking", "want", "need", "to", "buy", "rent", "show", "find",
        "searching", "search", "please", "property", "properties", "home", "homes", "budget",
        "of", "al", "under", "below", "max", "up", "over", "above", "from", "between", "less",
        "more", "than", "least", "most", "min",
    ]
)  # fmt: skip
PLACE_PREPOSITIONS = frozenset({"in", "at", "near", "around"})
NEARBY_WORDS = frozenset({"metro", "beach", "school", "schools", "mall", "airport", "park"})
PLACE_STOP = (STOPWORDS - {"al"}) | NEARBY_WORDS

_SIZE = re.compile(
    r"(?:(?:over|above|at least|min(?:imum)?|more than|from)\s+)?"
    r"(?P<n>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*\+?\s*"
    r"(?P<u>sq\.?\s*f(?:ee)?t|sqft|square\s+f(?:ee|oo)t"
    r"|sq\.?\s*m(?:eters?|etres?)?|sqm|m2|m²|square\s+met(?:er|re)s?)(?![a-z0-9])"
)
_BEDS = re.compile(
    r"(?<![a-z0-9])(?:(?P<studio>studios?)"
    r"|(?P<n>\d)\s*[-+]?\s*(?:br|bhk|beds?|bedrooms?)"
    r"|(?P<w>one|two|three|four|five|six|seven)[\s-]*(?:br|bhk|beds?|bedrooms?))(?![a-z])"
)
_MONEY = (
    r"(?P<aedX>aed\s*)?(?<![0-9.,])(?P<nX>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
    r"\s*(?P<sX>million|thousand|mn|m|k)?(?![a-z0-9])(?P<postX>\s*aed(?![a-z]))?"
)
M1, M2 = _MONEY.replace("X", "1"), _MONEY.replace("X", "2")
_RANGE = re.compile(rf"(?:between\s+|from\s+)?{M1}\s*(?:-|–|—|to|and)\s*{M2}")
_MAX = re.compile(
    r"(?:under|below|max(?:imum)?|up\s+to|less\s+than|within|budget(?:\s+of)?|at\s+most"
    rf"|no\s+more\s+than)\s*:?\s*{M1}"
)
_MIN = re.compile(
    rf"(?:from|over|above|at\s+least|more\s+than|min(?:imum)?|starting\s+(?:at|from))\s*:?\s*{M1}"
)
_BARE = re.compile(M1)
_TYPES = (
    (re.compile(r"(?<![a-z])(?:hotel|serviced)\s+apartments?(?![a-z])"), "hotel_apartment"),
    (re.compile(r"(?<![a-z])(?:town\s*houses?|townhomes?)(?![a-z])"), "townhouse"),
    (re.compile(r"(?<![a-z])villas?(?![a-z])"), "villa"),
    (re.compile(r"(?<![a-z])(?:apartments?|flats?)(?![a-z])"), "flat"),
)
_UNLISTED = re.compile(r"(?<![a-z])(?P<w>plots?|land)(?![a-z])")
_AMENITY_PATTERNS = tuple(
    (re.compile(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])"), canonical)
    for phrase, canonical in sorted(AMENITY_SYNONYMS.items(), key=lambda item: -len(item[0]))
)
_TOKEN = re.compile(r"[a-z0-9]+")
_WORD = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class ParsedQuery:
    area_ids: tuple[int, ...] = ()
    area_name: str | None = None
    building: str | None = None  # a building or project name, as the lexicon spells it
    # the named building's areas: context for display only, never a retrieval filter
    building_area_ids: tuple[int, ...] = ()
    bedrooms: int | None = None
    property_type: str | None = None
    budget_min: float | None = None
    budget_max: float | None = None
    min_size_sqm: float | None = None
    amenities: tuple[str, ...] = ()
    free_text: str = ""
    unrecognised: tuple[tuple[str, str], ...] = ()  # ("place" | "type", text)
    errors: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not (
            self.area_ids
            or self.building
            or self.bedrooms is not None
            or self.property_type
            or self.budget_min is not None
            or self.budget_max is not None
            or self.min_size_sqm is not None
            or self.amenities
            or self.free_text
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["area_ids"] = list(self.area_ids)
        data["building_area_ids"] = list(self.building_area_ids)
        data["amenities"] = list(self.amenities)
        data["unrecognised"] = [list(item) for item in self.unrecognised]
        data["errors"] = list(self.errors)
        return data


def _blank(text: str, start: int, end: int) -> str:
    return text[:start] + BLANK * (end - start) + text[end:]


def _size(text: str) -> tuple[str, float | None]:
    value = None
    for match in _SIZE.finditer(text):
        amount = float(match["n"].replace(",", ""))
        if "f" in match["u"]:
            amount *= SQFT_TO_SQM
        if value is None:
            value = round(amount, 2)
        text = _blank(text, *match.span())
    return text, value


def _bedrooms(text: str) -> tuple[str, int | None]:
    value = None
    for match in _BEDS.finditer(text):
        if match["studio"]:
            count = 0
        elif match["n"]:
            count = int(match["n"])
        else:
            count = NUMBER_WORDS[match["w"]]
        if value is None:
            value = count
        text = _blank(text, *match.span())
    return text, value


def _amount(match, index: str, inherit: tuple[str | None, bool] | None = None) -> float | None:
    suffix = match[f"s{index}"]
    has_aed = bool(match[f"aed{index}"] or match[f"post{index}"])
    if inherit is not None:
        suffix = suffix or inherit[0]
        has_aed = has_aed or inherit[1]
    value = float(match[f"n{index}"].replace(",", "")) * SCALE.get(suffix or "", 1.0)
    return value if (suffix or has_aed or value >= MIN_BARE_AMOUNT) else None


def _range_first(match, second: float | None) -> float | None:
    """The low bound of an "X-Y" range.

    A first number that already stands as a complete amount (its own suffix, an AED marker, or
    a value large enough on its own, e.g. "970,000") is used as is. Otherwise it is ambiguous
    (e.g. the "1" in "1-1.5M" or the "900" in "900-1.2M") and inherits the second bound's scale
    -- unless that reading would exceed the second bound, in which case a x1,000 reading is
    used instead when that fits under the second bound (so "900-1.2M" reads as 900k-1.2M).
    """
    bare = _amount(match, "1")
    if bare is not None:
        return bare
    inherited = _amount(match, "1", inherit=(match["s2"], bool(match["aed2"] or match["post2"])))
    if inherited is not None and second is not None and inherited > second:
        thousands = float(match["n1"].replace(",", "")) * 1_000.0
        if thousands <= second:
            return thousands
    return inherited


def _budget(text: str) -> tuple[str, float | None, float | None]:
    low = high = None
    for match in _RANGE.finditer(text):
        second = _amount(match, "2")
        first = _range_first(match, second)
        if first is not None and second is not None:
            low, high = first, second
            text = _blank(text, *match.span())
            break
    for pattern, which in ((_MAX, "max"), (_MIN, "min")):
        if (high if which == "max" else low) is not None:
            continue
        for match in pattern.finditer(text):
            amount = _amount(match, "1")
            if amount is not None:
                if which == "max":
                    high = amount
                else:
                    low = amount
                text = _blank(text, *match.span())
                break
    if low is None and high is None:
        for match in _BARE.finditer(text):
            amount = _amount(match, "1")
            if amount is not None:
                high = amount
                text = _blank(text, *match.span())
                break
    return text, low, high


def _after_preposition(words: list[str], start: int) -> bool:
    """True when words[start] follows "in", "at", "near" or "around" (optionally + "the")."""
    index = start - 1
    if index >= 0 and words[index] == "the":
        index -= 1
    return index >= 0 and words[index] in PLACE_PREPOSITIONS


def _places(text: str, lexicon: Lexicon) -> tuple[str, list]:
    tokens = [(m.group(), m.start(), m.end()) for m in _TOKEN.finditer(text) if m.group() != "al"]
    words = [token for token, _, _ in tokens]
    found = []
    for start, end, place in lexicon.match(words):
        if place.single_token and not _after_preposition(words, start):
            continue  # "villa with lakeside view": a common word, not the building "Lakeside"
        text = _blank(text, tokens[start][1], tokens[end - 1][2])
        found.append(place)
    return text, found


def _types(text: str) -> tuple[str, str | None, list[str]]:
    matches = []
    for pattern, kind in _TYPES:
        for match in pattern.finditer(text):
            matches.append((match.start(), kind))
            text = _blank(text, *match.span())
    unlisted = []
    for match in _UNLISTED.finditer(text):
        unlisted.append("plot" if match["w"].startswith("plot") else "land")
        text = _blank(text, *match.span())
    kind = min(matches)[1] if matches else None
    return text, kind, unlisted


def _amenities(text: str) -> tuple[str, tuple[str, ...]]:
    found = []
    for pattern, canonical in _AMENITY_PATTERNS:
        for match in pattern.finditer(text):
            found.append((match.start(), canonical))
            text = _blank(text, *match.span())
    ordered: list[str] = []
    for _, canonical in sorted(found):
        if canonical not in ordered:
            ordered.append(canonical)
    return text, tuple(ordered)


def _unrecognised_places(text: str) -> tuple[str, list[str]]:
    words = [(m.group(), m.start(), m.end()) for m in _WORD.finditer(text)]
    found = []
    index = 0
    while index < len(words):
        word, _, end = words[index]
        if word not in PLACE_PREPOSITIONS:
            index += 1
            continue
        phrase = []
        cursor, previous_end = index + 1, end
        if cursor < len(words) and words[cursor][0] == "the":
            previous_end = words[cursor][2]  # "in the Springs": the place is "springs"
            cursor += 1
        while cursor < len(words) and len(phrase) < 3:
            token, start, stop = words[cursor]
            if BLANK in text[previous_end:start] or token in PLACE_STOP or token.isdigit():
                break
            phrase.append((token, start, stop))
            previous_end = stop
            cursor += 1
        if phrase:
            found.append(" ".join(token for token, _, _ in phrase))
            text = _blank(text, phrase[0][1], phrase[-1][2])
        index = cursor
    return text, found


def _free_text(text: str) -> str:
    words = (word.strip("'") for word in _WORD.findall(text))
    return " ".join(
        word for word in words if len(word) > 1 and word not in STOPWORDS and not word.isdigit()
    )


def parse(text: str, lexicon: Lexicon) -> ParsedQuery:
    working = f" {(text or '').lower().replace('’', chr(39))} "
    working, min_size = _size(working)
    working, budget_min, budget_max = _budget(working)
    errors = []
    if budget_min is not None and budget_max is not None and budget_min > budget_max:
        errors.append("budget_min_exceeds_max")
        budget_min = budget_max = None
    working, places = _places(working, lexicon)
    working, bedrooms = _bedrooms(working)
    working, property_type, unlisted = _types(working)
    working, amenities = _amenities(working)
    working, unknown = _unrecognised_places(working)
    area = next((place for place in places if place.kind == "area"), None)
    named = next((place for place in places if place.kind != "area"), None)
    return ParsedQuery(
        area_ids=area.area_ids if area else (),
        area_name=area.name if area else None,
        building=named.name if named else None,
        building_area_ids=named.area_ids if named else (),
        bedrooms=bedrooms,
        property_type=property_type,
        budget_min=budget_min,
        budget_max=budget_max,
        min_size_sqm=min_size,
        amenities=amenities,
        free_text=_free_text(working),
        unrecognised=tuple(
            [("type", word) for word in unlisted] + [("place", phrase) for phrase in unknown]
        ),
        errors=tuple(errors),
    )
