"""Place names a query can mention: areas (official names and aliases), buildings, projects."""

import re
from dataclasses import dataclass

from ingestion.normalize import match_key

MIN_SINGLE_TOKEN_CHARS = 6  # one-word building names shorter than this are too generic
MAX_PHRASE_TOKENS = 6
ROMAN = ("i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x")
_PHASE = re.compile(r"\s+phase$", re.IGNORECASE)
_NUMBERED = re.compile(r"^(?P<base>.*\S)\s+(?P<number>\d{1,2}|[ivx]{1,4})$", re.IGNORECASE)


@dataclass(frozen=True)
class Place:
    kind: str  # "area" | "building" | "project"
    name: str
    area_ids: tuple[int, ...]
    # A one-word building, project or derived community name ("Lakeside") is also a common
    # word, so the parser accepts it only right after a place preposition ("at Lakeside").
    single_token: bool = False


def _numbered(name: str) -> tuple[str, int] | None:
    match = _NUMBERED.match(name)
    if match is None:
        return None
    number = match["number"].lower()
    if number.isdigit():
        value = int(number)
    elif number in ROMAN:
        value = ROMAN.index(number) + 1
    else:
        return None
    return (match["base"], value) if 1 <= value <= len(ROMAN) else None


def community_names(alias: str) -> set[str]:
    """Shorter names a DLD master-project alias implies.

    "Arabian Ranches II - Casa" -> "Arabian Ranches II", "Arabian Ranches 2", "Arabian Ranches";
    "The Springs 3" -> "The Springs 3", "The Springs III", "The Springs", "Springs" (and so on).
    The part before the first " - " is the community; a trailing number or roman numeral
    (and a "Phase" before it) is dropped to form a base name, and both spellings of the
    number are kept.
    """
    head = alias.split(" - ", 1)[0].strip()
    names = {head}
    numbered = _numbered(head)
    if numbered is not None:
        base, value = numbered
        names |= {f"{base} {value}", f"{base} {ROMAN[value - 1].upper()}"}
        names.add(_PHASE.sub("", base))  # "International City Phase 3" -> "International City"
    names |= {name[4:] for name in names if name.lower().startswith("the ")}
    return {name for name in names if name.strip()}


def derive_communities(communities, taken: set[str]) -> dict[str, tuple[str, set[int]]]:
    """match_key -> (display name, union of area ids) for names derived from the aliases.

    A derived name never replaces a key in `taken` (official names and existing aliases).
    """
    derived: dict[str, tuple[str, set[int]]] = {}
    for alias, area_id in communities:
        for name in sorted(community_names(alias)):
            key = match_key(name)
            if key and key not in taken:
                derived.setdefault(key, (name, set()))[1].add(int(area_id))
    return derived


def _guarded(key: str) -> bool:
    """A name that reads like an ordinary word ("lakeside", "the villa") only counts after a
    place preposition. A leading "the" doesn't make it specific; a digit ("luma21") does."""
    bare = key.removeprefix("the ")
    return len(bare.split()) == 1 and not any(character.isdigit() for character in bare)


@dataclass(frozen=True, eq=False)
class Lexicon:
    entries: dict[str, Place]  # match_key -> place
    max_tokens: int

    @classmethod
    def from_rows(cls, areas, aliases, buildings, projects, communities=()) -> "Lexicon":
        """areas: (area_id, name); aliases, buildings, projects: (name, area_id).

        communities: (alias, area_id) rows of DLD master-project aliases. They are aliases
        too, and they also yield derived community names (see `community_names`).

        Areas win any key they share with a building or project; buildings win over projects.
        """
        area_ids: dict[str, set[int]] = {}
        display: dict[str, str] = {}
        for area_id, name in [*areas, *((a, n) for n, a in [*aliases, *communities])]:
            key = match_key(name)
            if key:
                area_ids.setdefault(key, set()).add(int(area_id))
                display.setdefault(key, name)
        entries = {
            key: Place("area", display[key], tuple(sorted(ids))) for key, ids in area_ids.items()
        }
        for key, (name, ids) in derive_communities(communities, set(entries)).items():
            entries[key] = Place("area", name, tuple(sorted(ids)), _guarded(key))
        for kind, rows in (("building", buildings), ("project", projects)):
            grouped: dict[str, tuple[str, set[int]]] = {}
            for name, area_id in rows:
                key = match_key(name)
                if not key or key in entries:
                    continue
                if len(key.split()) < 2 and len(key) < MIN_SINGLE_TOKEN_CHARS:
                    continue
                grouped.setdefault(key, (name, set()))[1].add(int(area_id))
            for key, (name, ids) in grouped.items():
                entries[key] = Place(kind, name, tuple(sorted(ids)), _guarded(key))
        longest = max((len(key.split()) for key in entries), default=1)
        return cls(entries, min(longest, MAX_PHRASE_TOKENS))

    def match(self, tokens: list[str]) -> list[tuple[int, int, Place]]:
        """Greedy, left to right, longest phrase first: (start, end, place) token spans."""
        spans = []
        index = 0
        while index < len(tokens):
            for size in range(min(self.max_tokens, len(tokens) - index), 0, -1):
                place = self.entries.get(" ".join(tokens[index : index + size]))
                if place is not None:
                    spans.append((index, index + size, place))
                    index += size
                    break
            else:
                index += 1
        return spans


LEXICON_SQL = {
    "areas": "SELECT area_id, name_en FROM dld.areas ORDER BY area_id",
    "aliases": (
        "SELECT alias, area_id FROM dld.area_aliases WHERE source <> 'master_project' "
        "ORDER BY alias, area_id"
    ),
    "communities": (
        "SELECT alias, area_id FROM dld.area_aliases WHERE source = 'master_project' "
        "ORDER BY alias, area_id"
    ),
    "buildings": (
        "SELECT DISTINCT building_name, area_id FROM listings.listings "
        "WHERE building_name IS NOT NULL ORDER BY 1, 2"
    ),
    "projects": (
        "SELECT DISTINCT project_name, area_id FROM listings.listings "
        "WHERE project_name IS NOT NULL ORDER BY 1, 2"
    ),
}


def load_lexicon(conn) -> Lexicon:
    rows = {}
    with conn.cursor() as cur:
        for name, sql in LEXICON_SQL.items():
            cur.execute(sql)
            rows[name] = cur.fetchall()
    return Lexicon.from_rows(**rows)
