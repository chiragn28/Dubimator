"""Place names a query can mention: areas (official names and aliases), buildings, projects."""

from dataclasses import dataclass

from ingestion.normalize import match_key

MIN_SINGLE_TOKEN_CHARS = 6  # one-word building names shorter than this are too generic
MAX_PHRASE_TOKENS = 6


@dataclass(frozen=True)
class Place:
    kind: str  # "area" | "building" | "project"
    name: str
    area_ids: tuple[int, ...]


@dataclass(frozen=True)
class Lexicon:
    entries: dict[str, Place]  # match_key -> place
    max_tokens: int

    @classmethod
    def from_rows(cls, areas, aliases, buildings, projects) -> "Lexicon":
        """areas: (area_id, name); aliases, buildings, projects: (name, area_id).

        Areas win any key they share with a building or project; buildings win over projects.
        """
        area_ids: dict[str, set[int]] = {}
        display: dict[str, str] = {}
        for area_id, name in [*areas, *((a, n) for n, a in aliases)]:
            key = match_key(name)
            if key:
                area_ids.setdefault(key, set()).add(int(area_id))
                display.setdefault(key, name)
        entries = {
            key: Place("area", display[key], tuple(sorted(ids))) for key, ids in area_ids.items()
        }
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
                entries[key] = Place(kind, name, tuple(sorted(ids)))
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
    "aliases": "SELECT alias, area_id FROM dld.area_aliases ORDER BY alias, area_id",
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
