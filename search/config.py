"""Constants and tunables for property search.

Spec: docs/superpowers/specs/2026-09-16-phase5-search-ranking-design.md
"""

from dataclasses import dataclass
from pathlib import Path

from listings.generate import SUB_KINDS

DATA_DIR = Path("data/search")
KINDS = ("flat", "hotel_apartment", "townhouse", "villa")
SPLITS = ("train", "tune", "report")
QUERY_KINDS = ("specified", "vague", "no_match")
SLOTS = (
    "area", "building", "bedrooms", "property_type",
    "budget_min", "budget_max", "min_size_sqm", "amenities",
)  # fmt: skip
SQFT_TO_SQM = 0.092903

# The ranker's inputs, in the order it sees them. Computed from the PARSED query only.
FEATURES = (
    "beds_diff", "beds_stated", "price_over_max", "price_under_min", "budget_stated",
    "size_ratio", "area_match", "area_stated", "type_match", "type_stated", "building_match",
    "amenity_hits", "amenity_asked", "semantic_cos", "fulltext_rank", "semantic_pos",
    "fulltext_pos", "rrf_score", "price_to_estimate", "within_interval", "flag_bait_price",
    "flag_photo_reuse", "flag_inconsistent_relist", "cluster_size", "days_since_posted",
)  # fmt: skip
TRUST_FEATURES = (
    "flag_bait_price", "flag_photo_reuse", "flag_inconsistent_relist", "cluster_size",
)  # fmt: skip

# Ground truth and generator internals. Only the modules in LABEL_READERS may name these.
SEARCH_FORBIDDEN = ("fraud_label", "dup_group_id", "control_group_id", "true_slots", "is_synthetic")
# store.py only persists the query set; its read_queries() leaves true_slots out.
LABEL_READERS = ("grade.py", "queries.py", "evaluate.py", "store.py")


def listing_kind(property_type: str, sub_type: str | None) -> str | None:
    """The corpus's four listing kinds, derived exactly as listings.generate._sub_kind does."""
    if property_type == "villa":
        return "villa"
    return SUB_KINDS.get(sub_type or "")


# The same derivation in SQL, for retrieval filters. Table alias must be `l`.
KIND_SQL = (
    "(CASE WHEN l.property_type = 'villa' THEN 'villa' "
    + " ".join(
        f"WHEN l.property_sub_type = '{sub_type}' THEN '{kind}'"
        for sub_type, kind in sorted(SUB_KINDS.items())
    )
    + " END)"
)


@dataclass(frozen=True)
class SearchConfig:
    n_queries: int = 6_000
    kind_shares: tuple[float, float, float] = (0.70, 0.20, 0.10)  # specified, vague, no_match
    split_shares: tuple[float, float, float] = (0.60, 0.20, 0.20)  # train, tune, report
    alias_probability: float = 0.4
    building_probability: float = 0.1
    budget_headroom: tuple[float, float] = (0.0, 0.30)
    size_slack: tuple[float, float] = (0.0, 0.20)
    near_margin: float = 0.10
    semantic_k: int = 200
    fulltext_k: int = 200
    candidate_k: int = 200
    rrf_k: int = 60
    ef_search: int = 400
    n_trials: int = 40
    early_stopping_rounds: int = 50
    max_rounds: int = 2_000
    n_bootstrap: int = 1_000
    ndcg_k: int = 10
    device: str = "cuda"
    ranker_name: str = "zestimator-search-ranker"
    ranker_uri: str = "models:/zestimator-search-ranker@champion"
    price_model_uri: str = "models:/zestimator-price@champion"
    experiment: str = "search-ranking"
    seed: int = 7
