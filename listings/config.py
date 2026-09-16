"""Constants and tunables for the listings corpus and duplicate detection.

Spec: docs/superpowers/specs/2026-09-16-phase4-duplicate-fraud-design.md
"""

from dataclasses import dataclass

PHOTO_DATASET_URL = "https://codeload.github.com/emanhamed/Houses-dataset/zip/refs/heads/master"
PHOTO_DATASET_DIR = "Houses-dataset-master/Houses Dataset"
PHOTO_ROOMS = ("bathroom", "bedroom", "frontal", "kitchen")
HOME_UNIT_SUB_TYPES = ("Flat", "Hotel Apartment", "Stacked Townhouses")

# Pair features, in the order the model sees them.
PAIR_FEATURES = (
    "text_cosine", "image_max_cosine", "image_mean_cosine", "shared_photo_count",
    "abs_log_price_ratio", "abs_log_size_ratio", "same_area", "same_building",
    "same_project", "bedrooms_equal", "days_apart", "same_agent",
)  # fmt: skip

# Ground truth. Only listings/truth.py may read these: they are training targets and
# evaluation answers, never model inputs.
LABEL_COLUMNS = ("dup_group_id", "control_group_id", "fraud_label")


@dataclass(frozen=True)
class CorpusConfig:
    n_listings: int = 20_000
    n_base: int = 17_000
    n_exact_repost: int = 1_200
    n_reworded: int = 900
    n_edited_photo: int = 900
    n_bait_price: int = 600
    n_price_shifted_reposts: int = 300
    n_stock_sets: int = 40
    stock_share: float = 0.25
    stock_min_areas: int = 5
    min_building_sales: int = 4
    n_from_busy_buildings: int = 1_500
    asking_factor: tuple[float, float] = (1.00, 1.08)
    bait_factor: tuple[float, float] = (0.40, 0.65)
    price_shift: tuple[float, float] = (0.15, 0.30)
    repost_days: tuple[int, int] = (1, 30)
    n_agents: int = 400
    posted_from: str = "2023-01-01"
    posted_to: str = "2023-06-30"
    sales_from: str = "2021-01-01"
    crop_fraction: float = 0.85
    resize_fraction: float = 0.70
    jpeg_quality: int = 60
    seed: int = 42


@dataclass(frozen=True)
class DetectConfig:
    text_top_k: int = 20
    photo_top_k: int = 20  # nearest listings by average image vector
    ef_search: int = 100
    train_share: float = 0.60
    threshold_share: float = 0.20
    target_precision: float = 0.98
    baseline_image_cosine: float = 0.95
    max_photo_fanout: int = 50
    photo_reuse_min_areas: int = 5
    relist_price_spread: float = 0.20
    bait_margin: float = 0.10
    price_model_uri: str = "models:/zestimator-price@champion"
    experiment: str = "listing-dedup"
    seed: int = 42
