"""DB-free home price predictor: validation, location fallback, clipping, intervals, flags."""

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import polars as pl
import xgboost as xgb
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ingestion.normalize import match_key
from models.price.boosting import predict_xgb
from models.price.config import FEATURES, SIZE_BOUNDS
from models.price.evaluate import quantiles_for
from models.price.features import (
    _NO_PROJECT,
    LEVEL_KEYS,
    LocationPriors,
    MarketIndex,
    add_location_keys,
    add_model_columns,
    feature_frame,
)

KIND_TO_TYPE = {
    "apartment": ("unit", "flat"),
    "hotel_apartment": ("unit", "hotel_apartment"),
    "townhouse": ("unit", "townhouse"),
    "villa": ("villa", "villa"),
}
LEVEL_NAMES = {3: "building", 2: "project", 1: "area", 0: "city"}
CONFIDENCE = {3: "high", 2: "high", 1: "medium", 0: "low"}
TABLE_NAMES = (
    "market_index", "prior_city", "prior_area", "prior_project", "prior_building",
    "bounds_area", "bounds_segment", "size_percentiles", "areas", "aliases",
)  # fmt: skip
ROW_SCHEMA = {
    "property_type": pl.Utf8,
    "reg_type": pl.Utf8,
    "size_basis": pl.Utf8,
    "sub_kind": pl.Utf8,
    "room_kind": pl.Utf8,
    "bedrooms": pl.Float64,
    "area_sqm": pl.Float64,
    "has_parking": pl.Boolean,
    "area_id": pl.Int64,
    "project_name": pl.Utf8,
    "building_name": pl.Utf8,
    "segment": pl.Utf8,
}


class PriceInputError(ValueError):
    """A request the model must not answer; the API maps it to HTTP 422."""

    def __init__(self, field: str, message: str):
        super().__init__(f"{field}: {message}")
        self.field = field
        self.message = message


class PriceRequest(BaseModel):
    # allow_inf_nan=False: an infinite asking price would pass gt=0 and make
    # asking_vs_estimate_pct infinite, which JSON cannot carry (a 500 instead of a 422).
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    area: str | None = None
    area_id: int | None = None
    project: str | None = None
    building: str | None = None
    property_kind: Literal["apartment", "hotel_apartment", "townhouse", "villa"]
    status: Literal["ready", "off_plan"]
    size_sqm: float = Field(gt=0)
    size_basis: Literal["built_up", "plot"] = "built_up"
    bedrooms: int | None = Field(default=None, ge=0, le=8)
    is_penthouse: bool = False
    has_parking: bool | None = None
    asking_price_aed: float | None = Field(default=None, gt=0)

    @classmethod
    def parse(cls, data: dict) -> "PriceRequest":
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            error = exc.errors()[0]
            field = ".".join(str(part) for part in error["loc"]) or "request"
            raise PriceInputError(field, error["msg"]) from None


class PriceEstimate(BaseModel):
    estimate_aed: float
    range_80: tuple[float, float]
    range_95: tuple[float, float]
    price_per_sqm_aed: float
    location_level: Literal["building", "project", "area", "city"]
    confidence: Literal["high", "medium", "low"]
    market_label: Literal["below_market", "fair", "above_market"] | None
    asking_vs_estimate_pct: float | None
    flags: list[str]
    as_of: date
    model_version: str


@dataclass(frozen=True)
class ModelBundle:
    booster: object  # xgb.Booster; tests may pass any object with .predict(dmatrix)
    index: MarketIndex
    priors: LocationPriors
    categories: dict[str, list[str]]
    bounds_area: pl.DataFrame
    bounds_segment: pl.DataFrame
    size_percentiles: pl.DataFrame
    areas: pl.DataFrame
    aliases: pl.DataFrame
    conformal: dict[str, dict[str, float]]
    supported_segments: tuple[str, ...]
    data_end: date
    metadata: dict


def save_bundle(bundle: ModelBundle, path: Path) -> None:
    tables = path / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    bundle.booster.save_model(str(path / "booster.json"))
    frames = {
        "market_index": bundle.index.table,
        **{f"prior_{level}": table for level, table in bundle.priors.stats.items()},
        "bounds_area": bundle.bounds_area,
        "bounds_segment": bundle.bounds_segment,
        "size_percentiles": bundle.size_percentiles,
        "areas": bundle.areas,
        "aliases": bundle.aliases,
    }
    for name, frame in frames.items():
        frame.write_parquet(tables / f"{name}.parquet")
    metadata = {
        "features": list(FEATURES),
        "categories": bundle.categories,
        "conformal": bundle.conformal,
        "supported_segments": list(bundle.supported_segments),
        "data_end": bundle.data_end.isoformat(),
        "shrink_k": bundle.priors.shrink_k,
        "min_level_n": bundle.priors.min_level_n,
        "metadata": bundle.metadata,
    }
    (path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )


def load_bundle(path: Path) -> ModelBundle:
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    saved_features = meta.get("features")
    if saved_features != list(FEATURES):
        raise ValueError(
            f"model bundle at {path} was trained on features {saved_features}, but this code "
            f"builds {list(FEATURES)}: retrain the model or use the code version it was built with"
        )
    tables = {name: pl.read_parquet(path / "tables" / f"{name}.parquet") for name in TABLE_NAMES}
    booster = xgb.Booster()
    booster.load_model(str(path / "booster.json"))
    booster.set_param({"device": "cpu"})  # serving never assumes a GPU
    priors = LocationPriors(
        {level: tables[f"prior_{level}"] for level in LEVEL_KEYS},
        meta["shrink_k"],
        meta["min_level_n"],
    )
    return ModelBundle(
        booster=booster,
        index=MarketIndex(tables["market_index"]),
        priors=priors,
        categories=meta["categories"],
        bounds_area=tables["bounds_area"],
        bounds_segment=tables["bounds_segment"],
        size_percentiles=tables["size_percentiles"],
        areas=tables["areas"],
        aliases=tables["aliases"],
        conformal=meta["conformal"],
        supported_segments=tuple(meta["supported_segments"]),
        data_end=date.fromisoformat(meta["data_end"]),
        metadata=meta["metadata"],
    )


def _room_fields(request: PriceRequest) -> tuple[str, float | None]:
    if request.is_penthouse:
        return "penthouse", None
    if request.bedrooms is None:
        return "unknown", None
    if request.bedrooms == 0:
        return "studio", 0.0
    return "bedrooms", float(request.bedrooms)


class PricePredictor:
    def __init__(self, bundle: ModelBundle):
        self.bundle = bundle
        self._area_ids = set(bundle.areas["area_id"].to_list())

    @classmethod
    def from_dir(cls, path: Path) -> "PricePredictor":
        return cls(load_bundle(Path(path)))

    @property
    def model_version(self) -> str:
        return str(self.bundle.metadata.get("model_version", "unregistered"))

    def _validate(self, request: PriceRequest) -> tuple[str, str, str]:
        if (request.area is None) == (request.area_id is None):
            raise PriceInputError("area", "give exactly one of area or area_id")
        property_type, sub_kind = KIND_TO_TYPE[request.property_kind]
        if request.size_basis == "plot" and property_type != "villa":
            raise PriceInputError("size_basis", "'plot' is only valid for a villa")
        if request.is_penthouse and request.property_kind not in ("apartment", "hotel_apartment"):
            raise PriceInputError("is_penthouse", "only valid for an apartment or hotel apartment")
        low, high = SIZE_BOUNDS[(property_type, request.size_basis)]
        if not low <= request.size_sqm <= high:
            raise PriceInputError(
                "size_sqm",
                f"must be between {low:g} and {high:g} m² for a {request.property_kind} "
                f"({request.size_basis})",
            )
        segment = f"{property_type}_{request.status}_{request.size_basis}"
        if segment not in self.bundle.supported_segments:
            raise PriceInputError(
                "status",
                f"{request.property_kind} + {request.status} + {request.size_basis} "
                "is not supported by this model",
            )
        return property_type, sub_kind, segment

    def _resolve_area(self, request: PriceRequest) -> int:
        if request.area_id is not None:
            if request.area_id not in self._area_ids:
                raise PriceInputError("area_id", f"unknown area_id {request.area_id}")
            return request.area_id
        key = match_key(request.area)
        aliases = self.bundle.aliases
        ids = (
            sorted(set(aliases.filter(pl.col("alias_key") == key)["area_id"].to_list()))
            if key
            else []
        )
        if not ids:
            raise PriceInputError("area", f"unknown area {request.area!r}")
        if len(ids) > 1:
            raise PriceInputError(
                "area", f"area {request.area!r} is ambiguous: use area_id, one of {ids}"
            )
        return ids[0]

    def _y_bounds(self, property_type: str, size_basis: str, area_id: int) -> tuple[float, float]:
        same_kind = (pl.col("property_type") == property_type) & (
            pl.col("size_basis") == size_basis
        )
        table = self.bundle.bounds_area.filter(same_kind & (pl.col("area_id") == area_id))
        if table.height == 0:
            table = self.bundle.bounds_segment.filter(same_kind)
        if table.height != 1:
            raise RuntimeError(f"no plausibility bounds for {property_type}/{size_basis}")
        return float(table["lo"][0]), float(table["hi"][0])

    def _infer_project_key(
        self, property_type: str, area_id: int, building: str
    ) -> tuple[str | None, bool]:
        """Project scope for a building given without its project: (project_key, ambiguous).

        Building priors are keyed within their project, so a building-only request would only
        match buildings sold with no project. If the building name sits under exactly one
        project scope in this area, use it (the no-project sentinel means "no project"). If it
        sits under several, don't guess: (None, True).
        """
        scopes = (
            self.bundle.priors.stats["building"]
            .filter(
                (pl.col("property_type") == property_type)
                & (pl.col("area_id") == area_id)
                & (pl.col("building_key") == match_key(building))
            )["__p_building_project"]  # features.py's internal project-scope column
            .unique()
            .to_list()
        )
        if len(scopes) > 1:
            return None, True
        if len(scopes) == 1 and scopes[0] != _NO_PROJECT:
            return scopes[0], False
        return None, False

    def _unusual_size(self, segment: str, size: float) -> bool:
        row = self.bundle.size_percentiles.filter(pl.col("segment") == segment)
        if row.height == 0:
            return False
        return not float(row["p005"][0]) <= size <= float(row["p995"][0])

    def predict_one(self, request: PriceRequest | dict) -> PriceEstimate:
        if not isinstance(request, PriceRequest):
            request = PriceRequest.parse(request)
        property_type, sub_kind, segment = self._validate(request)
        area_id = self._resolve_area(request)
        index_value = self.bundle.index.value_at(segment, self.bundle.data_end)
        room_kind, bedrooms = _room_fields(request)
        row = pl.DataFrame(
            {
                "property_type": [property_type],
                "reg_type": [request.status],
                "size_basis": [request.size_basis],
                "sub_kind": [sub_kind],
                "room_kind": [room_kind],
                "bedrooms": [bedrooms],
                "area_sqm": [float(request.size_sqm)],
                "has_parking": [request.has_parking],
                "area_id": [area_id],
                "project_name": [request.project],
                "building_name": [request.building],
                "segment": [segment],
            },
            schema=ROW_SCHEMA,
        )
        row = add_location_keys(add_model_columns(row))
        ambiguous_building = False
        if request.building is not None and request.project is None:
            project_key, ambiguous_building = self._infer_project_key(
                property_type, area_id, request.building
            )
            if project_key is not None:
                row = row.with_columns(pl.lit(project_key, dtype=pl.Utf8).alias("project_key"))
        row = self.bundle.priors.transform(
            row.with_columns(pl.lit(index_value).alias("market_index"))
        )
        y_hat = float(
            predict_xgb(self.bundle.booster, feature_frame(row, self.bundle.categories))[0]
        )

        flags = []
        low, high = self._y_bounds(property_type, request.size_basis, area_id)
        if not low <= y_hat <= high:
            y_hat = min(max(y_hat, low), high)
            flags.append("implausible_clipped")
        unseen_project = request.project is not None and row["n_project"][0] == 0.0
        unseen_building = request.building is not None and row["n_building"][0] == 0.0
        if unseen_project or unseen_building or ambiguous_building:
            flags.append("location_fallback")
        if self._unusual_size(segment, request.size_sqm):
            flags.append("unusual_size")

        size = float(request.size_sqm)
        centre = y_hat + index_value
        q = quantiles_for(self.bundle.conformal, segment)
        estimate = math.exp(centre) * size
        range_80 = (math.exp(centre - q["q80"]) * size, math.exp(centre + q["q80"]) * size)
        range_95 = (math.exp(centre - q["q95"]) * size, math.exp(centre + q["q95"]) * size)
        label, pct = None, None
        if request.asking_price_aed is not None:
            asking = request.asking_price_aed
            if asking < range_80[0]:
                label = "below_market"
            elif asking > range_80[1]:
                label = "above_market"
            else:
                label = "fair"
            pct = round((asking / estimate - 1.0) * 100.0, 1)
        level = int(row["loc_level"][0])
        return PriceEstimate(
            estimate_aed=estimate,
            range_80=range_80,
            range_95=range_95,
            price_per_sqm_aed=estimate / size,
            location_level=LEVEL_NAMES[level],
            confidence=CONFIDENCE[level],
            market_label=label,
            asking_vs_estimate_pct=pct,
            flags=flags,
            as_of=self.bundle.data_end,
            model_version=self.model_version,
        )
