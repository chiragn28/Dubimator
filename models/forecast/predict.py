"""Forecasts: the current estimate x exp(predicted growth), with ranges, labels and drivers.

Spec: "Prediction" in docs/superpowers/specs/2026-09-16-phase6-price-forecasting-design.md.
Features for a request come from a snapshot computed once, at T = data_end + 1 day, by the
same code that builds training features (query rows carry a null ppsm and never count).
"""

import math
from datetime import date, timedelta

import numpy as np
import polars as pl

from ingestion.config import DbSettings
from ingestion.normalize import match_key
from models.forecast.config import (
    CATEGORICAL,
    FEATURES,
    HORIZONS,
    INFRA_TYPES,
    PLOT_VILLA,
    ForecastConfig,
)
from models.forecast.features import PROPERTY_FEATURES, add_core_features
from models.forecast.infra import add_infra_features, load_projects
from models.forecast.intervals import confidence, low_message
from models.forecast.model import ForecastModel, latest_gates, load_champion
from models.forecast.rows import excluded_summary, load_rows
from models.forecast.targets import add_base, add_keys
from models.price.predictor import KIND_TO_TYPE, PriceInputError, PriceRequest

SNAPSHOT_COLUMNS = (
    "base_level", "base_ppsm", "days_since_area_sale",
    *(name for name in FEATURES if name not in PROPERTY_FEATURES and name not in CATEGORICAL),
)  # fmt: skip
DRIVER_HORIZONS = ("1y", "3m")
LAG_TEXT = {"3m": "3 months", "12m": "12 months", "36m": "36 months"}
TYPE_TEXT = {
    "metro_rail": "metro or rail",
    "mall": "mall",
    "school": "school",
    "park": "park",
    "mixed_use": "mixed-use",
    "airport": "airport",
}
LABELS = {
    "ln_base_ppsm": "Recent price level",
    "base_level_building": "Where recent prices come from",
    "base_n": "Comparable sales in the last 3 months",
    "area_share_12m": "The area's share of Dubai sales",
    "days_since_building_sale": "Days since the building's last sale",
    "building_sales_12m": "Sales in this building in the last 12 months",
    "area_sales_12m": "Sales in this area in the last 12 months",
    "off_plan": "Registration status",
    "log_area_sqm": "Size",
    "bedrooms": "Bedrooms",
    "building_age_proxy_years": "Years since the building's first sale",
    "infra_months_to_next": "Months until the next nearby project completes",
    "infra_completed_24m": "Nearby project completions",
    "market_kind": "Property kind",
    "area_code": "Location",
    "project_code": "Project",
}


def _moved(value: float) -> str:
    change = math.expm1(value)
    return f"{'rose' if change >= 0 else 'fell'} {abs(change):.0%}"


def describe(feature: str, value) -> str:
    """A plain sentence about one feature's value (never about its effect)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        label = LABELS.get(feature, feature.replace("_", " ").capitalize())
        return f"{label}: unknown"
    for level, subject in (("area", "Area prices"), ("city", "Dubai-wide prices for this kind")):
        for lag, text in LAG_TEXT.items():
            if feature == f"{level}_mom_{lag}":
                return f"{subject} {_moved(value)} over the last {text}"
    for kind in INFRA_TYPES:
        if feature == f"infra_active_{kind}":
            return f"{value:.0f} {TYPE_TEXT[kind]} projects under way in the area"
        if feature == f"infra_mix_{kind}":
            return f"{value:.0%} of the area's active projects are {TYPE_TEXT[kind]}"
    templates = {
        "ln_base_ppsm": lambda v: f"Recent price level is AED {math.exp(v):,.0f}/m²",
        "base_level_building": lambda v: (
            "Recent prices come from the same building"
            if v >= 0.5
            else "Recent prices come from the wider area"
        ),
        "base_n": lambda v: f"{v:.0f} comparable sales in the last 3 months",
        "area_share_12m": lambda v: f"This area had {v:.1%} of Dubai sales of this kind last year",
        "days_since_building_sale": lambda v: f"The building's last sale was {v:.0f} days ago",
        "building_sales_12m": lambda v: f"{v:.0f} sales in this building in the last 12 months",
        "area_sales_12m": lambda v: f"{v:.0f} sales in this area in the last 12 months",
        "off_plan": lambda v: "Off-plan" if v >= 0.5 else "Ready (resale)",
        "log_area_sqm": lambda v: f"Size {math.exp(v):,.0f} m²",
        "bedrooms": lambda v: "Studio" if v == 0 else f"{v:.0f} bedrooms",
        "building_age_proxy_years": lambda v: f"Building first sold {v:.1f} years ago",
        "infra_months_to_next": lambda v: f"Next nearby project completes in {v:.0f} months",
        "infra_completed_24m": lambda v: (
            "A nearby project opened in the last 2 years"
            if v >= 0.5
            else "No nearby project opened in the last 2 years"
        ),
    }
    if feature in templates:
        return templates[feature](float(value))
    return f"{LABELS.get(feature, feature)}: {value}"


def driver_text(feature: str, value, contribution: float, horizon: str) -> str:
    effect = math.expm1(contribution)
    sign = "+" if effect >= 0 else "-"
    return f"{describe(feature, value)} ({sign}{abs(effect):.1%} to the {horizon} forecast)"


def query_rows(rows: pl.DataFrame, day: date) -> pl.DataFrame:
    """One ppsm-less row per building and per area x market kind, dated `day`."""
    kinds = ["area_id", "sub_kind", "market_kind"]
    buildings = (
        rows.filter(pl.col("building_name").is_not_null()).select(*kinds, "building_name").unique()
    )
    areas = (
        rows.select(kinds).unique().with_columns(pl.lit(None, dtype=pl.Utf8).alias("building_name"))
    )
    queries = pl.concat([buildings, areas]).sort(*kinds, "building_name", nulls_last=True)
    start = int(rows["row_id"].max()) + 1 if rows.height else 0
    index = pl.int_range(pl.len(), dtype=pl.Int64)
    return queries.with_columns(
        pl.lit(day).alias("instance_date"),
        pl.lit(None, dtype=pl.Float64).alias("ppsm"),
        (index + start).alias("row_id"),
        pl.format("query-{}", index).alias("transaction_id"),
    )


def build_snapshot(rows: pl.DataFrame, data_end: date, projects: pl.DataFrame) -> pl.DataFrame:
    day = data_end + timedelta(days=1)
    frame = pl.concat([rows, query_rows(rows, day)], how="diagonal_relaxed")
    frame = add_infra_features(add_core_features(add_base(add_keys(frame))), projects)
    return (
        frame.filter(pl.col("ppsm").is_null())
        .select("area_id", "market_kind", "building_key", *SNAPSHOT_COLUMNS)
        .unique(["area_id", "market_kind", "building_key"], keep="first", maintain_order=True)
    )


def resolve_area(request: PriceRequest, areas: pl.DataFrame, aliases: pl.DataFrame) -> int:
    if (request.area is None) == (request.area_id is None):
        raise PriceInputError("area", "give exactly one of area or area_id")
    if request.area_id is not None:
        if request.area_id not in set(areas["area_id"].to_list()):
            raise PriceInputError("area_id", f"unknown area_id {request.area_id}")
        return request.area_id
    key = match_key(request.area)
    ids = (
        sorted(set(aliases.filter(pl.col("alias_key") == key)["area_id"].to_list())) if key else []
    )
    if not ids:
        raise PriceInputError("area", f"unknown area {request.area!r}")
    if len(ids) > 1:
        raise PriceInputError(
            "area", f"area {request.area!r} is ambiguous: use area_id, one of {ids}"
        )
    return ids[0]


class Forecaster:
    def __init__(self, snapshot, data_end, areas, aliases, models, gates, price, excluded, config):
        self.snapshot = snapshot
        self.data_end = data_end
        self.areas = areas
        self.aliases = aliases
        self.gates = gates
        self.blocked = {
            name: reason
            for name, (model, _) in models.items()
            if model is not None and (reason := self._block_reason(name, model)) is not None
        }
        self.models = {
            name: (None, None) if name in self.blocked else entry for name, entry in models.items()
        }
        self.price = price
        self.excluded = excluded
        self.config = config

    def _block_reason(self, name: str, model: ForecastModel) -> str | None:
        """Why a loaded champion must not be served: a failed latest gate or stale data."""
        gate = self.gates.get(name, {})
        if gate.get("status") in ("failed", "insufficient_data"):
            return gate.get("reason")
        model_end = model.metadata.get("data_end")
        data_end = self.data_end.isoformat()
        if model_end != data_end:
            return (
                f"the {name} model was trained on data ending {model_end}; "
                f"retrain it on data ending {data_end}"
            )
        return None

    @classmethod
    def build(
        cls, rows, data_end, projects, areas, aliases, models, gates, price, excluded, config
    ):
        snapshot = build_snapshot(rows, data_end, projects)
        return cls(snapshot, data_end, areas, aliases, models, gates, price, excluded, config)

    @classmethod
    def from_registry(cls, settings: DbSettings, config: ForecastConfig) -> "Forecaster":
        from listings.fraud import load_price_predictor

        price = load_price_predictor(config.price_model_uri)
        if price is None:
            raise RuntimeError(
                f"the price model {config.price_model_uri} is unavailable; forecasts need the "
                "current estimate"
            )
        rows, quality, data_end, areas, aliases = load_rows(settings, config)
        models = {name: load_champion(f"{config.model_prefix}-{name}") for name in HORIZONS}
        return cls.build(
            rows, data_end, load_projects(), areas, aliases, models,
            latest_gates(config.experiment), price, quality.excluded, config,
        )  # fmt: skip

    def forecast(self, property: dict, as_of: date | None = None) -> dict:
        if as_of is not None and as_of != self.data_end:
            raise PriceInputError(
                "as_of", f"forecasts are only available as of the data end, {self.data_end}"
            )
        request = PriceRequest.parse({k: v for k, v in property.items() if k != "property_id"})
        estimate = self.price.predict_one(request)
        area_id = resolve_area(request, self.areas, self.aliases)
        _, sub_kind = KIND_TO_TYPE[request.property_kind]
        plot = sub_kind == "villa" and request.size_basis == "plot"
        features = self._features(request, area_id, PLOT_VILLA if plot else sub_kind)
        base = features.row(0, named=True)
        building = base["base_level"] == "building"
        context = {
            "estimate": estimate.estimate_aed,
            "segment": f"{request.status}_{'villa' if sub_kind == 'villa' else 'unit'}",
            "base": base,
            "days": base["days_since_building_sale" if building else "days_since_area_sale"],
            "area_id": area_id,
        }
        out = {
            "property_id": property.get("property_id"),
            "as_of": self.data_end.isoformat(),
            "current_estimate_aed": round(estimate.estimate_aed),
            "current_range_80": [round(value) for value in estimate.range_80],
        }
        for name in HORIZONS:
            out[f"forecast_{name}"] = self._horizon(name, features, context)
        out["key_drivers"] = self._drivers(features)
        out["exclusions_applied"] = self._exclusions(area_id, sub_kind)
        out["model_versions"] = {
            "price": estimate.model_version,
            **{
                f"forecast_{name}": version
                for name, (model, version) in self.models.items()
                if model is not None
            },
        }
        return out

    def _features(self, request: PriceRequest, area_id: int, market_kind: str) -> pl.DataFrame:
        same = (pl.col("area_id") == area_id) & (pl.col("market_kind") == market_kind)
        key = match_key(request.building) if request.building else None
        found = self.snapshot.filter(same & (pl.col("building_key") == f"{area_id}|{key}"))
        if key is None or found.height == 0:
            found = self.snapshot.filter(same & pl.col("building_key").is_null())
        if found.height == 0:
            raise PriceInputError(
                "property_kind", f"no sales history for a {request.property_kind} in this area"
            )
        if found["base_level"][0] is None:
            raise PriceInputError(
                "area", "not enough sales in the last 3 months to forecast this area and kind"
            )
        return found.head(1).with_columns(
            pl.lit(float(request.status == "off_plan")).alias("off_plan"),
            pl.lit(math.log(request.size_sqm)).alias("log_area_sqm"),
            pl.lit(request.bedrooms, dtype=pl.Float64).alias("bedrooms"),
            pl.lit(market_kind).alias("market_kind"),
            pl.lit(str(area_id)).alias("area_code"),
            pl.lit(request.project, dtype=pl.Utf8).alias("project_code"),
        )

    def _model(self, name: str) -> ForecastModel | None:
        return self.models.get(name, (None, None))[0]

    def _horizon(self, name: str, features: pl.DataFrame, context: dict) -> dict:
        model = self._model(name)
        if name in self.blocked:
            return {"status": "not_deployed", "reason": self.blocked[name]}
        if model is None:
            gate = self.gates.get(name, {})
            failed = gate.get("status") in ("failed", "insufficient_data")
            reason = gate.get("reason") if failed else f"no registered {name} model"
            return {"status": "not_deployed", "reason": reason}
        growth = float(model.predict_growth(features)[0])
        half = model.half_width(context["segment"])
        point, low, high = (context["estimate"] * math.exp(growth + d) for d in (0.0, -half, half))
        base = context["base"]
        label = confidence(
            base["base_level"],
            base["base_n"],
            context["days"],
            model.area_rows.get(context["area_id"], 0),
            self.config.low_confidence_area_rows,
        )
        block = {"point": round(point), "ci_low": round(low), "ci_high": round(high)}
        block["confidence"] = label
        if label == "LOW":
            block["message"] = low_message(point, low, high)
        return block

    def _drivers(self, features: pl.DataFrame) -> list[str]:
        name = next((h for h in DRIVER_HORIZONS if self._model(h) is not None), None)
        if name is None:
            return []
        contributions = self._model(name).contributions(features)[0][: len(FEATURES)]
        row = features.row(0, named=True)
        drivers = []
        for index in np.argsort(-np.abs(contributions))[:3]:
            value = float(contributions[index])
            if abs(value) < self.config.min_driver_contribution:
                break
            feature = FEATURES[index]
            drivers.append(driver_text(feature, row[feature], value, name))
        return drivers

    def _exclusions(self, area_id: int, sub_kind: str) -> list[str]:
        mine = self.excluded.filter(
            (pl.col("area_id") == area_id) & (pl.col("sub_kind") == sub_kind)
        )
        names = dict(self.areas.iter_rows())
        area_name = names.get(area_id, f"area {area_id}")
        out = []
        for row in excluded_summary(mine):
            kind = f"{row['reg_type'].replace('_', '-')} " if row["reg_type"] else ""
            reason = row["reason"].replace("_", " ") + ("s" if row["count"] != 1 else "")
            out.append(f"Dropped {row['count']:,} {kind}{reason} in {area_name}")
        return out
