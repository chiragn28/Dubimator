import math
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
import xgboost as xgb

from models.price.config import CATEGORICAL_FEATURES, FEATURES
from models.price.data import RAW_SCHEMA
from models.price.features import (
    INDEX_SCHEMA,
    LocationPriors,
    MarketIndex,
    add_location_keys,
    derive_segments,
)
from models.price.predictor import ModelBundle

BASE_HOME = {
    "transaction_id": "t",
    "instance_date": date(2020, 1, 15),
    "property_type": "unit",
    "property_sub_type": "Flat",
    "reg_type": "ready",
    "area_id": 1,
    "building_name": "Tower A",
    "project_name": "Project P",
    "rooms": "1 B/R",
    "has_parking": True,
    "area_sqm": 80.0,
    "price_aed": 800_000.0,
    "ingest_run_id": 1,
    "is_clean": True,
}


def build_raw_homes(*overrides: dict) -> pl.DataFrame:
    """One raw home row per override dict, each starting from BASE_HOME."""
    records = [{**BASE_HOME, "transaction_id": f"t{i}", **row} for i, row in enumerate(overrides)]
    return pl.DataFrame(records, schema=RAW_SCHEMA)


@pytest.fixture
def raw_homes():
    return build_raw_homes


def build_synthetic_homes(seed: int = 0, per_month: int = 6) -> pl.DataFrame:
    """Prepared home rows 2014-01..2023-03: prices rise 0.4%/month, area and villa premia."""
    rng = np.random.default_rng(seed)
    records = []
    month = date(2014, 1, 1)
    while month <= date(2023, 3, 1):
        months_since = (month.year - 2014) * 12 + month.month - 1
        level = math.log(9_000.0) + 0.004 * months_since
        for j in range(per_month):
            villa = j % 3 == 2
            area_id = int(rng.integers(1, 4))
            size = float(rng.uniform(250.0, 600.0) if villa else rng.uniform(40.0, 150.0))
            ln_ppsqm = level + 0.15 * area_id + (0.2 if villa else 0.0) + rng.normal(0.0, 0.1)
            records.append(
                {
                    "transaction_id": f"s{len(records)}",
                    "instance_date": month.replace(day=int(rng.integers(1, 28))),
                    "property_type": "villa" if villa else "unit",
                    "property_sub_type": "Villa" if villa else "Flat",
                    "reg_type": "ready",
                    "area_id": area_id,
                    "building_name": None if villa else f"Tower {area_id}{int(rng.integers(0, 3))}",
                    "project_name": f"Project {area_id}",
                    "rooms": f"{int(rng.integers(1, 4))} B/R",
                    "has_parking": bool(rng.integers(0, 2)),
                    "area_sqm": round(size, 1),
                    "price_aed": round(math.exp(ln_ppsqm) * size, -2),
                    "ingest_run_id": 1,
                    "is_clean": not (month >= date(2022, 11, 1) and j == 0),
                }
            )
        month = (month.replace(day=28) + timedelta(days=4)).replace(day=1)
    return derive_segments(pl.DataFrame(records, schema=RAW_SCHEMA))


@pytest.fixture
def synthetic_homes():
    return build_synthetic_homes


SUPPORTED = (
    "unit_off_plan_built_up", "unit_ready_built_up",
    "villa_off_plan_built_up", "villa_ready_built_up", "villa_ready_plot",
)  # fmt: skip
TINY_CATEGORIES = {
    "property_type": ["unit", "villa"],
    "reg_type": ["off_plan", "ready"],
    "size_basis": ["built_up", "plot"],
    "sub_kind": ["flat", "hotel_apartment", "townhouse", "villa"],
    "room_kind": ["bedrooms", "penthouse", "single_room", "studio", "unknown"],
    "area_id": ["10", "20"],
}


class ConstantBooster:
    """Stands in for an XGBoost booster: always predicts the same relative price."""

    def __init__(self, value: float):
        self.value = value

    def predict(self, dmatrix, **_):
        return np.full(dmatrix.num_row(), self.value)


def _train_tiny_booster() -> xgb.Booster:
    # Imported here, not at module level: pandas (via pyarrow) must never load before
    # models.price's package __init__ preloads the system msvcp140.dll, or LightGBM crashes
    # with an access violation later in the session (see models/price/__init__.py). This
    # module's own models.price imports above (executed at module load) already guarantee
    # that preload has happened by the time this function runs.
    import pandas as pd

    rng = np.random.default_rng(0)
    columns = {}
    for name in FEATURES:
        if name in CATEGORICAL_FEATURES:
            values = rng.choice(TINY_CATEGORIES[name], size=64)
            columns[name] = pd.Categorical(values, categories=TINY_CATEGORIES[name])
        else:
            columns[name] = rng.normal(size=64)
    frame = pd.DataFrame(columns)[list(FEATURES)]
    dtrain = xgb.DMatrix(frame, label=rng.normal(scale=0.05, size=64), enable_categorical=True)
    return xgb.train({"max_depth": 2, "tree_method": "hist"}, dtrain, num_boost_round=5)


def build_tiny_bundle(y_hat: float | None = None) -> ModelBundle:
    march = date(2023, 3, 1)
    index = MarketIndex(
        pl.DataFrame(
            [(segment, march, math.log(10_000.0), "segment_3") for segment in SUPPORTED],
            schema=INDEX_SCHEMA,
            orient="row",
        )
    )
    fit = add_location_keys(
        pl.DataFrame(
            {
                "property_type": ["unit"] * 5 + ["villa"] * 5,
                "area_id": [10] * 5 + [20] * 5,
                "project_name": ["Marina Gate"] * 5 + [None] * 5,
                "building_name": ["Marina Gate 1"] * 5 + [None] * 5,
                "y": [0.1] * 5 + [0.0] * 5,
                "bulk_weight": [1.0] * 10,
            }
        )
    )
    return ModelBundle(
        booster=_train_tiny_booster() if y_hat is None else ConstantBooster(y_hat),
        index=index,
        priors=LocationPriors.fit(fit, shrink_k=10.0, min_level_n=3.0),
        categories=TINY_CATEGORIES,
        bounds_area=pl.DataFrame(
            {
                "property_type": ["unit"],
                "size_basis": ["built_up"],
                "area_id": [10],
                "lo": [-0.5],
                "hi": [0.5],
            }
        ),
        bounds_segment=pl.DataFrame(
            {
                "property_type": ["unit", "villa", "villa"],
                "size_basis": ["built_up", "built_up", "plot"],
                "lo": [-1.0] * 3,
                "hi": [1.0] * 3,
            }
        ),
        size_percentiles=pl.DataFrame(
            {"segment": ["unit_ready_built_up"], "p005": [30.0], "p995": [400.0]}
        ),
        areas=pl.DataFrame(
            {
                "area_id": [10, 20, 30, 31],
                "name_en": ["Marsa Dubai", "Al Barsha South Fourth", "Mushrif", "Mushrif"],
            }
        ),
        aliases=pl.DataFrame(
            {
                "alias_key": [
                    "barsha south fourth",
                    "dubai marina",
                    "jbr",
                    "jvc",
                    "marsa dubai",
                    "mushrif",
                    "mushrif",
                ],
                "area_id": [20, 10, 10, 20, 10, 30, 31],
            }
        ),
        conformal={
            "_pooled": {"q80": 0.1, "q95": 0.2},
            "unit_ready_built_up": {"q80": 0.15, "q95": 0.3},
        },
        supported_segments=SUPPORTED,
        data_end=date(2023, 3, 17),
        metadata={"model_version": "run-abc"},
    )


@pytest.fixture
def tiny_bundle():
    return build_tiny_bundle


@pytest.fixture
def temp_mlflow(tmp_path, monkeypatch):
    """Throwaway MLflow tracking + registry store. Never the real server, never ./mlruns."""
    import mlflow

    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    # mlflow.set_tracking_uri() writes this module global; monkeypatch restores it after the test.
    monkeypatch.setattr("mlflow.tracking._tracking_service.utils._tracking_uri", uri)
    monkeypatch.setattr("mlflow.tracking.fluent._active_experiment_id", None)
    yield {"tracking_uri": uri, "artifact_location": (tmp_path / "artifacts").as_uri()}
    while mlflow.active_run():
        mlflow.end_run()
