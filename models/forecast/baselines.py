"""Baseline growth forecasts, computed on exactly the rows the model is scored on."""

import numpy as np
import polars as pl

from models.forecast.config import Horizon

BASELINES = ("no_change", "area_trend")


def baseline_growth(frame: pl.DataFrame, horizon: Horizon) -> dict[str, np.ndarray]:
    """no_change = 0; area_trend = the area's trailing ln change, else the city's, else 0."""
    trend = frame.select(
        pl.coalesce(
            pl.col(f"area_mom_{horizon.momentum}").fill_nan(None),
            pl.col(f"city_mom_{horizon.momentum}").fill_nan(None),
            pl.lit(0.0),
        ).alias("trend")
    )["trend"]
    return {
        "no_change": np.zeros(frame.height),
        "area_trend": trend.to_numpy().astype(float),
    }
