"""Split-conformal growth ranges, confidence labels and the LOW-confidence message."""

import math

import numpy as np
import polars as pl

from models.forecast.config import ForecastConfig
from models.price.evaluate import conformal_quantile

POOLED = "_pooled"
LOW_MESSAGE = (
    "Limited comparable data for this property type/area. Estimate has wide uncertainty (±{pct}%)."
)
HIGH_MIN_SALES = 10
HIGH_MAX_DAYS = 90
MEDIUM_MIN_SALES = 5
MEDIUM_MAX_DAYS = 180


def interval_segments(frame: pl.DataFrame) -> pl.Series:
    kind = pl.when(pl.col("sub_kind") == "villa").then(pl.lit("villa")).otherwise(pl.lit("unit"))
    return frame.select(pl.format("{}_{}", "reg_type", kind).alias("segment"))["segment"]


def fit_intervals(segments, abs_errors, config: ForecastConfig) -> dict[str, float]:
    """Half-widths (log growth) from calibration |errors|; small segments use the pooled one."""
    frame = pl.DataFrame(
        {"segment": list(segments), "error": np.asarray(abs_errors, dtype=float)},
        schema={"segment": pl.Utf8, "error": pl.Float64},
    )
    pooled = conformal_quantile(frame["error"].to_numpy(), config.alpha)
    if not math.isfinite(pooled):
        raise ValueError(
            f"too few calibration rows ({frame.height}) for an {1 - config.alpha:.0%} interval"
        )
    out = {POOLED: pooled}
    for segment in sorted(frame["segment"].unique().to_list()):
        errors = frame.filter(pl.col("segment") == segment)["error"].to_numpy()
        enough = errors.size >= config.min_conformal_rows
        out[segment] = conformal_quantile(errors, config.alpha) if enough else pooled
    return out


def half_width(intervals: dict[str, float], segment: str) -> float:
    return intervals.get(segment, intervals[POOLED])


def coverage_by_segment(segments, predicted, actual, intervals) -> dict[str, float]:
    segments = list(segments)
    halves = np.array([half_width(intervals, segment) for segment in segments])
    residuals = np.abs(np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float))
    inside = residuals <= halves
    if inside.size == 0:
        return {}
    frame = pl.DataFrame({"segment": segments, "inside": inside})
    by_segment = frame.group_by("segment").agg(pl.col("inside").mean()).sort("segment")
    return {"all": float(inside.mean()), **dict(by_segment.iter_rows())}


def confidence(base_level, base_n, comparable_days, area_rows: int, min_area_rows: int) -> str:
    if base_level is None or base_n is None or comparable_days is None:
        return "LOW"
    if area_rows < min_area_rows:
        return "LOW"
    if base_level == "building" and base_n >= HIGH_MIN_SALES and comparable_days <= HIGH_MAX_DAYS:
        return "HIGH"
    if base_n >= MEDIUM_MIN_SALES and comparable_days <= MEDIUM_MAX_DAYS:
        return "MEDIUM"
    return "LOW"


def low_message(point: float, low: float, high: float) -> str:
    return LOW_MESSAGE.format(pct=round(100 * (high - low) / 2 / point))
