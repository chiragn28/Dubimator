"""Price metrics, per-slice breakdowns, split-conformal intervals and coverage."""

import math

import numpy as np
import polars as pl

CONFORMAL_LEVELS = {"q80": 0.20, "q95": 0.05}


def price_metrics(actual, predicted) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    ape = np.abs(predicted - actual) / actual
    log_error = np.log(predicted) - np.log(actual)
    return {
        "mdape": float(np.median(ape)),
        "ppe10": float(np.mean(ape <= 0.10)),
        "ppe20": float(np.mean(ape <= 0.20)),
        "rmse_log": float(np.sqrt(np.mean(log_error**2))),
        "n": float(actual.size),
    }


def sliced_metrics(frame: pl.DataFrame, set_name: str) -> dict[str, float]:
    """Metrics overall, per segment, per location level and per bulk group (dedup)."""
    out: dict[str, float] = {}

    def add(slice_name: str, part: pl.DataFrame) -> None:
        if part.height == 0:
            return
        values = price_metrics(part["actual"].to_numpy(), part["predicted"].to_numpy())
        for metric, value in values.items():
            out[f"{set_name}.{slice_name}.{metric}"] = value

    add("all", frame)
    for segment in sorted(frame["segment"].unique().to_list()):
        add(f"segment.{segment}", frame.filter(pl.col("segment") == segment))
    for level in sorted(frame["loc_level"].unique().to_list()):
        add(f"loc_level.{level}", frame.filter(pl.col("loc_level") == level))
    add("dedup", frame.unique("bulk_group", keep="first", maintain_order=True))
    return out


def conformal_quantile(abs_errors, alpha: float) -> float:
    errors = np.sort(np.asarray(abs_errors, dtype=float))
    rank = math.ceil((errors.size + 1) * (1 - alpha))
    if errors.size == 0 or rank > errors.size:
        return math.inf
    return float(errors[rank - 1])


def fit_conformal(segments, abs_errors, min_rows: int) -> dict[str, dict[str, float]]:
    frame = pl.DataFrame({"segment": list(segments), "e": np.asarray(abs_errors, dtype=float)})
    pooled = {
        name: conformal_quantile(frame["e"].to_numpy(), alpha)
        for name, alpha in CONFORMAL_LEVELS.items()
    }
    if not all(math.isfinite(value) for value in pooled.values()):
        raise ValueError(f"too few validation rows ({frame.height}) for a 95% conformal interval")
    out = {"_pooled": pooled}
    for segment in sorted(frame["segment"].unique().to_list()):
        errors = frame.filter(pl.col("segment") == segment)["e"].to_numpy()
        if errors.size >= min_rows:
            out[segment] = {
                name: conformal_quantile(errors, alpha) for name, alpha in CONFORMAL_LEVELS.items()
            }
        else:
            out[segment] = dict(pooled)
    return out


def quantiles_for(conformal: dict[str, dict[str, float]], segment: str) -> dict[str, float]:
    return conformal.get(segment, conformal["_pooled"])


def coverage(actual, low, high) -> float:
    actual = np.asarray(actual, dtype=float)
    return float(np.mean((actual >= low) & (actual <= high)))
