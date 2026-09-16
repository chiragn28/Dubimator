"""Price-error metrics, evaluation segments, the bootstrap bound and the per-horizon gate.

There is deliberately no single headline accuracy number: every table is per segment.
"""

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from models.forecast.baselines import BASELINES
from models.forecast.config import Horizon

MODEL = "model"
FLAG_MAPE = 0.25
SEGMENTS = (
    "all", "off_plan", "ready", "top5", "rest",
    "age_lt1", "age_1to5", "age_gt5", "age_unknown", "age_gt2",
)  # fmt: skip
PRIMARY_SEGMENTS = ("ready", "top5", "age_gt2")
SEGMENT_TEXT = {"ready": "resale", "top5": "top-5-area", "age_gt2": "older-building (>2y)"}
BOOTSTRAP_CHUNK = 100
TABLE_SCHEMA = {
    "segment": pl.Utf8,
    "model": pl.Utf8,
    "rows": pl.Int64,
    "mape": pl.Float64,
    "median_ape": pl.Float64,
    "flagged": pl.Boolean,
}


def ape(predicted_growth, actual_growth) -> np.ndarray:
    """|predicted price / actual price - 1| where both prices share the same base."""
    difference = np.asarray(predicted_growth, dtype=float) - np.asarray(actual_growth, dtype=float)
    return np.abs(np.expm1(difference))


def top_areas(train: pl.DataFrame, count: int) -> list[int]:
    counts = train.group_by("area_id").len().sort(["len", "area_id"], descending=[True, False])
    return counts["area_id"].head(count).to_list()


def segment_masks(frame: pl.DataFrame, top: list[int]) -> dict[str, np.ndarray]:
    age = pl.col("building_age_proxy_years")
    in_top = pl.col("area_id").is_in(top)
    expressions = {
        "off_plan": pl.col("reg_type") == "off_plan",
        "ready": pl.col("reg_type") == "ready",
        "top5": in_top,
        "rest": ~in_top,
        "age_lt1": age < 1,
        "age_1to5": age.is_between(1, 5),
        "age_gt5": age > 5,
        "age_unknown": age.is_null(),
        "age_gt2": age > 2,
    }
    masks = frame.select(
        expression.fill_null(False).alias(name) for name, expression in expressions.items()
    )
    out = {"all": np.ones(frame.height, dtype=bool)}
    out.update({name: masks[name].to_numpy() for name in expressions})
    return {name: out[name] for name in SEGMENTS}


def segment_table(frame, predictions: dict[str, np.ndarray], actual, top) -> pl.DataFrame:
    actual = np.asarray(actual, dtype=float)
    records = []
    for segment, mask in segment_masks(frame, top).items():
        for name, predicted in predictions.items():
            errors = ape(np.asarray(predicted)[mask], actual[mask])
            mape = float(errors.mean()) if errors.size else math.nan
            records.append(
                {
                    "segment": segment,
                    "model": name,
                    "rows": int(mask.sum()),
                    "mape": mape,
                    "median_ape": float(np.median(errors)) if errors.size else math.nan,
                    "flagged": bool(errors.size) and mape > FLAG_MAPE,
                }
            )
    return pl.DataFrame(records, schema=TABLE_SCHEMA)


def bootstrap_mape_upper(errors, n_resamples: int, seed: int) -> float:
    """Upper end of the 95% percentile interval of the mean, from row-level resamples."""
    errors = np.asarray(errors, dtype=float)
    if errors.size == 0:
        return math.nan
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, n_resamples, BOOTSTRAP_CHUNK):
        size = min(BOOTSTRAP_CHUNK, n_resamples - start)
        picks = rng.integers(0, errors.size, size=(size, errors.size))
        means.append(errors[picks].mean(axis=1))
    return float(np.quantile(np.concatenate(means), 0.975))


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: tuple[str, ...]
    checks: dict[str, float]


def table_mape(table: pl.DataFrame, segment: str, model: str) -> float:
    row = table.filter((pl.col("segment") == segment) & (pl.col("model") == model))
    if row.height == 0 or row["rows"][0] == 0:
        return math.nan
    return float(row["mape"][0])


def gate(table: pl.DataFrame, horizon: Horizon, model_upper: float) -> GateResult:
    reasons = []
    checks = {}
    for segment in PRIMARY_SEGMENTS:
        value = table_mape(table, segment, MODEL)
        checks[f"{segment}_mape"] = value
        if math.isnan(value):
            reasons.append(f"{horizon.name} has no {SEGMENT_TEXT[segment]} test rows")
        elif value > horizon.mape_gate:
            reasons.append(
                f"{horizon.name} {SEGMENT_TEXT[segment]} MAPE {value:.1%} exceeds the "
                f"{horizon.mape_gate:.0%} gate"
            )
    model_all = table_mape(table, "all", MODEL)
    baseline_all = {name: table_mape(table, "all", name) for name in BASELINES}
    for name, value in baseline_all.items():
        checks[f"{name}_mape"] = value
        if not model_all < value:
            reasons.append(
                f"{horizon.name} MAPE {model_all:.1%} does not beat the {name} baseline "
                f"({value:.1%})"
            )
    strongest = min(baseline_all.values())
    checks.update(model_mape=model_all, model_upper=model_upper, strongest_baseline=strongest)
    if not model_upper < strongest:
        reasons.append(
            f"{horizon.name} MAPE 95% upper bound {model_upper:.1%} is not below the stronger "
            f"baseline ({strongest:.1%})"
        )
    return GateResult(not reasons, tuple(reasons), checks)


def _pct(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.2%}"


def format_table(name: str, table: pl.DataFrame) -> list[str]:
    models = table["model"].unique(maintain_order=True).to_list()
    header = f"{name:<4}{'segment':<13}{'rows':>8}" + "".join(f"{m:>13}" for m in models)
    lines = [header + f"{'model MdAPE':>13}  flag"]
    for segment in SEGMENTS:
        part = table.filter(pl.col("segment") == segment)
        mapes = dict(zip(part["model"].to_list(), part["mape"].to_list(), strict=True))
        mine = part.filter(pl.col("model") == MODEL).row(0, named=True)
        cells = "".join(f"{_pct(mapes[m]):>13}" for m in models)
        flag = "FLAG >25%" if mine["flagged"] else ""
        lines.append(
            f"{'':<4}{segment:<13}{part['rows'][0]:>8,}{cells}{_pct(mine['median_ape']):>13}  {flag}"
        )
    return lines
