"""Walk-forward folds per horizon with the target-window leakage guard.

A training row's target window must end before its fold's cutoff: T + end_days < cutoff.
The grid is anchored backwards from the last usable T, so the last fold (the test period) is a
full step that contains it. A non-test fold whose target windows can reach the test period is a
`gap` fold: scored and reported only. Up to tune_folds of the latest remaining folds tune the
model (and calibrate the ranges); the rest are `score` folds.
"""

from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl

from models.forecast.config import ForecastConfig, Horizon

MONTH_DAYS = 30.4375


@dataclass(frozen=True)
class Fold:
    index: int
    cutoff: date
    end: date  # exclusive: the next cutoff
    role: str  # "score", "tune", "gap" or "test"

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "cutoff": self.cutoff.isoformat(),
            "end": self.end.isoformat(),
            "role": self.role,
        }


def month_start(day: date) -> date:
    return day.replace(day=1)


def add_months(first_of_month: date, months: int) -> date:
    if first_of_month.day != 1:
        raise ValueError(f"add_months needs the first of a month, got {first_of_month}")
    total = first_of_month.year * 12 + first_of_month.month - 1 + months
    return date(total // 12, total % 12 + 1, 1)


def usable_rows(frame: pl.DataFrame, horizon: Horizon) -> pl.DataFrame:
    return frame.filter(pl.col(f"growth_{horizon.name}").is_not_null())


def _cutoffs(first_t: date, last_t: date, horizon: Horizon, config: ForecastConfig) -> list[date]:
    history = round(config.min_train_months * MONTH_DAYS)
    anchor = first_t + timedelta(days=history + horizon.end_days)
    earliest = add_months(month_start(anchor), 1)
    last = add_months(month_start(last_t), -(horizon.step_months - 1))
    cutoffs = []
    cutoff = last
    while cutoff >= earliest:
        cutoffs.append(cutoff)
        cutoff = add_months(cutoff, -horizon.step_months)
    return cutoffs[::-1]


def plan_folds(frame: pl.DataFrame, horizon: Horizon, config: ForecastConfig) -> list[Fold]:
    usable = usable_rows(frame, horizon)
    if usable.height == 0:
        return []
    cutoffs = _cutoffs(
        usable["instance_date"].min(), usable["instance_date"].max(), horizon, config
    )
    if not cutoffs:
        return []
    windows = [(start, add_months(start, horizon.step_months)) for start in cutoffs]
    test_cutoff = cutoffs[-1]
    reach = timedelta(days=horizon.end_days)
    roles = ["score"] * len(windows)
    roles[-1] = "test"
    for index, (_, end) in enumerate(windows[:-1]):
        if end + reach > test_cutoff:
            roles[index] = "gap"
    remaining = [index for index, role in enumerate(roles) if role == "score"]
    for index in remaining[-config.tune_folds :] if config.tune_folds > 0 else []:
        roles[index] = "tune"
    return [
        Fold(index, start, end, role)
        for index, ((start, end), role) in enumerate(zip(windows, roles, strict=True))
    ]


def fold_split(
    frame: pl.DataFrame, horizon: Horizon, fold: Fold
) -> tuple[pl.DataFrame, pl.DataFrame]:
    usable = usable_rows(frame, horizon)
    window_end = pl.col("instance_date").dt.offset_by(f"{horizon.end_days}d")
    train = usable.filter(window_end < fold.cutoff)
    val = usable.filter(pl.col("instance_date").is_between(fold.cutoff, fold.end, closed="left"))
    return train, val


def insufficiency(folds: list[Fold], test_rows: int, config: ForecastConfig) -> str | None:
    """Why a horizon cannot be trained and gated, or None when it can."""
    if len(folds) < config.min_folds:
        return f"only {len(folds)} walk-forward folds (needs {config.min_folds})"
    if not any(fold.role == "tune" for fold in folds):
        return "no tuning fold ends before the test period's target windows"
    if test_rows < config.min_test_rows:
        return f"only {test_rows:,} test rows (needs {config.min_test_rows:,})"
    return None
