"""Walk-forward folds per horizon with the target-window leakage guard.

A training row's target window must end before its fold's cutoff: T + end_days < cutoff.
The last fold is the test period; the (up to) tune_folds folds before it tune the model.
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
    role: str  # "score", "tune" or "test"

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


def plan_folds(frame: pl.DataFrame, horizon: Horizon, config: ForecastConfig) -> list[Fold]:
    usable = usable_rows(frame, horizon)
    if usable.height == 0:
        return []
    first_t = usable["instance_date"].min()
    last_t = usable["instance_date"].max()
    history = round(config.min_train_months * MONTH_DAYS)
    anchor = first_t + timedelta(days=history + horizon.end_days)
    cutoff = add_months(month_start(anchor), 1)
    cutoffs = []
    while cutoff <= last_t:
        cutoffs.append(cutoff)
        cutoff = add_months(cutoff, horizon.step_months)
    last = len(cutoffs) - 1
    folds = []
    for index, start in enumerate(cutoffs):
        if index == last:
            role = "test"
        elif index >= last - config.tune_folds:
            role = "tune"
        else:
            role = "score"
        folds.append(Fold(index, start, add_months(start, horizon.step_months), role))
    return folds


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
    if test_rows < config.min_test_rows:
        return f"only {test_rows:,} test rows (needs {config.min_test_rows:,})"
    return None
