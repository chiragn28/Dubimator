"""Area price statistics for the demo's area explorer: a 12-month summary per area and an
8-year (96-month) monthly history per area and market kind.

Spec: docs/superpowers/specs/2026-09-17-phase8-demo-design.md ("Area statistics").

Scope matches `models.price.data.HOMES_SQL` (via `load_homes`, which already applies it).
Rows then go through Phase 6's `validate_rows` (clean, positive price and size, a date and
an area, no duplicate transaction ids or repeat sales), with no outlier screening — a median is robust to the odd bad
price, so Phase 6's `screen_outliers` step is skipped here.
"""

from dataclasses import dataclass
from datetime import date

import polars as pl

from ingestion.config import DbSettings
from models.forecast.rows import market_kind, validate_rows
from models.price.config import TrainConfig
from models.price.data import load_homes
from models.price.features import month_number

SUMMARY_WINDOW_MONTHS = 12
SUMMARY_MIN_SALES = 20
HISTORY_MONTHS = 96
HISTORY_MIN_SALES = 5


@dataclass(frozen=True)
class AreaStats:
    summary: pl.DataFrame
    history: pl.DataFrame
    names: dict[int, str]
    data_end: date

    @classmethod
    def from_homes(cls, rows: pl.DataFrame, areas: pl.DataFrame, data_end: date) -> "AreaStats":
        """Pure: `rows` are `derive_segments`-shaped home rows (as `load_homes` returns)."""
        end_num = month_number(data_end)
        valid = validate_rows(rows)[0].filter(pl.col("area_id").is_not_null())
        clean = valid.with_columns(
            (pl.col("price_aed") / pl.col("area_sqm")).alias("ppsm"),
            market_kind().alias("market_kind"),
            pl.col("instance_date").dt.truncate("1mo").alias("month"),
        )
        clean = clean.with_columns(
            (pl.col("month").dt.year() * 12 + pl.col("month").dt.month() - 1).alias("_m")
        )

        last_start = end_num - (SUMMARY_WINDOW_MONTHS - 1)
        prev_start = last_start - SUMMARY_WINDOW_MONTHS
        prev_end = last_start - 1

        last12 = clean.filter(pl.col("_m") >= last_start)
        prev12 = clean.filter((pl.col("_m") >= prev_start) & (pl.col("_m") <= prev_end))

        last12_stats = last12.group_by("area_id").agg(
            pl.col("ppsm").median().alias("median_ppsm_12m"), pl.len().alias("sales_12m")
        )
        prev12_stats = prev12.group_by("area_id").agg(
            pl.col("ppsm").median().alias("_prev_median_ppsm")
        )
        summary = (
            last12_stats.join(prev12_stats, on="area_id", how="left")
            .filter(pl.col("sales_12m") >= SUMMARY_MIN_SALES)
            .with_columns(
                (pl.col("median_ppsm_12m") / pl.col("_prev_median_ppsm") - 1.0).alias("change_12m")
            )
            .select("area_id", "median_ppsm_12m", "change_12m", "sales_12m")
            .sort("area_id")
        )

        history_start = end_num - (HISTORY_MONTHS - 1)
        history_rows = clean.filter(pl.col("_m") >= history_start)
        history = (
            history_rows.group_by("area_id", "market_kind", "month")
            .agg(pl.col("ppsm").median().alias("median_ppsm"), pl.len().alias("sales"))
            .with_columns(
                pl.when(pl.col("sales") >= HISTORY_MIN_SALES)
                .then(pl.col("median_ppsm"))
                .otherwise(None)
                .alias("median_ppsm")
            )
            .sort("area_id", "market_kind", "month")
        )

        names = dict(zip(areas["area_id"].to_list(), areas["name_en"].to_list(), strict=True))
        return cls(summary=summary, history=history, names=names, data_end=data_end)

    @classmethod
    def load(cls, settings: DbSettings, homes=None) -> "AreaStats":
        """`homes`: an already-loaded `load_homes` result to reuse (else it is read here)."""
        if homes is None:
            homes = load_homes(settings, TrainConfig())
        return cls.from_homes(homes.rows, homes.areas, homes.data_end)

    def summary_records(self) -> list[dict]:
        records = [
            {
                "area_id": row["area_id"],
                "name": self.names.get(row["area_id"], str(row["area_id"])),
                "median_ppsm_12m": row["median_ppsm_12m"],
                "change_12m": row["change_12m"],
                "sales_12m": row["sales_12m"],
            }
            for row in self.summary.iter_rows(named=True)
        ]
        records.sort(key=lambda record: record["sales_12m"], reverse=True)
        return records

    def history_records(self, area_id: int) -> dict | None:
        if area_id not in self.names:
            return None
        rows = self.history.filter(pl.col("area_id") == area_id)
        series = []
        for kind in sorted(rows["market_kind"].unique().to_list()):
            kind_rows = rows.filter(pl.col("market_kind") == kind).sort("month")
            points = [
                {
                    "month": f"{row['month']:%Y-%m}",
                    "median_ppsm": row["median_ppsm"],
                    "sales": row["sales"],
                }
                for row in kind_rows.iter_rows(named=True)
            ]
            series.append({"market_kind": kind, "points": points})
        return {
            "area_id": area_id,
            "name": self.names[area_id],
            "data_end": self.data_end.isoformat(),
            "series": series,
        }
