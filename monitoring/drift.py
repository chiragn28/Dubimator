"""Data and model drift: PSI / KS on price-model inputs, forecast residuals, search log stats.

Spec: docs/superpowers/specs/2026-09-17-phase10-deploy-monitor-design.md
Numeric only (no Evidently): the statistics are small, pure functions and the HTML output is a
self-contained table.
"""

import html
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from models.forecast.config import HORIZON_SPECS, ForecastConfig
from models.forecast.evaluate import ape
from models.forecast.model import load_champion
from models.price.config import TrainConfig
from models.price.data import HomesData, load_homes

OUT_DIR = Path("data/monitoring")
EPSILON = 1e-4
PSI_THRESHOLD = 0.2
# KS above this also flags a numeric feature: PSI on quantile bins can miss a large shift
# that stays inside few bins, and the max ECDF gap catches it.
KS_THRESHOLD = 0.3
OTHER = "__other__"
NUMERIC_FEATURES = ("area_sqm", "bedrooms", "price_per_sqm")
CATEGORICAL_FEATURES = ("reg_type", "sub_kind", "area_id")
TOP_CATEGORIES = {"area_id": 30}
FORECAST_HORIZON = "3m"
NO_NEWER = "no newer resolved targets"
DRIFT_SCHEMA = {
    "feature": pl.Utf8,
    "kind": pl.Utf8,
    "psi": pl.Float64,
    "ks": pl.Float64,
    "flagged": pl.Boolean,
}


def _finite(values) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    return array[np.isfinite(array)]


def _psi_from_counts(reference: np.ndarray, current: np.ndarray) -> float:
    expected = np.maximum(reference / reference.sum(), EPSILON)
    actual = np.maximum(current / current.sum(), EPSILON)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _three_way_counts(values: np.ndarray, pivot: float) -> np.ndarray:
    return np.array(
        [np.sum(values < pivot), np.sum(values == pivot), np.sum(values > pivot)], dtype=float
    )


def psi(reference, current, bins: int = 10) -> float:
    """Population stability index with quantile bins taken from the reference.

    Non-finite values are ignored; empty bins are floored at EPSILON so the index stays finite.
    Returns NaN when either side has no finite values. A constant reference has no quantile
    edges to split on, so it uses three bins instead: below, equal to and above its one value.
    """
    ref, cur = _finite(reference), _finite(current)
    if ref.size == 0 or cur.size == 0:
        return math.nan
    if np.unique(ref).size == 1:
        value = ref[0]
        return _psi_from_counts(_three_way_counts(ref, value), _three_way_counts(cur, value))
    edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, bins + 1)))
    inner = edges[1:-1]
    slots = inner.size + 1
    ref_counts = np.bincount(np.searchsorted(inner, ref, side="right"), minlength=slots)
    cur_counts = np.bincount(np.searchsorted(inner, cur, side="right"), minlength=slots)
    return _psi_from_counts(ref_counts.astype(float), cur_counts.astype(float))


def _labels(values) -> list[str]:
    return ["<null>" if value is None else str(value) for value in values]


def psi_categorical(reference, current, top: int | None = None) -> float:
    """PSI over category shares. With `top`, categories outside the reference's `top` most
    frequent are pooled into one "other" bucket (keeps high-cardinality ids readable)."""
    ref, cur = _labels(reference), _labels(current)
    if not ref or not cur:
        return math.nan
    ref_counts = pl.Series(ref).value_counts(sort=True)
    if top is not None:
        keep = set(ref_counts.head(top)[ref_counts.columns[0]].to_list())
        ref = [value if value in keep else OTHER for value in ref]
        cur = [value if value in keep else OTHER for value in cur]
    categories = sorted(set(ref) | set(cur))
    index = {name: position for position, name in enumerate(categories)}
    ref_array = np.bincount([index[v] for v in ref], minlength=len(categories)).astype(float)
    cur_array = np.bincount([index[v] for v in cur], minlength=len(categories)).astype(float)
    return _psi_from_counts(ref_array, cur_array)


def ks_stat(reference, current) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (max ECDF gap); NaN if either side is empty."""
    ref, cur = np.sort(_finite(reference)), np.sort(_finite(current))
    if ref.size == 0 or cur.size == 0:
        return math.nan
    points = np.concatenate([ref, cur])
    ref_cdf = np.searchsorted(ref, points, side="right") / ref.size
    cur_cdf = np.searchsorted(cur, points, side="right") / cur.size
    return float(np.max(np.abs(ref_cdf - cur_cdf)))


def feature_drift(
    reference_frame: pl.DataFrame,
    current_frame: pl.DataFrame,
    numeric: Sequence[str],
    categorical: Sequence[str],
    *,
    threshold: float = PSI_THRESHOLD,
    top_categories: dict[str, int] | None = None,
) -> pl.DataFrame:
    """One row per feature: feature, kind, psi, ks (numeric only) and flagged.

    A feature is flagged when psi > threshold or, for numeric features, ks > KS_THRESHOLD.
    """
    top_categories = top_categories or {}
    rows = []
    for name in numeric:
        ref, cur = reference_frame[name].to_numpy(), current_frame[name].to_numpy()
        value, ks = psi(ref, cur), ks_stat(ref, cur)
        flagged = bool(value > threshold or ks > KS_THRESHOLD)
        rows.append((name, "numeric", value, ks, flagged))
    for name in categorical:
        value = psi_categorical(
            reference_frame[name].to_list(),
            current_frame[name].to_list(),
            top=top_categories.get(name),
        )
        rows.append((name, "categorical", value, None, bool(value > threshold)))
    table = pl.DataFrame(rows, schema=DRIFT_SCHEMA, orient="row")
    return table.with_columns(pl.col("psi", "ks").fill_nan(None))


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def search_log_stats(lines: Iterable[str]) -> dict:
    """Search traffic from the API's JSON log lines.

    Requests and 5xx errors come from `api.access` lines on /v1/search. Query lengths come from
    `api.search` lines (logged at DEBUG, so they appear only when the API logs at that level).
    The zero-result rate needs a `results` count on those lines; None when none carries one.
    """
    requests = errors = skipped = 0
    chars, words, results = [], [], []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            skipped += 1
            continue
        if not isinstance(entry, dict):
            skipped += 1
            continue
        logger = entry.get("logger")
        if logger == "api.access" and entry.get("path") == "/v1/search":
            requests += 1
            errors += int(int(entry.get("status", 0)) >= 500)
        elif logger == "api.search" and isinstance(entry.get("q"), str):
            query = entry["q"]
            chars.append(len(query))
            words.append(len(query.split()))
            if isinstance(entry.get("results"), int):
                results.append(entry["results"])
    return {
        "search_requests": requests,
        "search_errors": errors,
        "search_error_rate": errors / requests if requests else None,
        "queries_logged": len(chars),
        "query_chars_mean": _mean(chars),
        "query_words_mean": _mean(words),
        "zero_result_rate": sum(r == 0 for r in results) / len(results) if results else None,
        "skipped_lines": skipped,
    }


def resolved_frame(rows: pl.DataFrame) -> pl.DataFrame:
    """Forecast rows with features and a usable 3m target (the full Phase 6 dataset build)."""
    from models.forecast.features import build_dataset
    from models.forecast.folds import usable_rows
    from models.forecast.infra import load_projects
    from models.forecast.rows import prepare_rows

    kept, _ = prepare_rows(rows, ForecastConfig())
    frame, _ = build_dataset(kept, kept["instance_date"].max(), load_projects())
    return usable_rows(frame, HORIZON_SPECS[FORECAST_HORIZON])


def forecast_section(homes: HomesData) -> dict:
    """3m champion error on rows newer than its test window whose targets have fully happened."""
    name = f"{ForecastConfig().model_prefix}-{FORECAST_HORIZON}"
    base = {"model": name}
    try:
        model, version = load_champion(name)
    except Exception as exc:  # noqa: BLE001 — a missing registry is reported, not fatal
        return {**base, "status": "unavailable", "reason": str(exc)[:300]}
    if model is None:
        return {**base, "status": "no champion"}
    test_end = date.fromisoformat(str(model.metadata["test_end"]))
    gate_mape = (model.metadata.get("gate") or {}).get("model_mape")
    base |= {"version": version, "test_end": test_end.isoformat(), "gate_mape": gate_mape}
    window_days = HORIZON_SPECS[FORECAST_HORIZON].end_days
    latest_start = homes.data_end - timedelta(days=window_days)
    candidates = homes.rows.filter(
        pl.col("is_clean")
        & (pl.col("instance_date") >= test_end)
        & (pl.col("instance_date") <= latest_start)
    )
    if candidates.height == 0:  # cheap check before the full feature build
        return {**base, "status": NO_NEWER}
    scored = resolved_frame(homes.rows).filter(pl.col("instance_date") >= test_end)
    if scored.height == 0:
        return {**base, "status": NO_NEWER}
    errors = ape(model.predict_growth(scored), scored[f"growth_{FORECAST_HORIZON}"].to_numpy())
    mape = float(np.mean(errors))
    return {
        **base,
        "status": "scored",
        "rows": scored.height,
        "mape": mape,
        "mape_vs_gate": mape / gate_mape if gate_mape else None,
    }


@dataclass(frozen=True)
class DriftReport:
    payload: dict
    json_path: Path
    html_path: Path


def _clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _fmt(value) -> str:
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _table(headers: Sequence[str], rows: Iterable[Sequence], flag_col: int | None = None) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = []
    for row in rows:
        css = ' class="flag"' if flag_col is not None and row[flag_col] else ""
        cells = "".join(f"<td>{html.escape(_fmt(cell))}</td>" for cell in row)
        body.append(f"<tr{css}>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


STYLE = """
body{font-family:system-ui,sans-serif;margin:24px;color:#1b1f24;background:#fff}
h1{font-size:1.4rem}h2{font-size:1.1rem;margin-top:1.6rem}
table{border-collapse:collapse;margin:8px 0}
th,td{border:1px solid #d0d7de;padding:4px 10px;text-align:left}
th{background:#f3f4f6}tr.flag td{background:#fde2e1}
.muted{color:#57606a}
"""


def render_html(payload: dict) -> str:
    windows = payload["windows"]
    features = [
        (r["feature"], r["kind"], r["psi"], r["ks"], r["flagged"]) for r in payload["features"]
    ]
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>Drift report {html.escape(payload['generated'])}</title>",
        f"<style>{STYLE}</style></head><body>",
        f"<h1>Zestimator drift report — {html.escape(payload['generated'])}</h1>",
        (
            f'<p class="muted">PSI above {PSI_THRESHOLD} (or KS above {KS_THRESHOLD}) is '
            f"flagged. Data ends {html.escape(payload['data_end'])}.</p>"
        ),
        "<h2>Windows</h2>",
        _table(
            ("window", "start", "end", "rows"),
            [(k, w["start"], w["end"], w["rows"]) for k, w in windows.items()],
        ),
        "<h2>Price-model inputs</h2>",
        _table(("feature", "kind", "psi", "ks", "flagged"), features, flag_col=4),
        "<h2>Forecast residuals (3m)</h2>",
        _table(("field", "value"), list(payload["forecast"].items())),
    ]
    if payload["search"] is not None:
        parts += ["<h2>Search traffic</h2>", _table(("field", "value"), payload["search"].items())]
    parts.append("</body></html>")
    return "\n".join(parts)


def report(
    settings,
    months: int = 3,
    log_file: Path | None = None,
    out_dir: Path = OUT_DIR,
    today: date | None = None,
) -> DriftReport:
    """Compare the price model's training window with the last `months` months of sales."""
    config = TrainConfig()
    homes = load_homes(settings, config)
    clean = homes.rows.filter(pl.col("is_clean")).with_columns(
        (pl.col("price_aed") / pl.col("area_sqm")).alias("price_per_sqm"),
        pl.col("area_id").cast(pl.Utf8),
    )
    reference = clean.filter(
        pl.col("instance_date").is_between(config.train_start, config.val_start, closed="left")
    )
    current_start = pl.select(pl.lit(homes.data_end).dt.offset_by(f"-{months}mo")).item()
    current = clean.filter(
        pl.col("instance_date").is_between(current_start, homes.data_end, closed="both")
    )
    table = feature_drift(
        reference,
        current,
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
        top_categories=TOP_CATEGORIES,
    )
    search = None
    if log_file is not None:
        with Path(log_file).open(encoding="utf-8", errors="replace") as handle:
            search = search_log_stats(handle)
    today = today or datetime.now(UTC).date()
    payload = {
        "generated": today.isoformat(),
        "data_end": homes.data_end.isoformat(),
        "lineage": homes.lineage,
        "psi_threshold": PSI_THRESHOLD,
        "ks_threshold": KS_THRESHOLD,
        "windows": {
            "reference": {
                "start": config.train_start.isoformat(),
                "end": config.val_start.isoformat(),
                "rows": reference.height,
            },
            "current": {
                "start": current_start.isoformat(),
                "end": homes.data_end.isoformat(),
                "rows": current.height,
            },
        },
        "features": [
            {key: _clean(value) for key, value in row.items()} for row in table.to_dicts()
        ],
        "flagged": table.filter(pl.col("flagged"))["feature"].to_list(),
        "forecast": forecast_section(homes),
        "search": search,
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"drift_{today.isoformat()}"
    json_path = out_dir / f"{stem}.json"
    html_path = out_dir / f"{stem}.html"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    html_path.write_text(render_html(payload), encoding="utf-8")
    return DriftReport(payload, json_path, html_path)
