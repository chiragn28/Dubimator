"""report() on synthetic home rows (load_homes and the champion loader are monkeypatched)."""

import json
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from models.price.data import HomesData
from monitoring import drift


def _rows(start: date, days: int, n: int, size_mean: float, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [start + timedelta(days=int(offset)) for offset in rng.integers(0, days, n)]
    sizes = rng.normal(size_mean, 10.0, n).clip(20, None)
    return pl.DataFrame(
        {
            "transaction_id": [f"t{seed}-{i}" for i in range(n)],
            "instance_date": dates,
            "reg_type": rng.choice(["Existing Properties", "Off-Plan Properties"], n).tolist(),
            "sub_kind": rng.choice(["apartment", "villa"], n).tolist(),
            "area_id": rng.integers(1, 30, n).tolist(),
            "bedrooms": rng.integers(0, 4, n).astype(float).tolist(),
            "area_sqm": sizes.tolist(),
            "price_aed": (sizes * rng.normal(10_000, 500, n)).tolist(),
            "is_clean": [True] * n,
        },
        schema_overrides={"area_id": pl.Int64},
    )


@pytest.fixture
def homes() -> HomesData:
    train = _rows(date(2016, 1, 1), 2000, 3000, 80.0, seed=1)
    recent = _rows(date(2023, 1, 1), 89, 800, 120.0, seed=2)  # sizes shifted up
    dirty = _rows(date(2023, 2, 1), 10, 50, 500.0, seed=3).with_columns(is_clean=pl.lit(False))
    rows = pl.concat([train, recent, dirty])
    return HomesData(
        rows=rows,
        data_end=date(2023, 3, 30),
        lineage={"ingest_run_id": 1, "source_sha256": "abc"},
        drop_counts={},
        supported_segments=(),
        areas=pl.DataFrame({"area_id": [1], "name_en": ["Marina"]}),
        aliases=pl.DataFrame(schema={"alias_key": pl.Utf8, "area_id": pl.Int64}),
    )


@pytest.fixture
def patched(monkeypatch, homes, tmp_path):
    monkeypatch.setattr(drift, "load_homes", lambda settings, config: homes)
    monkeypatch.setattr(drift, "load_champion", lambda name: (None, None))
    return tmp_path


def test_report_writes_json_and_html(patched):
    result = drift.report(object(), months=3, out_dir=patched, today=date(2026, 9, 17))
    json_path = patched / "drift_2026-09-17.json"
    html_path = patched / "drift_2026-09-17.html"
    assert result.json_path == json_path and result.html_path == html_path
    payload = json.loads(json_path.read_text("utf-8"))
    assert payload["windows"]["reference"] == {
        "start": "2015-01-01",
        "end": "2022-07-01",
        "rows": 3000,
    }
    current = payload["windows"]["current"]
    assert current["end"] == "2023-03-30" and current["rows"] == 800  # dirty rows excluded
    features = {row["feature"]: row for row in payload["features"]}
    assert set(features) == {
        "area_sqm", "bedrooms", "price_per_sqm", "reg_type", "sub_kind", "area_id",
    }  # fmt: skip
    assert features["area_sqm"]["flagged"] is True
    assert features["bedrooms"]["flagged"] is False
    assert payload["flagged"] == [n for n, r in features.items() if r["flagged"]]
    assert payload["forecast"]["status"] == "no champion"
    assert payload["search"] is None
    html = html_path.read_text("utf-8")
    assert html.startswith("<!DOCTYPE html>")
    assert "area_sqm" in html and "<style>" in html
    assert "http://" not in html and "https://" not in html  # self-contained


def test_report_reads_search_log(patched):
    log = patched / "api.log"
    log.write_text(
        json.dumps({"logger": "api.search", "q": "marina <b>view</b>", "results": 0}) + "\n",
        "utf-8",
    )
    result = drift.report(object(), months=3, log_file=log, out_dir=patched, today=date(2026, 1, 2))
    assert result.payload["search"]["zero_result_rate"] == 1.0
    assert "<b>view</b>" not in result.html_path.read_text("utf-8")  # values are escaped


def test_current_window_follows_months(patched):
    result = drift.report(object(), months=1, out_dir=patched, today=date(2026, 1, 2))
    assert result.payload["windows"]["current"]["start"] == "2023-02-28"
    assert result.payload["windows"]["current"]["rows"] < 800


def _model(test_end: str, mape: float = 0.08):
    return SimpleNamespace(
        metadata={"test_end": test_end, "gate": {"model_mape": mape}},
        predict_growth=lambda frame: np.zeros(frame.height),
    )


def test_forecast_section_no_newer_resolved_targets(monkeypatch, homes):
    monkeypatch.setattr(drift, "load_champion", lambda name: (_model("2023-01-01"), "4"))
    section = drift.forecast_section(homes)
    assert section["status"] == "no newer resolved targets"
    assert section["version"] == "4"
    assert section["gate_mape"] == 0.08
    assert section["test_end"] == "2023-01-01"


def test_forecast_section_scores_newer_rows(monkeypatch, homes):
    monkeypatch.setattr(drift, "load_champion", lambda name: (_model("2016-06-01"), "2"))
    scored = pl.DataFrame(
        {
            "instance_date": [date(2016, 6, 1), date(2016, 7, 1), date(2016, 1, 1)],
            "growth_3m": [0.0, np.log(1.25), 0.0],
        }
    )
    monkeypatch.setattr(drift, "resolved_frame", lambda rows: scored)
    section = drift.forecast_section(homes)
    assert section["status"] == "scored"
    assert section["rows"] == 2  # the 2016-01-01 row is inside the model's own test window
    # APE of predicting growth 0: 0 and |exp(-ln 1.25) - 1| = 0.2
    assert section["mape"] == pytest.approx(0.1)
    assert section["mape_vs_gate"] == pytest.approx(0.1 / 0.08)


def test_forecast_section_handles_loader_errors(monkeypatch, homes):
    def boom(name):
        raise RuntimeError("mlflow unreachable")

    monkeypatch.setattr(drift, "load_champion", boom)
    section = drift.forecast_section(homes)
    assert section["status"] == "unavailable"
    assert "mlflow unreachable" in section["reason"]
