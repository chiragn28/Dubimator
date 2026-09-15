import dataclasses
import json
import math
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from models.price.config import FEATURES
from models.price.features import LocationPriors, add_location_keys
from models.price.predictor import (
    PriceInputError,
    PricePredictor,
    PriceRequest,
    load_bundle,
    save_bundle,
)

REQ = {"area": "Dubai Marina", "property_kind": "apartment", "status": "ready", "size_sqm": 100.0}


def predict(bundle, **changes):
    return PricePredictor(bundle).predict_one({**REQ, **changes})


def test_estimate_ranges_and_metadata(tiny_bundle):
    estimate = predict(tiny_bundle(y_hat=0.0))
    assert estimate.estimate_aed == pytest.approx(1_000_000.0)
    assert estimate.price_per_sqm_aed == pytest.approx(10_000.0)
    assert estimate.range_80 == pytest.approx((1e6 * math.exp(-0.15), 1e6 * math.exp(0.15)))
    assert estimate.range_95 == pytest.approx((1e6 * math.exp(-0.3), 1e6 * math.exp(0.3)))
    assert (estimate.location_level, estimate.confidence) == ("area", "medium")
    assert estimate.as_of == date(2023, 3, 17)
    assert estimate.model_version == "run-abc"
    assert estimate.flags == []
    assert estimate.market_label is None and estimate.asking_vs_estimate_pct is None


def test_known_building_resolves_at_building_level(tiny_bundle):
    estimate = predict(tiny_bundle(y_hat=0.0), project="Marina Gate", building="MARINA GATE 1")
    assert (estimate.location_level, estimate.confidence) == ("building", "high")
    assert estimate.flags == []


def test_building_without_project_infers_its_only_project(tiny_bundle):
    estimate = predict(tiny_bundle(y_hat=0.0), building="Marina Gate 1")
    assert (estimate.location_level, estimate.confidence) == ("building", "high")
    assert estimate.flags == []


def _priors_with_a_shared_building_name():
    """'Tower One' sold under two projects in area 10, so the name alone is ambiguous."""
    fit = add_location_keys(
        pl.DataFrame(
            {
                "property_type": ["unit"] * 10,
                "area_id": [10] * 10,
                "project_name": ["Alpha"] * 5 + ["Beta"] * 5,
                "building_name": ["Tower One"] * 10,
                "y": [0.1] * 5 + [0.3] * 5,
                "bulk_weight": [1.0] * 10,
            }
        )
    )
    return LocationPriors.fit(fit, shrink_k=10.0, min_level_n=3.0)


def test_building_under_several_projects_is_not_guessed(tiny_bundle):
    bundle = dataclasses.replace(
        tiny_bundle(y_hat=0.0), priors=_priors_with_a_shared_building_name()
    )
    estimate = predict(bundle, building="Tower One")
    assert estimate.location_level != "building"
    assert estimate.flags == ["location_fallback"]
    with_project = predict(bundle, project="Beta", building="Tower One")
    assert (with_project.location_level, with_project.flags) == ("building", [])


def test_unknown_building_falls_back_and_is_flagged(tiny_bundle):
    estimate = predict(tiny_bundle(y_hat=0.0), project="Marina Gate", building="Marina Gate 9")
    assert (estimate.location_level, estimate.confidence) == ("project", "high")
    assert estimate.flags == ["location_fallback"]


def test_alias_resolves_and_an_area_without_sales_is_city_level(tiny_bundle):
    estimate = predict(tiny_bundle(y_hat=0.0), area="JVC")
    assert (estimate.location_level, estimate.confidence) == ("city", "low")
    assert estimate.estimate_aed > 0


def test_area_id_can_be_given_directly(tiny_bundle):
    bundle = tiny_bundle(y_hat=0.0)
    request = {**REQ, "area": None, "area_id": 10}
    assert PricePredictor(bundle).predict_one(request).location_level == "area"
    with pytest.raises(PriceInputError) as info:
        PricePredictor(bundle).predict_one({**request, "area_id": 999})
    assert info.value.field == "area_id"


@pytest.mark.parametrize(
    ("area", "message"), [("Atlantis", "unknown area"), ("Mushrif", "ambiguous")]
)
def test_unknown_and_ambiguous_area_names_are_rejected(tiny_bundle, area, message):
    with pytest.raises(PriceInputError, match=message) as info:
        predict(tiny_bundle(y_hat=0.0), area=area)
    assert info.value.field == "area"


@pytest.mark.parametrize("changes", [{"area_id": 10}, {"area": None}])
def test_exactly_one_of_area_and_area_id(tiny_bundle, changes):
    with pytest.raises(PriceInputError, match="exactly one of area or area_id"):
        predict(tiny_bundle(y_hat=0.0), **changes)


def test_implausible_predictions_are_clipped_and_flagged(tiny_bundle):
    high = predict(tiny_bundle(y_hat=5.0))
    assert high.estimate_aed == pytest.approx(1e6 * math.exp(0.5))  # area 10 bound hi = 0.5
    assert high.flags == ["implausible_clipped"]
    low = predict(tiny_bundle(y_hat=-50.0))
    assert low.estimate_aed == pytest.approx(1e6 * math.exp(-0.5))
    assert low.estimate_aed > 0
    villa = predict(
        tiny_bundle(y_hat=5.0), area="JVC", property_kind="villa", size_basis="plot", size_sqm=500.0
    )
    assert villa.estimate_aed == pytest.approx(10_000.0 * 500.0 * math.exp(1.0))  # segment bound


def test_market_labels_at_the_interval_edges(tiny_bundle):
    bundle = tiny_bundle(y_hat=0.0)
    low, high = predict(bundle).range_80
    assert predict(bundle, asking_price_aed=high).market_label == "fair"
    assert predict(bundle, asking_price_aed=low).market_label == "fair"
    assert predict(bundle, asking_price_aed=high * 1.01).market_label == "above_market"
    below = predict(bundle, asking_price_aed=low * 0.99)
    assert below.market_label == "below_market"
    assert predict(bundle, asking_price_aed=1_100_000.0).asking_vs_estimate_pct == 10.0


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"size_sqm": 11.0}, "size_sqm"),
        ({"size_sqm": 2_500.0}, "size_sqm"),
        ({"size_sqm": -5.0}, "size_sqm"),
        ({"size_basis": "plot"}, "size_basis"),
        ({"bedrooms": 9}, "bedrooms"),
        ({"is_penthouse": True, "property_kind": "villa", "size_sqm": 300.0}, "is_penthouse"),
        ({"property_kind": "villa", "status": "off_plan", "size_basis": "plot",
          "size_sqm": 400.0}, "status"),
        ({"property_kind": "castle"}, "property_kind"),
        ({"surprise": 1}, "surprise"),
    ],
)  # fmt: skip
def test_invalid_requests_raise_price_input_error(tiny_bundle, changes, field):
    with pytest.raises(PriceInputError) as info:
        predict(tiny_bundle(y_hat=0.0), **changes)
    assert info.value.field == field


@pytest.mark.parametrize(
    ("field", "value"),
    [("asking_price_aed", math.inf), ("asking_price_aed", math.nan), ("size_sqm", math.nan)],
)
def test_non_finite_numbers_are_rejected(tiny_bundle, field, value):
    with pytest.raises(PriceInputError) as info:
        predict(tiny_bundle(y_hat=0.0), **{field: value})
    assert info.value.field == field


def test_missing_required_field_names_the_field():
    with pytest.raises(PriceInputError) as info:
        PriceRequest.parse({"area": "JBR", "status": "ready", "size_sqm": 80.0})
    assert info.value.field == "property_kind"
    assert str(info.value).startswith("property_kind: ")


def test_unusual_size_is_flagged_not_rejected(tiny_bundle):
    estimate = predict(tiny_bundle(y_hat=0.0), size_sqm=25.0)
    assert estimate.flags == ["unusual_size"]
    assert estimate.estimate_aed == pytest.approx(250_000.0)


def test_penthouse_and_studio_requests_are_accepted(tiny_bundle):
    bundle = tiny_bundle(y_hat=0.0)
    assert predict(bundle, is_penthouse=True, bedrooms=3).estimate_aed > 0
    assert predict(bundle, bedrooms=0).estimate_aed > 0


TRAINING_ONLY_MODULES = ("sklearn", "optuna", "lightgbm", "matplotlib", "psycopg2")


def test_serving_imports_no_training_only_packages():
    """The pyfunc's requirements omit these, so importing the predictor must not need them.

    xgboost imports sklearn opportunistically when it is installed (and falls back when it is
    not), so sklearn is blocked outright: the import must still succeed without it.
    """
    repo_root = Path(__file__).resolve().parents[3]
    code = (
        "import sys\n"
        "sys.modules['sklearn'] = None  # any 'import sklearn' now raises ImportError\n"
        "import models.price.predictor\n"
        f"loaded = [m for m in {TRAINING_ONLY_MODULES!r} if sys.modules.get(m) is not None]\n"
        "print('loaded:', loaded)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "loaded: []" in result.stdout, result.stdout


def test_bundle_round_trips_through_a_directory(tiny_bundle, tmp_path):
    bundle = tiny_bundle()  # real booster
    save_bundle(bundle, tmp_path / "model")
    assert (tmp_path / "model" / "booster.json").is_file()
    loaded = load_bundle(tmp_path / "model")
    assert loaded.supported_segments == bundle.supported_segments
    assert loaded.conformal == bundle.conformal
    assert loaded.data_end == bundle.data_end
    direct = PricePredictor(bundle).predict_one(REQ)
    reloaded = PricePredictor.from_dir(tmp_path / "model").predict_one(REQ)
    assert reloaded.estimate_aed == pytest.approx(direct.estimate_aed)
    assert reloaded.model_version == "run-abc"


def test_bundle_records_its_features_and_refuses_a_different_list(tiny_bundle, tmp_path):
    save_bundle(tiny_bundle(), tmp_path / "model")
    meta_path = tmp_path / "model" / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["features"] == list(FEATURES)
    meta["features"] = list(reversed(FEATURES))
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="features"):
        load_bundle(tmp_path / "model")
