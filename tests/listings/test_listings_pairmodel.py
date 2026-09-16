import numpy as np
import polars as pl
import pytest

from listings.config import PAIR_FEATURES, DetectConfig
from listings.detect import DetectionResult, fit_pair_model, price_blind_scores
from listings.pairmodel import PAIR_MODEL_NAME, PairModel, load_pair_model, register_pair_model


def toy_result():
    rng = np.random.default_rng(0)
    rows = 200
    frame = pl.DataFrame({name: rng.random(rows) for name in PAIR_FEATURES})
    labels = frame["text_cosine"].to_numpy() > 0.5
    model = fit_pair_model(frame, labels, DetectConfig())
    return frame, DetectionResult(frame, 0.5, model, PAIR_FEATURES, {"flagged": 1.0})


def test_scores_match_the_detect_pipeline(tmp_path):
    frame, result = toy_result()
    pair = PairModel(result.model, result.threshold, result.feature_names, detect_run_id=7)
    expected = result.model.predict_proba(frame.select(PAIR_FEATURES).to_numpy())[:, 1]
    assert np.allclose(pair.scores(frame), expected)
    assert np.allclose(pair.price_blind_scores(frame), price_blind_scores(result.model, frame))
    loaded = PairModel.load(pair.save(tmp_path / "pair"))
    assert np.allclose(loaded.scores(frame), expected)
    assert (loaded.threshold, loaded.features, loaded.detect_run_id) == (0.5, PAIR_FEATURES, 7)


def test_register_and_load_the_champion(tmp_path, temp_mlflow):
    frame, result = toy_result()
    version = register_pair_model(
        result, detect_run_id=3, directory=tmp_path / "pair",
        tracking_uri=temp_mlflow["tracking_uri"],
        artifact_location=temp_mlflow["artifact_location"],
    )  # fmt: skip
    assert version == "1"
    loaded = load_pair_model()
    assert loaded is not None and loaded.version == "1"
    assert loaded.detect_run_id == 3
    assert np.allclose(
        loaded.scores(frame), PairModel(result.model, 0.5, PAIR_FEATURES, 3).scores(frame)
    )


def test_load_returns_none_without_a_champion(temp_mlflow):
    assert load_pair_model(f"models:/{PAIR_MODEL_NAME}@champion") is None


def test_scores_reject_missing_features():
    frame, result = toy_result()
    pair = PairModel(result.model, 0.5, PAIR_FEATURES, None)
    with pytest.raises(Exception, match="same_agent"):
        pair.scores(frame.drop("same_agent"))
