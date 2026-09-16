import dataclasses
import logging

import numpy as np
import polars as pl
import pytest

from search.config import FEATURES, TRUST_FEATURES, SearchConfig
from search.ranker import (
    RANKER_CLASSES,
    LGBMRanker,
    XGBRanker,
    load_champion,
    load_ranker,
    log_ranker,
)
from search.train import (
    fit_ablation,
    gate_passes,
    mean_ndcg,
    register_ranker,
    train_rankers,
    tune_ranker,
)

CONFIG = dataclasses.replace(
    SearchConfig(), device="cpu", max_rounds=60, early_stopping_rounds=10, n_trials=2
)


def learnable_table(n_queries: int = 90, per_query: int = 12, seed: int = 0) -> pl.DataFrame:
    """Grades follow area_match and beds_diff; fused_pos is shuffled, so order must be learned."""
    rng = np.random.default_rng(seed)
    rows = []
    for query_id in range(1, n_queries + 1):
        split = {3: "tune", 4: "report"}.get(query_id % 5, "train")  # 60 / 20 / 20
        positions = rng.permutation(per_query) + 1
        for index in range(per_query):
            area = float(rng.random() < 0.5)
            beds = float(rng.integers(0, 3))
            grade = int(area * 2 + (beds == 0) * 1) if query_id % 7 else 0
            features = {name: float(rng.random()) for name in FEATURES}
            features.update(area_match=area, beds_diff=beds)
            rows.append(
                {
                    "query_id": query_id,
                    "split": split,
                    "kind": "no_match" if query_id % 7 == 0 else "specified",
                    "listing_id": query_id * 100 + index,
                    "grade": grade,
                    "fused_pos": int(positions[index]),
                    **features,
                }
            )
    return pl.DataFrame(rows).sort("query_id", "fused_pos")


TABLE = learnable_table()
TRAIN = TABLE.filter(pl.col("split") == "train")
TUNE = TABLE.filter(pl.col("split") == "tune")
FUSED = TUNE.with_columns((-pl.col("fused_pos")).alias("score"))


def _fused_ndcg():
    from search.metrics import per_query_metrics, summarize

    return summarize(per_query_metrics(FUSED, "score"))["ndcg_at_10"]


@pytest.mark.parametrize("kind", ["xgboost", "lightgbm"])
def test_each_ranker_learns_the_signal(kind):
    params = {"max_depth": 3} if kind == "xgboost" else {"num_leaves": 7, "min_data_in_leaf": 5}
    ranker = RANKER_CLASSES[kind].fit(TRAIN, TUNE, FEATURES, params, CONFIG)
    assert ranker.kind == kind and ranker.features == FEATURES
    assert ranker.best_iteration >= 0
    scores = ranker.score(TUNE)
    assert scores.shape == (TUNE.height,) and np.isfinite(scores).all()
    assert mean_ndcg(ranker, TUNE, 10) > _fused_ndcg() + 0.1
    importance = ranker.importance()
    assert importance.columns == ["feature", "gain"]
    assert importance.sort("gain", descending=True)["feature"][0] in {"area_match", "beds_diff"}


@pytest.mark.parametrize("cls", [XGBRanker, LGBMRanker])
def test_save_and_load_round_trip(cls, tmp_path):
    ranker = cls.fit(TRAIN, TUNE, FEATURES, {}, CONFIG)
    loaded = load_ranker(ranker.save(tmp_path / "ranker"))
    assert type(loaded) is cls and loaded.best_iteration == ranker.best_iteration
    assert loaded.params == ranker.params
    np.testing.assert_allclose(loaded.score(TUNE), ranker.score(TUNE), rtol=1e-6)


def test_tune_ranker_returns_the_best_trial_deterministically():
    first = tune_ranker("lightgbm", TRAIN, TUNE, FEATURES, CONFIG, n_trials=3)
    again = tune_ranker("lightgbm", TRAIN, TUNE, FEATURES, CONFIG, n_trials=3)
    ranker, ndcg, params = first
    assert ndcg == pytest.approx(mean_ndcg(ranker, TUNE, 10))
    assert params == again[2] and ndcg == pytest.approx(again[1])
    assert {"num_leaves", "learning_rate"} <= set(params)


def test_train_rankers_picks_the_higher_tune_ndcg():
    result = train_rankers(TABLE, CONFIG, n_trials=2)
    assert set(result.rankers) == {"xgboost", "lightgbm"}
    best = max(result.tune_ndcg, key=result.tune_ndcg.get)
    assert result.winner == best
    assert set(result.best_params) == {"xgboost", "lightgbm"}


def test_the_report_split_never_reaches_training(monkeypatch):
    seen = []
    real_fit = XGBRanker.fit.__func__

    def spy(cls, train, tune, features, params, config):
        seen.append(set(train["split"]) | set(tune["split"]))
        return real_fit(cls, train, tune, features, params, config)

    monkeypatch.setattr(XGBRanker, "fit", classmethod(spy))
    train_rankers(TABLE, CONFIG, n_trials=1)
    assert seen and all("report" not in splits for splits in seen)


def test_ablation_drops_the_trust_features():
    result = train_rankers(TABLE, CONFIG, n_trials=1)
    ablation = fit_ablation(TABLE, result, CONFIG)
    assert ablation.kind == result.winner
    assert ablation.features == tuple(f for f in FEATURES if f not in TRUST_FEATURES)
    assert ablation.score(TUNE).shape == (TUNE.height,)


@pytest.mark.parametrize(
    ("ndcg", "ci_low", "baseline", "expected"),
    [(0.80, 0.75, 0.70, True), (0.80, 0.65, 0.70, False), (0.60, 0.55, 0.70, False),
     (0.70, 0.70, 0.70, False), (float("nan"), 0.1, 0.1, False)],
)  # fmt: skip
def test_gate(ndcg, ci_low, baseline, expected):
    assert gate_passes(ndcg, ci_low, baseline) is expected


def test_pyfunc_registration_and_champion_loading(temp_mlflow, tmp_path):
    import mlflow

    ranker = XGBRanker.fit(TRAIN, TUNE, FEATURES, {"max_depth": 3}, CONFIG)
    mlflow.create_experiment(
        "search-ranking-test", artifact_location=temp_mlflow["artifact_location"]
    )  # never the default ./mlruns
    mlflow.set_experiment("search-ranking-test")
    with mlflow.start_run():
        uri = log_ranker(ranker, tmp_path / "ranker")
    loaded = mlflow.pyfunc.load_model(uri)
    predicted = np.asarray(loaded.predict(TUNE.select(FEATURES).to_pandas()))
    np.testing.assert_allclose(predicted, ranker.score(TUNE), rtol=1e-6)

    config = dataclasses.replace(CONFIG, ranker_name="search-ranker-test")
    with mlflow.start_run():
        version = register_ranker(ranker, config, tmp_path / "registered")
    assert version == "1"
    champion = load_champion("models:/search-ranker-test@champion")
    assert champion is not None and champion.kind == "xgboost"
    np.testing.assert_allclose(champion.score(TUNE), ranker.score(TUNE), rtol=1e-6)


def test_missing_champion_is_none_with_a_warning(temp_mlflow, caplog):
    with caplog.at_level(logging.WARNING, logger="search.ranker"):
        assert load_champion("models:/no-such-ranker@champion") is None
    assert temp_mlflow["tracking_uri"] in caplog.text
