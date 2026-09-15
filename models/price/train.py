"""Phase 3 orchestration: load → features → baselines → tuning → evaluation → gate →
production refit → MLflow registration."""

import json
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import mlflow
import numpy as np
import polars as pl

from ingestion.config import DbSettings
from models.price.baselines import comps_b0, lightgbm_b1
from models.price.boosting import fit_xgb, make_dmatrix, predict_xgb, resolve_device
from models.price.config import TrainConfig
from models.price.data import load_homes
from models.price.evaluate import coverage, fit_conformal, quantiles_for, sliced_metrics
from models.price.features import (
    FeatureSet,
    build_feature_set,
    feature_frame,
    fit_bounds,
    fit_size_percentiles,
)
from models.price.plots import save_eval_plots
from models.price.predictor import ModelBundle, save_bundle
from models.price.registry import configure, log_metrics, log_price_model, register_champion
from models.price.split import add_bulk_groups, assign_split
from models.price.tune import tune_xgboost

EVAL_SETS = ("val", "test_clean", "test_honest")


@dataclass
class TrainingSummary:
    device: str
    rows: dict[str, int]
    drop_counts: dict[str, int]
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    best_params: dict = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)
    gate_passed: bool = False
    model_uri: str | None = None
    registered_version: str | None = None


def _prices(frame: pl.DataFrame, y_hat) -> np.ndarray:
    return (
        np.exp(np.asarray(y_hat) + frame["market_index"].to_numpy()) * frame["area_sqm"].to_numpy()
    )


def _evaluate(
    fs: FeatureSet, predict: Callable[[pl.DataFrame], np.ndarray]
) -> tuple[dict[str, float], dict[str, pl.DataFrame]]:
    metrics: dict[str, float] = {}
    frames: dict[str, pl.DataFrame] = {}
    for name in EVAL_SETS:
        frame = fs.frames.get(name)
        if frame is None or frame.height == 0:
            continue
        evaluated = frame.select(
            pl.col("price_aed").alias("actual"), "segment", "loc_level", "bulk_group"
        ).with_columns(pl.Series("predicted", _prices(frame, predict(frame))))
        metrics |= sliced_metrics(evaluated, name)
        frames[name] = evaluated
    return metrics, frames


def _coverage_metrics(
    frames: dict[str, pl.DataFrame], conformal: dict[str, dict[str, float]]
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in ("test_clean", "test_honest"):
        frame = frames.get(name)
        if frame is None:
            continue
        for label in ("q80", "q95"):
            q = np.array([quantiles_for(conformal, s)[label] for s in frame["segment"].to_list()])
            predicted = frame["predicted"].to_numpy()
            frame_q = frame.with_columns(
                pl.Series("low", predicted * np.exp(-q)), pl.Series("high", predicted * np.exp(q))
            )
            parts = [("all", frame_q)] + [
                (f"segment.{segment}", frame_q.filter(pl.col("segment") == segment))
                for segment in sorted(frame_q["segment"].unique().to_list())
            ]
            for slice_name, part in parts:
                out[f"{name}.{slice_name}.coverage_{label[1:]}"] = coverage(
                    part["actual"].to_numpy(), part["low"].to_numpy(), part["high"].to_numpy()
                )
    return out


def eval_filters(data_end: date) -> dict[str, pl.Expr]:
    """Evaluation sets. Test stops at data_end (the last clean sale): a stat-excluded sale dated
    later would fall outside the market index's fitted months."""
    split, clean = pl.col("split"), pl.col("is_clean")
    in_test = (split == "test") & (pl.col("instance_date") <= data_end)
    return {
        "val": (split == "val") & clean,
        "test_clean": in_test & clean,
        "test_honest": in_test,
    }


def _conformal_from(frame: pl.DataFrame, min_rows: int) -> dict[str, dict[str, float]]:
    abs_error = np.abs(np.log(frame["predicted"].to_numpy()) - np.log(frame["actual"].to_numpy()))
    return fit_conformal(frame["segment"].to_list(), abs_error, min_rows)


def run_training(
    settings: DbSettings,
    config: TrainConfig,
    device: str = "auto",
    register: bool = True,
    tracking_uri: str | None = None,
    artifact_location: str | None = None,
    log: Callable[[str], None] = print,
) -> TrainingSummary:
    started = time.perf_counter()
    device = resolve_device(device)
    homes = load_homes(settings, config)
    rows = add_bulk_groups(assign_split(homes.rows, config))
    split, clean = pl.col("split"), pl.col("is_clean")
    eval_fs = build_feature_set(
        rows,
        config,
        homes.data_end,
        fit_filter=(split == "train") & clean,
        apply_filters=eval_filters(homes.data_end),
    )
    fit, val = eval_fs.frames["fit"], eval_fs.frames["val"]
    x_fit, x_val = feature_frame(fit, eval_fs.categories), feature_frame(val, eval_fs.categories)
    y_fit, y_val = fit["y"].to_numpy(), val["y"].to_numpy()
    w_val = val["bulk_weight"].to_numpy()  # ruling 3: bulk weight only for validation rows
    summary = TrainingSummary(
        device=device,
        rows={name: frame.height for name, frame in eval_fs.frames.items()},
        drop_counts=homes.drop_counts,
    )
    summary.seconds["prepare"] = time.perf_counter() - started
    log(f"Prepared rows {summary.rows} (device {device}); dropped {homes.drop_counts}")

    configure(config.experiment, tracking_uri, artifact_location)
    base_params = {
        "device": device,
        "data_end": homes.data_end.isoformat(),
        "supported_segments": ",".join(homes.supported_segments),
        **{f"lineage.{key}": value for key, value in homes.lineage.items()},
        **{f"rows.{key}": value for key, value in summary.rows.items()},
    }

    with mlflow.start_run(run_name="b0-comps"):
        mlflow.log_params(base_params)
        metrics, _ = _evaluate(eval_fs, lambda frame: comps_b0(fit, frame, config.comps_min_n))
        log_metrics(metrics)
        summary.metrics["b0-comps"] = metrics
    log(f"b0-comps test_clean MdAPE {summary.metrics['b0-comps'].get('test_clean.all.mdape')}")

    with mlflow.start_run(run_name="b1-lightgbm"):
        mlflow.log_params(base_params)
        clock = time.perf_counter()
        lgbm = lightgbm_b1(x_fit, y_fit, eval_fs.weights, x_val, y_val, w_val, config)
        summary.seconds["b1-lightgbm"] = time.perf_counter() - clock
        metrics, _ = _evaluate(
            eval_fs,
            lambda frame: lgbm.predict(
                feature_frame(frame, eval_fs.categories), num_iteration=lgbm.best_iteration
            ),
        )
        log_metrics(metrics | {"best_iteration": lgbm.best_iteration})
        summary.metrics["b1-lightgbm"] = metrics
    log(
        f"b1-lightgbm test_clean MdAPE {summary.metrics['b1-lightgbm'].get('test_clean.all.mdape')}"
    )

    dtrain = make_dmatrix(x_fit, y_fit, eval_fs.weights)
    dval = make_dmatrix(x_val, y_val, w_val)
    with mlflow.start_run(run_name="xgb-tune"):
        mlflow.log_params(base_params | {"n_trials": config.n_trials})

        def on_trial(number: int, params: dict, value: float, best_iteration: int, seconds: float):
            with mlflow.start_run(run_name=f"trial-{number:03d}", nested=True):
                mlflow.log_params(params)
                log_metrics(
                    {
                        "val_rmse_weighted": value,
                        "best_iteration": best_iteration,
                        "seconds": seconds,
                    }
                )
            log(
                f"  trial {number:3d}: val rmse {value:.5f}, {best_iteration} rounds, {seconds:.1f}s"
            )

        clock = time.perf_counter()
        result = tune_xgboost(dtrain, dval, device, config, on_trial)
        summary.seconds["xgb-tune"] = time.perf_counter() - clock
        mlflow.log_params({f"best.{key}": value for key, value in result.best_params.items()})
        log_metrics(
            {"best_val_rmse_weighted": result.best_value, "seconds": summary.seconds["xgb-tune"]}
        )
    summary.best_params = result.best_params

    with mlflow.start_run(run_name="xgb-champion-eval"):
        mlflow.log_params(base_params | result.best_params)
        booster = fit_xgb(result.best_params, dtrain, dval, device, config)
        metrics, frames = _evaluate(
            eval_fs, lambda frame: predict_xgb(booster, feature_frame(frame, eval_fs.categories))
        )
        val_frame = frames["val"]
        # Val-calibrated quantiles measure the method honestly: their test coverage is reported.
        conformal_val = _conformal_from(val_frame, config.min_conformal_rows)
        metrics |= _coverage_metrics(frames, conformal_val)
        # Shipped quantiles come from the test period's errors. Val also drove early stopping
        # and tuning, so its errors are optimistic; test was never used for fitting or tuning.
        conformal_production = (
            _conformal_from(frames["test_clean"], config.min_conformal_rows)
            if "test_clean" in frames
            else None
        )
        log_metrics(metrics | {"best_iteration": float(booster.best_iteration)})
        summary.metrics["xgb-champion-eval"] = metrics
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            plot_frame = frames.get("test_clean", val_frame)
            save_eval_plots(plot_frame, booster.get_score(importance_type="total_gain"), out)
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
            (out / "conformal_val.json").write_text(json.dumps(conformal_val, indent=2))
            if conformal_production is not None:
                (out / "conformal_production.json").write_text(
                    json.dumps(conformal_production, indent=2)
                )
            mlflow.log_artifacts(tmp)
    log(f"xgb-champion-eval test_clean MdAPE {metrics.get('test_clean.all.mdape')}")

    baseline = summary.metrics["b0-comps"].get("test_clean.all.mdape")
    champion = metrics.get("test_clean.all.mdape")
    summary.gate_passed = config.gate_ratio is None or (
        baseline is not None and champion is not None and champion <= config.gate_ratio * baseline
    )
    if not summary.gate_passed:
        log(f"Acceptance gate failed: champion {champion} vs {config.gate_ratio} x B0 {baseline}")
        summary.seconds["total"] = time.perf_counter() - started
        return summary

    if conformal_production is None:
        raise RuntimeError("no clean test rows: cannot calibrate the production price ranges")
    clock = time.perf_counter()
    prod_fs = build_feature_set(
        rows, config, homes.data_end, fit_filter=clean & (split != "seed"), apply_filters={}
    )
    prod_fit = prod_fs.frames["fit"]
    dall = make_dmatrix(
        feature_frame(prod_fit, prod_fs.categories), prod_fit["y"].to_numpy(), prod_fs.weights
    )
    num_rounds = int(booster.best_iteration) + 1
    headline = {key: value for key, value in metrics.items() if ".all." in key}
    with mlflow.start_run(run_name="xgb-production") as run:
        mlflow.log_params(
            base_params
            | result.best_params
            | {"num_boost_round": num_rounds, "rows.production_fit": prod_fit.height}
        )
        prod_booster = fit_xgb(
            result.best_params, dall, None, device, config, num_rounds=num_rounds
        )
        bounds_area, bounds_segment = fit_bounds(prod_fit, config.bounds_min_area_n)
        bundle = ModelBundle(
            booster=prod_booster,
            index=prod_fs.index,
            priors=prod_fs.priors,
            categories=prod_fs.categories,
            bounds_area=bounds_area,
            bounds_segment=bounds_segment,
            size_percentiles=fit_size_percentiles(prod_fit),
            areas=homes.areas,
            aliases=homes.aliases,
            conformal=conformal_production,
            supported_segments=homes.supported_segments,
            data_end=homes.data_end,
            metadata={
                "model_version": run.info.run_id,
                "lineage": homes.lineage,
                "params": result.best_params,
                "num_boost_round": num_rounds,
                "eval_metrics": headline,
                "device": device,
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "model_dir"
            save_bundle(bundle, model_dir)
            summary.model_uri = log_price_model(model_dir)
        # The eval model's scores, not this model's: prefixed and tagged so they don't read as
        # a test score of the production refit (which has no held-out data).
        mlflow.set_tag("metrics_source", "xgb-champion-eval")
        log_metrics({f"eval.{key}": value for key, value in headline.items()})
    summary.seconds["xgb-production"] = time.perf_counter() - clock
    if register:
        summary.registered_version = register_champion(summary.model_uri, config.model_name)
        log(f"Registered {config.model_name} v{summary.registered_version} as @champion")
    summary.seconds["total"] = time.perf_counter() - started
    return summary
