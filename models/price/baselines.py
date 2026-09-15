"""B0: index-adjusted comps. B1: LightGBM on the same features (CPU)."""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from models.price.config import TrainConfig

COMPS_LEVELS = (("area_id", "segment", "room_bucket"), ("area_id", "segment"), ("segment",))


def _room_bucket() -> pl.Expr:
    counted = pl.col("room_kind").is_in(["studio", "bedrooms"])
    bedrooms = pl.col("bedrooms").cast(pl.Int64).cast(pl.Utf8)
    return pl.when(counted).then(bedrooms).otherwise(pl.col("room_kind")).alias("room_bucket")


def _weighted_medians(fit: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Per group: the smallest y whose cumulative bulk weight reaches half the group's weight."""
    return (
        fit.sort("y")
        .with_columns(
            pl.col("bulk_weight").cum_sum().over(keys).alias("__cw"),
            pl.col("bulk_weight").sum().over(keys).alias("__tw"),
        )
        .filter(pl.col("__cw") >= pl.col("__tw") / 2)
        .group_by(keys)
        .agg(pl.col("y").min().alias("med"), pl.col("__tw").first().alias("n"))
    )


def comps_b0(fit: pl.DataFrame, target: pl.DataFrame, min_n: float) -> np.ndarray:
    """Weighted-median relative price of comparable sales, falling back to broader groups."""
    fit = fit.with_columns(_room_bucket())
    out = target.with_columns(_room_bucket()).with_row_index("__row")
    for i, keys in enumerate(COMPS_LEVELS):
        table = _weighted_medians(fit, list(keys)).rename({"med": f"__med{i}", "n": f"__n{i}"})
        out = out.join(table, on=list(keys), how="left")
    out = out.sort("__row")
    pick = (
        pl.when(pl.col("__n0") >= min_n)
        .then(pl.col("__med0"))
        .when(pl.col("__n1") >= min_n)
        .then(pl.col("__med1"))
        .otherwise(pl.col("__med2"))
    )
    predicted = out.select(pick.alias("y_hat"))["y_hat"]
    if predicted.null_count():
        raise ValueError("comps baseline has no segment-level median for some rows")
    return predicted.to_numpy()


def _loaded_msvcp140() -> Path | None:
    """Path of the msvcp140.dll loaded in this process (Windows only), or None."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetModuleHandleW.restype = wintypes.HMODULE
    kernel32.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    kernel32.GetModuleFileNameW.restype = wintypes.DWORD
    handle = kernel32.GetModuleHandleW("msvcp140.dll")
    if not handle:
        return None
    buffer = ctypes.create_unicode_buffer(32_768)
    length = kernel32.GetModuleFileNameW(handle, buffer, len(buffer))
    return Path(buffer.value[:length]) if length else None


def _check_msvcp140() -> None:
    """Fail clearly, instead of with an access violation, if pyarrow's msvcp140.dll is loaded."""
    loaded = _loaded_msvcp140()
    if loaded is not None and "pyarrow" in (part.lower() for part in loaded.parts):
        raise RuntimeError(
            f"LightGBM would crash: the msvcp140.dll loaded in this process is pyarrow's older "
            f"bundled copy ({loaded}), because pandas/pyarrow/mlflow was imported before "
            f"models.price could preload the system one. Fix: import models.price before "
            f"pandas/pyarrow/mlflow (see models/price/__init__.py)."
        )


def lightgbm_b1(x_fit, y_fit, w_fit, x_val, y_val, w_val, config: TrainConfig) -> lgb.Booster:
    if sys.platform == "win32":
        _check_msvcp140()
    params = {
        "objective": "regression",
        "learning_rate": config.lgbm_learning_rate,
        "seed": config.seed,
        "verbose": -1,
    }
    train = lgb.Dataset(x_fit, label=y_fit, weight=w_fit)
    valid = lgb.Dataset(x_val, label=y_val, weight=w_val, reference=train)
    return lgb.train(
        params,
        train,
        num_boost_round=config.max_rounds,
        valid_sets=[valid],
        callbacks=[lgb.early_stopping(config.early_stopping_rounds, verbose=False)],
    )
