"""Evaluation plots logged on the champion-eval run."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def save_eval_plots(frame: pl.DataFrame, importance: dict[str, float], out_dir: Path) -> list[Path]:
    plt.switch_backend("Agg")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def save(fig, name: str) -> None:
        path = out_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        paths.append(path)

    actual = frame["actual"].to_numpy()
    predicted = frame["predicted"].to_numpy()

    names = sorted(importance, key=importance.get)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(names, [importance[name] for name in names])
    ax.set_xlabel("total gain")
    ax.set_title("Feature importance")
    save(fig, "feature_importance.png")

    log_ratio = np.log(predicted / actual)
    segment_values = frame["segment"].to_numpy()
    segments = sorted(frame["segment"].unique().to_list())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot([log_ratio[segment_values == s] for s in segments], showfliers=False)
    ax.set_xticks(range(1, len(segments) + 1), segments, rotation=20)
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.set_ylabel("ln(predicted / actual)")
    ax.set_title("Residuals by segment")
    save(fig, "residuals_by_segment.png")

    ape = np.abs(predicted - actual) / actual
    levels_values = frame["loc_level"].to_numpy()
    levels = sorted(frame["loc_level"].unique().to_list())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        [str(level) for level in levels],
        [float(np.median(ape[levels_values == level])) for level in levels],
    )
    ax.set_xlabel("location level (0 city, 1 area, 2 project, 3 building)")
    ax.set_ylabel("MdAPE")
    ax.set_title("Error by location level")
    save(fig, "error_by_loc_level.png")

    x, y = np.log10(actual), np.log10(predicted)
    fig, ax = plt.subplots(figsize=(6, 6))
    cells = ax.hexbin(x, y, gridsize=60, bins="log", mincnt=1)
    limits = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(limits, limits, color="red", linewidth=0.8)
    ax.set_xlabel("log10 actual price (AED)")
    ax.set_ylabel("log10 predicted price (AED)")
    ax.set_title("Predicted vs actual")
    fig.colorbar(cells, ax=ax)
    save(fig, "pred_vs_actual.png")
    return paths
