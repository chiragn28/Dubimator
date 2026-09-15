import polars as pl

from models.price.plots import save_eval_plots


def test_writes_the_four_evaluation_plots(tmp_path):
    frame = pl.DataFrame(
        {
            "actual": [100.0, 200.0, 300.0, 400.0] * 5,
            "predicted": [110.0, 190.0, 330.0, 380.0] * 5,
            "segment": ["unit_ready_built_up", "villa_ready_plot"] * 10,
            "loc_level": [0, 1, 2, 3] * 5,
        }
    )
    paths = save_eval_plots(frame, {"prior_building": 3.0, "log_area_sqm": 1.0}, tmp_path / "plots")
    assert sorted(path.name for path in paths) == [
        "error_by_loc_level.png", "feature_importance.png",
        "pred_vs_actual.png", "residuals_by_segment.png",
    ]  # fmt: skip
    assert all(path.stat().st_size > 0 for path in paths)
