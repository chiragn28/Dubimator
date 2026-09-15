import json

from models.price import __main__ as cli
from models.price.predictor import PricePredictor
from models.price.train import TrainingSummary

METRICS = {
    f"{set_name}.all.{metric}": value
    for set_name in ("val", "test_clean", "test_honest")
    for metric, value in (("mdape", 0.12), ("ppe10", 0.45), ("ppe20", 0.75), ("rmse_log", 0.2))
}


def summary(gate_passed=True):
    return TrainingSummary(
        device="cpu",
        rows={"fit": 10, "val": 3},
        drop_counts={},
        metrics={"b0-comps": METRICS, "xgb-champion-eval": METRICS},
        seconds={"total": 1.0},
        gate_passed=gate_passed,
        registered_version="3" if gate_passed else None,
    )


def test_train_loads_dotenv_before_building_db_settings(monkeypatch, capsys):
    monkeypatch.delenv("POSTGRES_PORT", raising=False)
    monkeypatch.setattr(cli, "load_dotenv", lambda: monkeypatch.setenv("POSTGRES_PORT", "6543"))
    seen = {}

    def fake_run(settings, config, **kwargs):
        seen.update(port=settings.port, trials=config.n_trials, kwargs=kwargs)
        return summary()

    monkeypatch.setattr(cli, "run_training", fake_run)
    assert cli.main(["train", "--trials", "3", "--device", "cpu", "--no-register"]) == 0
    assert seen == {"port": 6543, "trials": 3, "kwargs": {"device": "cpu", "register": False}}
    out = capsys.readouterr().out
    assert "xgb-champion-eval" in out and "12.00%" in out
    assert "version 3" in out


def test_train_exits_2_when_the_gate_fails(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "run_training", lambda *a, **k: summary(gate_passed=False))
    assert cli.main(["train"]) == 2
    assert "Acceptance gate failed" in capsys.readouterr().err


def test_train_exits_1_on_errors(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)

    def boom(*args, **kwargs):
        raise RuntimeError("database unreachable")

    monkeypatch.setattr(cli, "run_training", boom)
    assert cli.main(["train"]) == 1
    assert "Training failed: RuntimeError: database unreachable" in capsys.readouterr().err


def test_predict_prints_the_estimate_as_json(monkeypatch, capsys, tiny_bundle):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "_load_champion", lambda: PricePredictor(tiny_bundle(y_hat=0.0)))
    code = cli.main(
        ["predict", "--area", "Dubai Marina", "--kind", "apartment", "--status", "ready",
         "--size", "100", "--bedrooms", "2", "--parking", "yes", "--asking", "1100000"]
    )  # fmt: skip
    assert code == 0
    estimate = json.loads(capsys.readouterr().out)
    assert round(estimate["estimate_aed"]) == 1_000_000
    assert estimate["market_label"] == "fair"


def test_predict_reports_invalid_input(monkeypatch, capsys, tiny_bundle):
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "_load_champion", lambda: PricePredictor(tiny_bundle(y_hat=0.0)))
    code = cli.main(["predict", "--area", "Atlantis", "--kind", "apartment", "--status", "ready",
                     "--size", "100"])  # fmt: skip
    assert code == 1
    assert "Invalid input: area:" in capsys.readouterr().err
