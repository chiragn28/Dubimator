"""python -m monitoring drift: argument wiring and summary output."""

from pathlib import Path

import pytest

import monitoring.__main__ as cli
from monitoring.drift import DriftReport


@pytest.fixture
def calls(monkeypatch):
    seen = {}

    def fake_report(settings, months, log_file):
        seen.update(settings=settings, months=months, log_file=log_file)
        payload = {
            "data_end": "2023-03-30",
            "windows": {
                "reference": {"start": "2015-01-01", "end": "2022-07-01", "rows": 10},
                "current": {"start": "2022-12-30", "end": "2023-03-30", "rows": 5},
            },
            "features": [
                {"feature": "area_sqm", "kind": "numeric", "psi": 0.31, "ks": 0.2, "flagged": True}
            ],
            "flagged": ["area_sqm"],
            "forecast": {"status": "no newer resolved targets"},
            "search": None,
        }
        return DriftReport(payload, Path("out/drift.json"), Path("out/drift.html"))

    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli.DbSettings, "from_env", classmethod(lambda cls: "SETTINGS"))
    monkeypatch.setattr(cli, "report", fake_report)
    return seen


def test_drift_defaults(calls, capsys):
    assert cli.main(["drift"]) == 0
    assert calls == {"settings": "SETTINGS", "months": 3, "log_file": None}
    out = capsys.readouterr().out
    assert "area_sqm" in out and "FLAG" in out
    assert "no newer resolved targets" in out
    assert "drift.json" in out and "drift.html" in out


def test_drift_options(calls, tmp_path):
    log = tmp_path / "api.log"
    assert cli.main(["drift", "--months", "6", "--log-file", str(log)]) == 0
    assert calls["months"] == 6 and calls["log_file"] == log


def test_months_must_be_positive(calls):
    with pytest.raises(SystemExit):
        cli.main(["drift", "--months", "0"])


def test_command_is_required():
    with pytest.raises(SystemExit):
        cli.main([])
