import sys
from types import SimpleNamespace

from demo import __main__ as launcher


def _capture_run(monkeypatch):
    seen = []
    monkeypatch.setattr(launcher, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda command, check: seen.append(command) or SimpleNamespace(returncode=0),
    )
    return seen


def _option(command, name):
    return command[command.index(name) + 1]


def test_launcher_runs_streamlit_on_the_requested_port(monkeypatch):
    calls = {}
    monkeypatch.setattr(launcher, "load_dotenv", lambda: calls.setdefault("dotenv", True))

    def fake_run(command, check):
        calls["command"] = command
        calls["check"] = check
        return SimpleNamespace(returncode=3)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    assert launcher.main(["--port", "9000"]) == 3
    command = calls["command"]
    assert command[:4] == [sys.executable, "-m", "streamlit", "run"]
    assert command[4].endswith("app.py")
    assert _option(command, "--server.port") == "9000"
    assert _option(command, "--server.headless") == "true"
    assert calls["check"] is False
    assert calls["dotenv"] is True


def test_launcher_defaults_to_port_8501_on_localhost(monkeypatch):
    seen = _capture_run(monkeypatch)
    assert launcher.main([]) == 0
    assert _option(seen[0], "--server.port") == "8501"
    assert _option(seen[0], "--server.address") == "127.0.0.1"


def test_launcher_host_option(monkeypatch):
    seen = _capture_run(monkeypatch)
    assert launcher.main(["--host", "0.0.0.0"]) == 0
    assert _option(seen[0], "--server.address") == "0.0.0.0"
