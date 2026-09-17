import sys
from types import SimpleNamespace

from demo import __main__ as launcher


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
    assert command[command.index("--server.port") + 1] == "9000"
    assert command[command.index("--server.headless") + 1] == "true"
    assert calls["check"] is False
    assert calls["dotenv"] is True


def test_launcher_defaults_to_port_8501(monkeypatch):
    monkeypatch.setattr(launcher, "load_dotenv", lambda: None)
    seen = []
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda command, check: seen.append(command) or SimpleNamespace(returncode=0),
    )
    assert launcher.main([]) == 0
    assert "8501" in seen[0]
