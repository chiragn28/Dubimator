"""scripts/tunnel_url.py: find the quick-tunnel URL in `docker compose logs` output."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "tunnel_url.py"
spec = importlib.util.spec_from_file_location("tunnel_url", SCRIPT)
tunnel_url = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tunnel_url)

LOGS = """\
cloudflared-1  | 2026-09-17T01:00:00Z INF Thank you for trying Cloudflare Tunnel.
cloudflared-1  | 2026-09-17T01:00:00Z INF +----------------------------------------+
cloudflared-1  | 2026-09-17T01:00:00Z INF |  https://calm-river-7f3a.trycloudflare.com  |
cloudflared-1  | 2026-09-17T01:00:01Z INF see https://developers.cloudflare.com/x
cloudflared-1  | 2026-09-17T02:00:00Z INF |  https://second-one.trycloudflare.com  |
"""


def test_extract_url_takes_the_first_match():
    assert tunnel_url.extract_url(LOGS) == "https://calm-river-7f3a.trycloudflare.com"


def test_extract_url_none_when_absent():
    assert tunnel_url.extract_url("INF Registered tunnel connection\n") is None
    assert tunnel_url.extract_url("https://api.trycloudflare.com.evil.example") is None


def _fake_run(stdout: str, returncode: int = 0, seen: list | None = None):
    def run(cmd, **kwargs):
        if seen is not None:
            seen.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")

    return run


def test_main_prints_url(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(tunnel_url.subprocess, "run", _fake_run(LOGS, seen=seen))
    assert tunnel_url.main([]) == 0
    assert capsys.readouterr().out.strip() == "https://calm-river-7f3a.trycloudflare.com"
    assert seen == [["docker", "compose", "logs", "--no-color", "cloudflared-quick"]]


def test_main_service_option(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(tunnel_url.subprocess, "run", _fake_run(LOGS, seen=seen))
    assert tunnel_url.main(["--service", "other"]) == 0
    assert seen[0][-1] == "other"


def test_main_reports_missing_url(monkeypatch, capsys):
    monkeypatch.setattr(tunnel_url.subprocess, "run", _fake_run("nothing yet\n"))
    assert tunnel_url.main([]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "no trycloudflare.com URL" in captured.err


def test_main_reports_compose_failure(monkeypatch, capsys):
    monkeypatch.setattr(tunnel_url.subprocess, "run", _fake_run("", returncode=1))
    assert tunnel_url.main([]) == 1
    assert "docker compose logs" in capsys.readouterr().err


@pytest.mark.parametrize("error", [FileNotFoundError("docker")])
def test_main_reports_missing_docker(monkeypatch, capsys, error):
    def run(cmd, **kwargs):
        raise error

    monkeypatch.setattr(tunnel_url.subprocess, "run", run)
    assert tunnel_url.main([]) == 1
    assert "docker" in capsys.readouterr().err
