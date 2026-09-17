"""Static checks: Prometheus / Grafana config parse and use the API's real metric names."""

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MONITORING = ROOT / "monitoring"
API_METRICS = {"api_requests_total", "api_request_seconds", "api_component_up", "api_model_info"}


def test_prometheus_scrapes_api_with_a_key_file():
    config = yaml.safe_load((MONITORING / "prometheus.yml").read_text("utf-8"))
    job = next(j for j in config["scrape_configs"] if j["job_name"] == "dubimator-api")
    assert job["static_configs"][0]["targets"] == ["api:8000"]
    assert job["metrics_path"] == "/metrics"
    assert job["authorization"] == {
        "type": "Bearer",
        "credentials_file": "/etc/prometheus-secrets/api_key",
    }


def test_metric_names_match_the_api():
    source = (ROOT / "api" / "metrics.py").read_text("utf-8")
    for name in API_METRICS:
        assert f'"{name}"' in source


def test_dashboard_panels_use_known_metrics():
    dashboard = json.loads((MONITORING / "grafana" / "dashboards" / "api.json").read_text("utf-8"))
    exprs = [t["expr"] for panel in dashboard["panels"] for t in panel["targets"]]
    used = {m for expr in exprs for m in re.findall(r"\bapi_[a-z_]+", expr)}
    base = {re.sub(r"_(bucket|sum|count)$", "", m) for m in used}
    assert base == API_METRICS
    assert any("histogram_quantile(0.95" in e for e in exprs)
    assert any("histogram_quantile(0.5" in e for e in exprs)
    assert any('status=~"5.."' in e for e in exprs) and any('status=~"4.."' in e for e in exprs)
    titles = {panel["title"] for panel in dashboard["panels"]}
    assert "Model versions" in titles
    for panel in dashboard["panels"]:
        assert panel["datasource"]["uid"] == "prometheus"


def test_grafana_provisioning_points_at_the_mounted_paths():
    datasources = yaml.safe_load(
        (MONITORING / "grafana" / "provisioning" / "datasources" / "prometheus.yml").read_text()
    )
    assert datasources["datasources"][0]["uid"] == "prometheus"
    assert datasources["datasources"][0]["url"] == "http://prometheus:9090"
    providers = yaml.safe_load(
        (MONITORING / "grafana" / "provisioning" / "dashboards" / "dashboards.yml").read_text()
    )
    assert providers["providers"][0]["options"]["path"] == "/var/lib/grafana/dashboards"


def _compose() -> dict:
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text("utf-8"))


def test_tunnels_share_only_the_public_network_with_the_demo():
    services = _compose()["services"]
    assert set(services["demo"]["networks"]) == {"default", "public"}
    for name in ("cloudflared-quick", "cloudflared-named"):
        assert services[name]["networks"] == ["public"]
    for name, service in services.items():
        if name != "demo" and not name.startswith("cloudflared"):
            assert "public" not in service.get("networks", []), name


def test_demo_waits_for_a_healthy_api():
    demo = _compose()["services"]["demo"]
    assert demo["depends_on"]["api"]["condition"] == "service_healthy"


def test_prometheus_prefers_its_own_scrape_key():
    init = _compose()["services"]["prometheus-init"]
    assert set(init["environment"]) == {"PROMETHEUS_API_KEY", "API_KEYS"}
    script = init["command"][-1]
    # PROMETHEUS_API_KEY is read first; the first API_KEYS entry is only the fallback.
    assert script.index('"$$PROMETHEUS_API_KEY"') < script.index('"$$API_KEYS"')
