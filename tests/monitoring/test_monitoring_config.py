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
    job = next(j for j in config["scrape_configs"] if j["job_name"] == "zestimator-api")
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
