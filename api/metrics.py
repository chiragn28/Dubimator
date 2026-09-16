"""Prometheus metrics on a private registry (one per app, so tests never collide)."""

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client.exposition import generate_latest

BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


class Metrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.requests = Counter("api_requests_total", "HTTP requests",
                                ("route", "method", "status"), registry=self.registry)  # fmt: skip
        self.latency = Histogram("api_request_seconds", "Request latency",
                                 ("route",), buckets=BUCKETS, registry=self.registry)  # fmt: skip
        self.component_up = Gauge("api_component_up", "1 when the component loaded",
                                  ("component",), registry=self.registry)  # fmt: skip
        self.model_info = Gauge("api_model_info", "Loaded model version",
                                ("component", "version"), registry=self.registry)  # fmt: skip

    def observe(self, route: str, method: str, status: int, seconds: float) -> None:
        self.requests.labels(route=route, method=method, status=str(status)).inc()
        self.latency.labels(route=route).observe(seconds)

    def set_components(self, state) -> None:
        self.model_info.clear()
        for name, info in state.status().items():
            self.component_up.labels(component=name).set(1.0 if info["up"] else 0.0)
            if info["up"]:
                self.model_info.labels(component=name, version=info["version"]).set(1.0)

    def render(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
