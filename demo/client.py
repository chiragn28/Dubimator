"""HTTP client the Streamlit demo uses to talk to the Phase 7 API.

The demo never touches the database, MLflow or the model/search/listings
packages directly — every capability goes through this client over HTTP.
"""

from __future__ import annotations

import functools
import os

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


class ApiProblem(Exception):
    """Raised for anything that stops an API call from returning a result."""

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        field: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.field = field
        self.request_id = request_id


class ApiClient:
    """A thin wrapper around `httpx` for the demo's server-side calls.

    The API key lives only here — it is attached as `X-API-Key` on every
    request and is never handed to the browser.
    """

    def __init__(
        self,
        base_url: str,
        key: str | None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._key = key
        self._client = httpx.Client(base_url=self._base_url, transport=transport, timeout=timeout)

    def _request(self, method: str, path: str, *, params=None, json=None) -> dict:
        if not self._key:
            raise ApiProblem(0, "no_key", "DEMO_API_KEY is not set")

        headers = {"X-API-Key": self._key}
        try:
            response = self._client.request(method, path, params=params, json=json, headers=headers)
        except httpx.TransportError:
            # Connect/read/write errors, timeouts and protocol errors all mean the same thing
            # to a visitor: the API isn't answering properly at the configured URL.
            raise ApiProblem(
                0, "unreachable", f"cannot reach the API at {self._base_url}"
            ) from None

        request_id_header = response.headers.get("X-Request-ID")
        try:
            body = response.json()
        except ValueError:
            body = None  # not JSON (e.g. a proxy's HTML error page)

        if response.status_code // 100 != 2:
            envelope = body if isinstance(body, dict) else {}
            error = envelope.get("error")
            if not isinstance(error, dict):
                error = {}
            raise ApiProblem(
                response.status_code,
                error.get("code") or "http_error",
                error.get("message") or f"HTTP {response.status_code}",
                field=error.get("field"),
                request_id=envelope.get("request_id") or request_id_header,
            )

        if not isinstance(body, dict):
            raise ApiProblem(
                response.status_code,
                "bad_response",
                f"the API at {self._base_url} sent an unexpected response",
                request_id=request_id_header,
            )
        return body

    def _bytes(self, path: str) -> bytes:
        """GET `path` and return the raw body. `_request` insists on a JSON object."""
        if not self._key:
            raise ApiProblem(0, "no_key", "DEMO_API_KEY is not set")
        headers = {"X-API-Key": self._key}
        try:
            response = self._client.request("GET", path, headers=headers)
        except httpx.TransportError:
            raise ApiProblem(
                0, "unreachable", f"cannot reach the API at {self._base_url}"
            ) from None
        if response.status_code // 100 != 2:
            body = {}
            try:
                parsed = response.json()
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                body = parsed
            error = body.get("error") if isinstance(body.get("error"), dict) else {}
            raise ApiProblem(
                response.status_code,
                error.get("code") or "http_error",
                error.get("message") or f"HTTP {response.status_code}",
                request_id=body.get("request_id") or response.headers.get("X-Request-ID"),
            )
        return response.content

    def ready(self) -> dict:
        return self._request("GET", "/v1/ready")

    def price(self, body: dict) -> dict:
        return self._request("POST", "/v1/price", json=body)

    def forecast(self, body: dict) -> dict:
        return self._request("POST", "/v1/forecast", json=body)

    def search(self, q: str, k: int = 10) -> dict:
        return self._request("GET", "/v1/search", params={"q": q, "k": k})

    def listing_flags(self, listing_id) -> dict:
        return self._request("GET", f"/v1/listings/{listing_id}/flags")

    def listing_photos(self, listing_id) -> dict:
        return self._request("GET", f"/v1/listings/{listing_id}/photos")

    def photo(self, photo_id) -> bytes | None:
        """A photo's JPEG bytes, or None when this deployment has no image for it.

        Photos are decoration: a missing one must never break a page, so a 404 (no corpus
        images on the server) comes back as None instead of an `ApiProblem`.
        """
        try:
            return self._bytes(f"/v1/photos/{photo_id}")
        except ApiProblem as problem:
            if problem.status == 404:
                return None
            raise

    def check_listing(self, body: dict) -> dict:
        return self._request("POST", "/v1/listings/check", json=body)

    def areas(self) -> dict:
        return self._request("GET", "/v1/areas")

    def area_history(self, area_id) -> dict:
        return self._request("GET", f"/v1/areas/{area_id}/history")


def client_from_env(environ: dict | None = None) -> ApiClient:
    """Build an `ApiClient` from `DEMO_API_URL` and `DEMO_API_KEY`."""
    env = environ if environ is not None else os.environ
    base_url = env.get("DEMO_API_URL", DEFAULT_BASE_URL)
    key = env.get("DEMO_API_KEY")
    return ApiClient(base_url=base_url, key=key)


@functools.lru_cache(maxsize=1)
def get_client() -> ApiClient:
    """The cached client the demo's pages call. Tests monkeypatch this."""
    return client_from_env()
