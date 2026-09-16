"""Command line: python -m api serve|smoke.

Spec: docs/superpowers/specs/2026-09-17-phase7-api-design.md ("Endpoints", "Testing").
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from api.app import create_app
from api.settings import ApiSettings
from api.state import default_loaders

# A module global so tests can point it at a tmp dir instead of the real `data/api/`.
DATA_DIR = Path("data/api")

# The smoke bodies (spec "Testing"): a Dubai Marina, 85 m², 1-bed, ready apartment for
# price/forecast, and a matching free-text query for search.
PRICE_FORECAST_BODY = {
    "area": "Dubai Marina",
    "property_kind": "apartment",
    "status": "ready",
    "size_sqm": 85,
    "bedrooms": 1,
}
SEARCH_QUERY = "2 bed apartment in dubai marina under 2m"

SAMPLE_LISTING_SQL = """
SELECT listing_id, title, description, asking_price_aed, area_id, building_name,
       project_name, property_type, property_sub_type, reg_type, size_sqm
FROM listings.listings
WHERE listing_id = (SELECT min(listing_id) FROM listings.listings)
"""
SAMPLE_PHOTOS_SQL = (
    "SELECT photo_id FROM listings.listing_photos WHERE listing_id = %s ORDER BY position"
)


def _sample_listing() -> dict | None:
    """The lowest-`listing_id` corpus row, shaped for `/v1/listings/{id}/flags` and
    `/v1/listings/check`. None when the corpus is empty or the row can't be mapped
    (for example an unrecognized `property_sub_type`); either way `smoke` just skips
    the two listing calls and prints why.

    Kept behind this one function so `tests/api/test_api_cli.py` can monkeypatch it
    instead of touching a real database.
    """
    from ingestion.config import DbSettings
    from listings.fraud import _property_kind

    try:
        conn = DbSettings.from_env().connect()
    except Exception as exc:  # noqa: BLE001 — reported as a skip, not a crash
        print(f"sample listing unavailable: {type(exc).__name__}: {exc}")
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(SAMPLE_LISTING_SQL)
            row = cur.fetchone()
            if row is None:
                return None
            (
                listing_id, title, description, asking_price_aed, area_id, building_name,
                project_name, property_type, property_sub_type, reg_type, size_sqm,
            ) = row  # fmt: skip
            cur.execute(SAMPLE_PHOTOS_SQL, (listing_id,))
            photo_ids = [photo_id for (photo_id,) in cur.fetchall()]
        conn.rollback()
        villa = property_type == "villa"
        property_kind = "villa" if villa else _property_kind(property_sub_type)
        return {
            "listing_id": listing_id,
            "title": title,
            "description": description,
            "asking_price_aed": float(asking_price_aed),
            "area_id": area_id,
            "building_name": building_name,
            "project_name": project_name,
            "property_kind": property_kind,
            "status": reg_type,
            "size_sqm": float(size_sqm),
            "photo_ids": photo_ids,
        }
    except Exception as exc:  # noqa: BLE001 — reported as a skip, not a crash
        print(f"sample listing unavailable: {type(exc).__name__}: {exc}")
        return None
    finally:
        conn.close()


def _check_body(sample: dict) -> dict:
    keys = (
        "title", "description", "asking_price_aed", "area_id", "building_name",
        "project_name", "property_kind", "status", "size_sqm", "photo_ids",
    )  # fmt: skip
    return {key: sample[key] for key in keys}


def _test_client(app):
    """A `TestClient`, built the same way `tests/api/api_fixtures.py` builds one — the
    warnings this suppresses are library-internal (see that module's comment) and would
    otherwise fail `tests/api/test_api_cli.py` under `-W error`.
    """
    from starlette.exceptions import StarletteDeprecationWarning

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r".*httpx2.*", category=StarletteDeprecationWarning
        )
        warnings.filterwarnings(
            "ignore", message=r".*BlockingPortal.*", category=DeprecationWarning
        )
        from fastapi.testclient import TestClient
    return TestClient(app, raise_server_exceptions=False)


def _timed_call(client, method: str, path: str, **kwargs) -> tuple[int, float]:
    started = time.perf_counter()
    response = getattr(client, method)(path, **kwargs)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return response.status_code, elapsed_ms


def _print_table(rows: list[dict]) -> None:
    headers = ("endpoint", "status", "first_ms", "warm_ms")
    widths = [
        max(len(headers[i]), *(len(str(row[headers[i]])) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]  # fmt: skip
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    for row in rows:
        print("  ".join(str(row[headers[i]]).ljust(widths[i]) for i in range(len(headers))))


def _smoke_calls(headers: dict) -> list[tuple[str, str, str, dict, bool, bool]]:
    """Each call is (name, method, path, kwargs, allow_404, required)."""
    calls = [
        ("GET /v1/ready", "get", "/v1/ready", {"headers": headers}, False, True),
        ("POST /v1/price", "post", "/v1/price", {"headers": headers, "json": PRICE_FORECAST_BODY}, False, True),
        ("POST /v1/forecast", "post", "/v1/forecast", {"headers": headers, "json": PRICE_FORECAST_BODY}, False, True),
        ("GET /v1/search", "get", "/v1/search", {"headers": headers, "params": {"q": SEARCH_QUERY}}, False, True),
    ]  # fmt: skip
    sample = _sample_listing()
    if sample is None:
        print("no usable sample listing: skipping /v1/listings/{id}/flags and /v1/listings/check")
    else:
        listing_id = sample["listing_id"]
        calls.append((
            f"GET /v1/listings/{listing_id}/flags", "get", f"/v1/listings/{listing_id}/flags",
            {"headers": headers}, True, True,
        ))  # fmt: skip
        calls.append((
            "POST /v1/listings/check", "post", "/v1/listings/check",
            {"headers": headers, "json": _check_body(sample)}, False, True,
        ))  # fmt: skip
    calls.append(("GET /metrics", "get", "/metrics", {"headers": headers}, False, True))
    return calls


def _smoke(args: argparse.Namespace) -> int:
    load_dotenv()
    if args.components:
        os.environ["API_COMPONENTS"] = ",".join(args.components)
    settings = ApiSettings.from_env()
    headers = {"X-API-Key": settings.api_keys[0]} if settings.api_keys else {}

    started = time.perf_counter()
    app = create_app(settings, default_loaders())
    with _test_client(app) as client:
        startup_seconds = time.perf_counter() - started
        calls = _smoke_calls(headers)
        # Phase 8 may add an `areas` component while this task is in flight; include it
        # in the table when present, but don't let it affect the exit code — its request
        # shape belongs to that phase, not this one.
        if "areas" in settings.components and any(
            getattr(route, "path", None) == "/v1/areas" for route in app.routes
        ):
            calls.append(("GET /v1/areas", "get", "/v1/areas", {"headers": headers}, False, False))

        rows: list[dict] = []
        all_ok = True
        for name, method, path, kwargs, allow_404, required in calls:
            first_status, first_ms = _timed_call(client, method, path, **kwargs)
            warm_status, warm_ms = _timed_call(client, method, path, **kwargs)

            expected = {200, 404} if allow_404 else {200}
            passed = first_status in expected and warm_status in expected
            if required and not passed:
                all_ok = False
            rows.append({
                "endpoint": name, "status": warm_status,
                "first_ms": round(first_ms, 1), "warm_ms": round(warm_ms, 1),
            })  # fmt: skip

    print(f"startup: {startup_seconds:.2f}s")
    _print_table(rows)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "smoke.json").write_text(
        json.dumps({"startup_seconds": round(startup_seconds, 2), "results": rows}, indent=2),
        encoding="utf-8",
    )
    return 0 if all_ok else 1


def _serve(args: argparse.Namespace) -> int:
    load_dotenv()
    settings = ApiSettings.from_env()
    app = create_app(settings, default_loaders())
    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_config=None)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m api",
        description="Run the unified API, or smoke-test every endpoint in-process.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="run the API with uvicorn")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    smoke = commands.add_parser("smoke", help="in-process latency/status check of every endpoint")
    smoke.add_argument(
        "--components", nargs="+", default=None, help="override API_COMPONENTS for this run"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = _parser().parse_args(argv)
    handlers = {"serve": _serve, "smoke": _smoke}
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
