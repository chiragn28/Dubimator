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
       project_name, property_type, property_sub_type, reg_type, size_sqm, bedrooms
FROM listings.listings
WHERE listing_id = (SELECT min(listing_id) FROM listings.listings)
"""
SAMPLE_PHOTOS_SQL = (
    "SELECT photo_id FROM listings.listing_photos WHERE listing_id = %s ORDER BY position"
)


def _sample_listing() -> dict | None:
    """The lowest-`listing_id` corpus row, shaped for `/v1/listings/{id}/flags` and
    `/v1/listings/check` (the kind, `size_basis` and `bedrooms` follow
    `listings.fraud.price_request`). None when the corpus is empty or the row can't be
    mapped (for example an unrecognized `property_sub_type`); either way `smoke` just
    skips the two listing calls and prints why.

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
                project_name, property_type, property_sub_type, reg_type, size_sqm, bedrooms,
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
            "size_basis": "plot" if villa and property_sub_type is None else "built_up",
            "bedrooms": None if bedrooms is None else int(bedrooms),
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
        "project_name", "property_kind", "status", "size_sqm", "size_basis", "bedrooms",
        "photo_ids",
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


def _timed_call(client, method: str, path: str, **kwargs):
    started = time.perf_counter()
    response = getattr(client, method)(path, **kwargs)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return response, elapsed_ms


def _run_call(client, name: str, method: str, path: str, kwargs: dict) -> tuple[dict, object]:
    """One endpoint called twice (first, then warm): (the table row, the warm response)."""
    first, first_ms = _timed_call(client, method, path, **kwargs)
    warm, warm_ms = _timed_call(client, method, path, **kwargs)
    row = {
        "endpoint": name,
        "first_status": first.status_code,
        "warm_status": warm.status_code,
        "first_ms": round(first_ms, 1),
        "warm_ms": round(warm_ms, 1),
    }
    return row, warm


def _passed(row: dict) -> bool:
    """Every endpoint must answer 200, on the first call and the warm one."""
    return row["first_status"] == 200 and row["warm_status"] == 200


TABLE_COLUMNS = ("endpoint", "first_status", "warm_status", "first_ms", "warm_ms")


def _print_table(rows: list[dict]) -> None:
    widths = [
        max(len(column), *(len(str(row[column])) for row in rows)) if rows else len(column)
        for column in TABLE_COLUMNS
    ]
    print("  ".join(column.ljust(width) for column, width in zip(TABLE_COLUMNS, widths)))
    for row in rows:
        print("  ".join(str(row[c]).ljust(w) for c, w in zip(TABLE_COLUMNS, widths)))


def _smoke_calls(headers: dict) -> list[tuple[str, str, str, dict]]:
    """The fixed calls, each (name, method, path, kwargs); all must answer 200."""
    calls = [
        ("GET /v1/ready", "get", "/v1/ready", {"headers": headers}),
        ("POST /v1/price", "post", "/v1/price", {"headers": headers, "json": PRICE_FORECAST_BODY}),
        ("POST /v1/forecast", "post", "/v1/forecast", {"headers": headers, "json": PRICE_FORECAST_BODY}),
        ("GET /v1/search", "get", "/v1/search", {"headers": headers, "params": {"q": SEARCH_QUERY}}),
    ]  # fmt: skip
    sample = _sample_listing()
    if sample is None:
        print("no usable sample listing: skipping /v1/listings/{id}/flags and /v1/listings/check")
    else:
        listing_id = sample["listing_id"]
        calls.append((
            f"GET /v1/listings/{listing_id}/flags", "get", f"/v1/listings/{listing_id}/flags",
            {"headers": headers},
        ))  # fmt: skip
        calls.append((
            "POST /v1/listings/check", "post", "/v1/listings/check",
            {"headers": headers, "json": _check_body(sample)},
        ))  # fmt: skip
    return calls


def _photo_rows(client, headers: dict, listing_id) -> tuple[list[dict], bool]:
    """GET a listing's photo list, then the first photo's bytes: (rows, all passed).

    A listing with no photos, or a host without the (gitignored) corpus images, is not a
    failure: the list is still a 200, and the byte call is skipped with a printed reason.
    """
    path = f"/v1/listings/{listing_id}/photos"
    row, response = _run_call(client, f"GET {path}", "get", path, {"headers": headers})
    rows, ok = [row], _passed(row)
    photos = response.json().get("photos") if response.status_code == 200 else None
    if not photos:
        print(f"listing {listing_id} has no photos: cannot call /v1/photos/{{id}}")
        return rows, ok
    photo_path = f"/v1/photos/{photos[0]['photo_id']}"
    photo, photo_response = _run_call(client, f"GET {photo_path}", "get", photo_path,
                                      {"headers": headers})  # fmt: skip
    if photo_response.status_code == 404:
        print(f"{photo_path}: no image file on this host (API_PHOTO_ROOT), skipping")
        return rows, ok
    rows.append(photo)
    return rows, ok and _passed(photo)


def _area_rows(client, headers: dict) -> tuple[list[dict], bool]:
    """GET /v1/areas, then the history of its first summary area: (rows, all passed)."""
    row, response = _run_call(client, "GET /v1/areas", "get", "/v1/areas", {"headers": headers})
    rows, ok = [row], _passed(row)
    areas = response.json().get("areas") if response.status_code == 200 else None
    if not areas:
        print("no area in GET /v1/areas: cannot call /v1/areas/{id}/history")
        return rows, False
    area_id = areas[0]["area_id"]
    path = f"/v1/areas/{area_id}/history"
    history, _ = _run_call(client, f"GET {path}", "get", path, {"headers": headers})
    rows.append(history)
    return rows, ok and _passed(history)


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
        rows: list[dict] = []
        for name, method, path, kwargs in _smoke_calls(headers):
            rows.append(_run_call(client, name, method, path, kwargs)[0])
        all_ok = all(_passed(row) for row in rows)
        if "listings" in settings.components and (sample := _sample_listing()):
            photo_rows, photos_ok = _photo_rows(client, headers, sample["listing_id"])
            rows += photo_rows
            all_ok = all_ok and photos_ok
        if "areas" in settings.components:
            area_rows, areas_ok = _area_rows(client, headers)
            rows += area_rows
            all_ok = all_ok and areas_ok
        metrics, _ = _run_call(client, "GET /metrics", "get", "/metrics", {"headers": headers})
        rows.append(metrics)
        all_ok = all_ok and _passed(metrics)

    print(f"startup: {startup_seconds:.2f}s")
    _print_table(rows)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "smoke.json"
    out.write_text(
        json.dumps({"startup_seconds": round(startup_seconds, 2), "results": rows}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out}")
    if not all_ok:
        print("smoke FAILED: every endpoint must answer 200 (first and warm call)")
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
