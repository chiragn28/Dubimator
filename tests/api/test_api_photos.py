"""Route tests for GET /v1/listings/{id}/photos and GET /v1/photos/{id}.

The photo files are gitignored (see listings/photos.py), so these write their own JPEG into a
tmp_path and point API_PHOTO_ROOT at it: the route's contract is "serve the bytes the database
names", not "serve the real corpus".
"""

import io

import pytest
from PIL import Image

from tests.api.api_fixtures import KEY, client, fake_loaders, settings


def _headers():
    return {"X-API-Key": KEY}


@pytest.fixture
def photo_root(tmp_path):
    """A photo root holding photos/529_frontal.jpg, the file FakeChecker.photo_path names."""
    directory = tmp_path / "photos"
    directory.mkdir()
    buffer = io.BytesIO()
    Image.new("RGB", (4, 3), (12, 34, 56)).save(buffer, format="JPEG")
    (directory / "529_frontal.jpg").write_bytes(buffer.getvalue())
    return tmp_path


def test_listing_photos_lists_them_in_display_order(photo_root):
    with client(settings(API_PHOTO_ROOT=str(photo_root)), fake_loaders()) as c:
        response = c.get("/v1/listings/1/photos", headers=_headers())
    assert response.status_code == 200
    body = response.json()
    assert body["listing_id"] == 1
    assert body["photos"] == [{"photo_id": 529, "room": "frontal", "position": 0}]


def test_listing_photos_404s_for_an_unknown_listing(photo_root):
    with client(settings(API_PHOTO_ROOT=str(photo_root)), fake_loaders()) as c:
        response = c.get("/v1/listings/404/photos", headers=_headers())
    assert response.status_code == 404
    assert response.json()["error"]["message"] == "listing 404 not found"


def test_a_listing_with_no_photos_is_an_empty_list_not_a_404(photo_root):
    with client(settings(API_PHOTO_ROOT=str(photo_root)), fake_loaders()) as c:
        response = c.get("/v1/listings/7/photos", headers=_headers())
    assert response.status_code == 200
    assert response.json()["photos"] == []


def test_photo_returns_the_jpeg_bytes(photo_root):
    expected = (photo_root / "photos" / "529_frontal.jpg").read_bytes()
    with client(settings(API_PHOTO_ROOT=str(photo_root)), fake_loaders()) as c:
        response = c.get("/v1/photos/529", headers=_headers())
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == expected


def test_photo_404s_when_the_row_exists_but_the_file_does_not(photo_root):
    """A deployment without the corpus images: the API still answers, with a 404."""
    with client(settings(API_PHOTO_ROOT=str(photo_root)), fake_loaders()) as c:
        response = c.get("/v1/photos/530", headers=_headers())
    assert response.status_code == 404
    assert "not on this server" in response.json()["error"]["message"]


def test_photo_404s_for_an_unknown_photo_id(photo_root):
    with client(settings(API_PHOTO_ROOT=str(photo_root)), fake_loaders()) as c:
        response = c.get("/v1/photos/404", headers=_headers())
    assert response.status_code == 404
    assert response.json()["error"]["message"] == "photo 404 not found"


def test_photos_are_off_when_no_photo_root_is_configured():
    with client(settings(API_PHOTO_ROOT=""), fake_loaders()) as c:
        response = c.get("/v1/photos/529", headers=_headers())
    assert response.status_code == 404
    assert response.json()["error"]["message"] == "this deployment serves no photos"


@pytest.mark.parametrize("path", ["/v1/listings/1/photos", "/v1/photos/529"])
def test_both_photo_routes_need_a_key(photo_root, path):
    with client(settings(API_PHOTO_ROOT=str(photo_root)), fake_loaders()) as c:
        response = c.get(path)
    assert response.status_code == 401
