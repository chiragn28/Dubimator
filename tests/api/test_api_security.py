import hashlib
import json
import logging

import pytest

from api.errors import ApiError
from api.logging import JsonFormatter, new_request_id, request_id_var
from api.security import RateLimiter, check_key, extract_key, key_id
from api.settings import COMPONENTS, ApiSettings


def test_settings_from_env():
    settings = ApiSettings.from_env(
        {"API_KEYS": " a1 , b2 ,,", "API_RATE_LIMIT_PER_MINUTE": "5",
         "API_CORS_ORIGINS": "http://localhost:8501", "API_COMPONENTS": "price,search"}
    )  # fmt: skip
    assert settings.api_keys == ("a1", "b2")
    assert settings.rate_limit_per_minute == 5
    assert settings.cors_origins == ("http://localhost:8501",)
    assert settings.components == ("price", "search")
    assert settings.auth_enabled
    assert ApiSettings.from_env({"API_KEYS": "k"}).components == COMPONENTS
    assert ApiSettings.from_env({"API_KEYS": "k"}).rate_limit_per_minute == 60


def test_settings_refuse_to_run_without_keys():
    with pytest.raises(RuntimeError, match="API_KEYS is empty"):
        ApiSettings.from_env({})
    open_settings = ApiSettings.from_env({"API_ALLOW_NO_KEYS": "1"})
    assert not open_settings.auth_enabled


@pytest.mark.parametrize("bad", [{"API_KEYS": "k", "API_RATE_LIMIT_PER_MINUTE": "0"},
                                 {"API_KEYS": "k", "API_COMPONENTS": "price,nope"}])  # fmt: skip
def test_settings_reject_bad_values(bad):
    with pytest.raises(RuntimeError):
        ApiSettings.from_env(bad)


def test_key_helpers():
    assert key_id("secret") == hashlib.sha256(b"secret").hexdigest()[:8]
    assert extract_key({"x-api-key": "k1"}) == "k1"
    assert extract_key({"authorization": "Bearer k2"}) == "k2"
    assert extract_key({"authorization": "Basic zzz"}) is None
    assert extract_key({}) is None
    settings = ApiSettings.from_env({"API_KEYS": "k1,k2"})
    assert check_key("k2", settings) == "k2"
    assert check_key("k3", settings) is None
    assert check_key(None, settings) is None


def test_rate_limiter_refills_over_time():
    now = [0.0]
    limiter = RateLimiter(per_minute=2, clock=lambda: now[0])
    assert limiter.acquire("a") is None
    assert limiter.acquire("a") is None
    wait = limiter.acquire("a")
    assert wait == pytest.approx(30.0)
    assert limiter.acquire("b") is None  # buckets are per identity
    now[0] = 30.0
    assert limiter.acquire("a") is None
    assert limiter.acquire("a") == pytest.approx(30.0)


def test_request_ids():
    assert new_request_id("abc-123_X.y") == "abc-123_X.y"
    generated = new_request_id("bad id with spaces")
    assert len(generated) == 32 and generated.isalnum()
    assert new_request_id("x" * 65) != "x" * 65
    assert len(new_request_id(None)) == 32


def test_json_formatter_includes_request_id_and_extras():
    token = request_id_var.set("req-1")
    try:
        record = logging.LogRecord("api.access", logging.INFO, __file__, 1, "done", None, None)
        record.status = 200
        line = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)
    assert line["request_id"] == "req-1"
    assert line["message"] == "done"
    assert line["status"] == 200
    assert line["level"] == "INFO" and line["logger"] == "api.access"


def test_api_error_carries_its_fields():
    error = ApiError(429, "rate_limited", "slow down", headers={"Retry-After": "3"})
    assert (error.status, error.code, error.field) == (429, "rate_limited", None)
    assert error.headers == {"Retry-After": "3"}
