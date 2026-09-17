"""API keys and the per-key token-bucket rate limiter."""

import hashlib
import hmac
import threading
import time

from api.settings import ApiSettings


def key_id(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]


def extract_key(headers) -> str | None:
    key = headers.get("x-api-key")
    if key:
        return key
    authorization = headers.get("authorization") or ""
    scheme, _, value = authorization.partition(" ")
    return value.strip() or None if scheme.lower() == "bearer" else None


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def _match(presented: str | None, keys: tuple[str, ...]) -> str | None:
    """The configured key equal to `presented`, or None. Compares fixed-length SHA-256
    digests against every key, with no early return, so timing reveals neither a key's
    length nor its position in the list."""
    if presented is None:
        return None
    digest = _digest(presented)
    matched = None
    for key in keys:
        if hmac.compare_digest(digest, _digest(key)) and matched is None:
            matched = key
    return matched


def check_key(presented: str | None, settings: ApiSettings) -> str | None:
    return _match(presented, settings.api_keys)


def check_admin(presented: str | None, settings: ApiSettings) -> bool:
    return _match(presented, settings.admin_keys) is not None


class RateLimiter:
    """Token bucket per identity: `per_minute` tokens, refilled continuously."""

    def __init__(self, per_minute: int, clock=time.monotonic):
        self.capacity = float(per_minute)
        self.rate = per_minute / 60.0
        self.clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def acquire(self, identity: str) -> float | None:
        with self._lock:
            now = self.clock()
            tokens, updated = self._buckets.get(identity, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - updated) * self.rate)
            if tokens >= 1.0:
                self._buckets[identity] = (tokens - 1.0, now)
                return None
            self._buckets[identity] = (tokens, now)
            return (1.0 - tokens) / self.rate
