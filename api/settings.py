"""API settings from the environment (call load_dotenv() first)."""

import os
from dataclasses import dataclass

COMPONENTS = ("price", "forecast", "search", "listings", "areas")


def _csv(value: str | None) -> tuple[str, ...]:
    return tuple(part.strip() for part in (value or "").split(",") if part.strip())


@dataclass(frozen=True)
class ApiSettings:
    api_keys: tuple[str, ...]
    rate_limit_per_minute: int = 60
    cors_origins: tuple[str, ...] = ()
    components: tuple[str, ...] = COMPONENTS
    embed_device: str = "auto"
    auth_enabled: bool = True
    # Keys allowed to call POST /v1/admin/reload; empty disables reload.
    admin_keys: tuple[str, ...] = ()

    @classmethod
    def from_env(cls, environ=None) -> "ApiSettings":
        env = os.environ if environ is None else environ
        keys = _csv(env.get("API_KEYS"))
        allow_open = env.get("API_ALLOW_NO_KEYS") == "1"
        if not keys and not allow_open:
            raise RuntimeError(
                "API_KEYS is empty: set at least one key in .env (comma-separated), or "
                "API_ALLOW_NO_KEYS=1 for local tests only"
            )
        limit = int(env.get("API_RATE_LIMIT_PER_MINUTE", "60"))
        if limit < 1:
            raise RuntimeError("API_RATE_LIMIT_PER_MINUTE must be at least 1")
        components = _csv(env.get("API_COMPONENTS")) or COMPONENTS
        unknown = sorted(set(components) - set(COMPONENTS))
        if unknown:
            raise RuntimeError(f"API_COMPONENTS has unknown names: {unknown}")
        admin_keys = _csv(env.get("API_ADMIN_KEYS"))
        if set(admin_keys) - set(keys):
            raise RuntimeError("every API_ADMIN_KEYS key must also be listed in API_KEYS")
        return cls(
            api_keys=keys,
            rate_limit_per_minute=limit,
            cors_origins=_csv(env.get("API_CORS_ORIGINS")),
            components=components,
            embed_device=env.get("API_EMBED_DEVICE", "auto"),
            auth_enabled=bool(keys),
            admin_keys=admin_keys,
        )
