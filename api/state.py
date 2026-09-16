"""Loaded components, their versions and load errors; atomic reload."""

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from api.errors import ApiError
from api.settings import COMPONENTS, ApiSettings

LOGGER = logging.getLogger("api.state")


@dataclass
class LoadedComponent:
    value: object
    version: str
    close: Callable[[], None] | None = None


Loader = Callable[[ApiSettings, dict], LoadedComponent]


@dataclass
class AppState:
    enabled: tuple[str, ...]
    components: dict[str, object] = field(default_factory=dict)
    versions: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    closers: list[Callable[[], None]] = field(default_factory=list)
    _locks: dict[str, threading.Lock] = field(default_factory=dict)

    def get(self, name: str):
        if name not in self.enabled:
            raise ApiError(503, "unavailable", f"{name} is not enabled on this server")
        if name not in self.components:
            raise ApiError(503, "unavailable", f"{name} is unavailable: {self.errors.get(name)}")
        return self.components[name]

    def lock(self, name: str) -> threading.Lock:
        return self._locks.setdefault(name, threading.Lock())

    def status(self) -> dict[str, dict]:
        return {
            name: {
                "up": name in self.components,
                "version": self.versions.get(name),
                "error": self.errors.get(name),
            }
            for name in self.enabled
        }

    def close(self) -> None:
        for closer in self.closers:
            try:
                closer()
            except Exception:  # closing is best effort
                LOGGER.warning("closing a component failed", exc_info=True)


def build_state(settings: ApiSettings, loaders: dict[str, Loader]) -> AppState:
    state = AppState(enabled=tuple(n for n in COMPONENTS if n in settings.components))
    context: dict = {}
    for name in state.enabled:
        try:
            loaded = loaders[name](settings, context)
        except Exception as exc:  # noqa: BLE001 — one broken component must not stop the API
            state.errors[name] = f"{type(exc).__name__}: {exc}"
            LOGGER.warning("component %s failed to load", name, extra={"error": state.errors[name]})
            continue
        state.components[name] = loaded.value
        state.versions[name] = loaded.version
        state._locks[name] = threading.Lock()
        if loaded.close is not None:
            state.closers.append(loaded.close)
    return state


class StateHolder:
    def __init__(self, state: AppState):
        self.current = state
        self._reload_lock = threading.Lock()

    def swap(self, new: AppState) -> AppState:
        old, self.current = self.current, new
        return old

    def reload(self, settings: ApiSettings, loaders: dict[str, Loader]) -> AppState:
        with self._reload_lock:
            new = build_state(settings, loaders)
            self.swap(new).close()
            return new
