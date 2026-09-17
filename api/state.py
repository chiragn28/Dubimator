"""Loaded components, their versions and load errors; atomic reload; per-component rebuild.

Locking: every component has one lock (`AppState.lock(name)`), and routes hold it for the
whole operation. A component's closer takes the same lock, so neither a reload nor a
rebuild closes a connection that a request is still using — the close waits for it.
"""

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from api.errors import ApiError, redact
from api.settings import COMPONENTS, ApiSettings

LOGGER = logging.getLogger("api.state")
CONNECTION_LOST = "database connection lost"


@dataclass
class LoadedComponent:
    value: object
    version: str
    close: Callable[[], None] | None = None
    # A liveness probe: False (or an exception) means the component can't serve requests.
    check: Callable[[], bool] | None = None


Loader = Callable[[ApiSettings, dict], LoadedComponent]


@dataclass
class AppState:
    enabled: tuple[str, ...]
    components: dict[str, object] = field(default_factory=dict)
    versions: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    closers: dict[str, Callable[[], None]] = field(default_factory=dict)
    checks: dict[str, Callable[[], bool]] = field(default_factory=dict)
    # The loaders' shared context (the embedder, the price model), kept for rebuilds.
    context: dict = field(default_factory=dict)
    _locks: dict[str, threading.Lock] = field(default_factory=dict)

    def get(self, name: str):
        if name not in self.enabled:
            raise ApiError(503, "unavailable", f"{name} is not enabled on this server")
        if name not in self.components:
            raise ApiError(503, "unavailable", f"{name} is unavailable: {self.errors.get(name)}")
        return self.components[name]

    def lock(self, name: str) -> threading.Lock:
        return self._locks.setdefault(name, threading.Lock())

    def _alive(self, name: str) -> bool:
        """Runs the probe under the component's lock; a busy component counts as up."""
        check = self.checks.get(name)
        if check is None:
            return True
        lock = self.lock(name)
        if not lock.acquire(blocking=False):
            return True
        try:
            return bool(check())
        except Exception:
            LOGGER.warning("liveness probe for %s raised", name, exc_info=True)
            return False
        finally:
            lock.release()

    def status(self, live: bool = False) -> dict[str, dict]:
        result = {}
        for name in self.enabled:
            up = name in self.components
            error = self.errors.get(name)
            if live and up and not self._alive(name):
                up, error = False, CONNECTION_LOST
            result[name] = {"up": up, "version": self.versions.get(name), "error": error}
        return result

    def install(self, name: str, loaded: LoadedComponent) -> Callable[[], None] | None:
        """Puts `loaded` in place of `name`; returns the replaced component's closer."""
        old_closer = self.closers.pop(name, None)
        self.components[name] = loaded.value
        self.versions[name] = loaded.version
        self.errors.pop(name, None)
        if loaded.check is not None:
            self.checks[name] = loaded.check
        else:
            self.checks.pop(name, None)
        if loaded.close is not None:
            self.closers[name] = _locked(self.lock(name), loaded.close)
        return old_closer

    def close(self) -> None:
        for closer in list(self.closers.values()):
            _close_quietly(closer)


def _locked(lock: threading.Lock, close: Callable[[], None]) -> Callable[[], None]:
    def run() -> None:
        with lock:  # waits for the request that is using the component
            close()

    return run


def _close_quietly(closer: Callable[[], None]) -> None:
    try:
        closer()
    except Exception:  # closing is best effort
        LOGGER.warning("closing a component failed", exc_info=True)


def _run_loader(name: str, loader: Loader, settings: ApiSettings, context: dict):
    """(loaded, None) or (None, redacted error); the full error goes to the log."""
    try:
        return loader(settings, context), None
    except Exception as exc:
        LOGGER.warning(
            "component %s failed to load",
            name,
            exc_info=True,
            extra={"error": f"{type(exc).__name__}: {exc}"},
        )
        return None, redact(exc)


def build_state(settings: ApiSettings, loaders: dict[str, Loader]) -> AppState:
    state = AppState(enabled=tuple(n for n in COMPONENTS if n in settings.components))
    for name in state.enabled:
        state.lock(name)
        loaded, error = _run_loader(name, loaders[name], settings, state.context)
        if loaded is None:
            state.errors[name] = error
        else:
            state.install(name, loaded)
    return state


class StateHolder:
    def __init__(self, state: AppState):
        self.current = state
        # Serializes reloads and rebuilds: a rebuild must not install into a state that a
        # reload is closing.
        self._reload_lock = threading.Lock()
        self._rebuild_guards: dict[str, threading.Lock] = {}
        self._guards_lock = threading.Lock()

    def swap(self, new: AppState) -> AppState:
        old, self.current = self.current, new
        return old

    def reload(self, settings: ApiSettings, loaders: dict[str, Loader]) -> AppState:
        if not self._reload_lock.acquire(blocking=False):
            raise ApiError(409, "conflict", "a reload or reconnect is already in progress")
        try:
            new = build_state(settings, loaders)
            self.swap(new).close()
            return new
        finally:
            self._reload_lock.release()

    def rebuild(self, name: str, settings: ApiSettings, loaders: dict[str, Loader]) -> bool:
        """Reloads one component with a fresh context (reusing the shared embedder), swaps
        it in, then closes the old one under its lock. A failed load keeps the old
        component, so the next failing request tries again. True when it was swapped."""
        with self._reload_lock:
            state = self.current
            context = {}
            if "embedder" in state.context:
                context["embedder"] = state.context["embedder"]
            loaded, error = _run_loader(name, loaders[name], settings, context)
            if loaded is None:
                LOGGER.warning("rebuilding %s failed", name, extra={"error": error})
                return False
            old_closer = state.install(name, loaded)
            if "embedder" not in state.context and "embedder" in context:
                state.context["embedder"] = context["embedder"]
            LOGGER.info("component %s rebuilt", name, extra={"version": loaded.version})
        if old_closer is not None:
            _close_quietly(old_closer)
        return True

    def rebuild_in_background(
        self, name: str, settings: ApiSettings, loaders: dict[str, Loader]
    ) -> threading.Thread | None:
        """Starts a daemon thread running `rebuild`, unless one is already running for
        `name` (then returns None)."""
        with self._guards_lock:
            guard = self._rebuild_guards.setdefault(name, threading.Lock())
        if not guard.acquire(blocking=False):
            return None

        def run() -> None:
            try:
                self.rebuild(name, settings, loaders)
            except Exception:
                LOGGER.exception("rebuilding %s crashed", name)
            finally:
                guard.release()

        thread = threading.Thread(target=run, name=f"rebuild-{name}", daemon=True)
        thread.start()
        return thread


PRICE_MODEL_URI = "models:/zestimator-price@champion"
PROBE_SQL = "SELECT 1"


def _connection_alive(conn) -> bool:
    """The liveness probe for a component that holds one psycopg2 connection."""
    import psycopg2

    if conn.closed:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute(PROBE_SQL)
            cur.fetchone()
        return True
    except psycopg2.Error:
        return False
    finally:
        try:
            conn.rollback()
        except psycopg2.Error:
            pass


def _load_price(settings: ApiSettings, context: dict) -> LoadedComponent:
    """The Phase 3 champion; also stashed in `context` so the listings loader can reuse it."""
    from listings.fraud import load_price_predictor, resolve_price_model_version

    predictor = load_price_predictor(PRICE_MODEL_URI)
    if predictor is None:
        raise RuntimeError("price champion unavailable")
    version = resolve_price_model_version(PRICE_MODEL_URI)
    context["price"] = predictor
    context["price_version"] = version
    return LoadedComponent(value=predictor, version=version)


def _load_forecast(settings: ApiSettings, context: dict) -> LoadedComponent:
    from ingestion.config import DbSettings
    from models.forecast.config import ForecastConfig
    from models.forecast.predict import Forecaster

    # Not shared with the areas loader's `load_homes`: `Forecaster.from_registry` loads
    # (and screens) its own rows inside models/forecast, so sharing one read would mean
    # splitting that constructor, which is outside this package.
    forecaster = Forecaster.from_registry(DbSettings.from_env(), ForecastConfig())
    version = (
        ",".join(
            f"{horizon}:{v}"
            for horizon, (model, v) in forecaster.models.items()
            if model is not None
        )
        or "none"
    )
    return LoadedComponent(value=forecaster, version=version)


def _shared_embedder(settings: ApiSettings, context: dict):
    """The one `SentenceTransformerEmbedder` search and listings both use (built once)."""
    if "embedder" not in context:
        from listings.embed import SentenceTransformerEmbedder, resolve_device

        context["embedder"] = SentenceTransformerEmbedder(
            device=resolve_device(settings.embed_device)
        )
    return context["embedder"]


def _load_search(settings: ApiSettings, context: dict) -> LoadedComponent:
    from ingestion.config import DbSettings
    from search.engine import SearchEngine

    conn = DbSettings.from_env().connect()
    try:
        embedder = _shared_embedder(settings, context)
        engine = SearchEngine(conn, embedder)
    except Exception:
        conn.close()
        raise
    return LoadedComponent(
        value=engine,
        version=engine.ranker_label,
        close=conn.close,
        check=lambda: _connection_alive(conn),
    )


def _load_listings(settings: ApiSettings, context: dict) -> LoadedComponent:
    from ingestion.config import DbSettings
    from listings.check import ListingChecker
    from listings.fraud import load_price_predictor, resolve_price_model_version
    from listings.pairmodel import load_pair_model

    conn = DbSettings.from_env().connect()
    try:
        embedder = _shared_embedder(settings, context)
        pair = load_pair_model()
        if pair is None:
            raise RuntimeError("duplicate pair model not registered: run python -m listings detect")
        price = context.get("price")
        price_version = context.get("price_version")
        if price is None:
            price = load_price_predictor(PRICE_MODEL_URI)
            price_version = resolve_price_model_version(PRICE_MODEL_URI) if price else None
        checker = ListingChecker.from_connection(
            conn, embedder, pair, price, price_version=price_version
        )
    except Exception:
        conn.close()
        raise
    return LoadedComponent(
        value=checker,
        version=f"pair:{pair.version}",
        close=conn.close,
        check=lambda: _connection_alive(conn),
    )


def _load_areas(settings: ApiSettings, context: dict) -> LoadedComponent:
    from api.areas import AreaStats
    from ingestion.config import DbSettings

    # A loader that already read `load_homes` may leave it here; popped so the (large)
    # frame isn't kept alive by the state's context.
    stats = AreaStats.load(DbSettings.from_env(), homes=context.pop("homes", None))
    return LoadedComponent(value=stats, version=f"data_end:{stats.data_end}")


def default_loaders() -> dict[str, Loader]:
    """The real loaders for `python -m api serve`: `price`, `forecast`, `search`, `listings`,
    `areas`.

    Each imports its heavy modules lazily, inside the function body, so `import api` stays
    light — importing this module must not pull in mlflow, sentence-transformers or psycopg2.
    """
    return {
        "price": _load_price,
        "forecast": _load_forecast,
        "search": _load_search,
        "listings": _load_listings,
        "areas": _load_areas,
    }
