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


PRICE_MODEL_URI = "models:/zestimator-price@champion"


def _load_price(settings: ApiSettings, context: dict) -> LoadedComponent:
    """The Phase 3 champion; also stashed in `context` so the listings loader can reuse it."""
    from listings.fraud import load_price_predictor, resolve_price_model_version

    predictor = load_price_predictor(PRICE_MODEL_URI)
    if predictor is None:
        raise RuntimeError("price champion unavailable")
    context["price"] = predictor
    version = resolve_price_model_version(PRICE_MODEL_URI)
    return LoadedComponent(value=predictor, version=version)


def _load_forecast(settings: ApiSettings, context: dict) -> LoadedComponent:
    from ingestion.config import DbSettings
    from models.forecast.config import ForecastConfig
    from models.forecast.predict import Forecaster

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
    """The one `SentenceTransformerEmbedder` search and listings both use."""
    from listings.embed import SentenceTransformerEmbedder, resolve_device

    return context.setdefault(
        "embedder", SentenceTransformerEmbedder(device=resolve_device(settings.embed_device))
    )


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
    return LoadedComponent(value=engine, version=engine.ranker_label, close=conn.close)


def _load_listings(settings: ApiSettings, context: dict) -> LoadedComponent:
    from ingestion.config import DbSettings
    from listings.check import ListingChecker
    from listings.fraud import load_price_predictor
    from listings.pairmodel import load_pair_model

    conn = DbSettings.from_env().connect()
    try:
        embedder = _shared_embedder(settings, context)
        pair = load_pair_model()
        if pair is None:
            raise RuntimeError("duplicate pair model not registered: run python -m listings detect")
        price = context.get("price") or load_price_predictor(PRICE_MODEL_URI)
        checker = ListingChecker.from_connection(conn, embedder, pair, price)
    except Exception:
        conn.close()
        raise
    return LoadedComponent(value=checker, version=f"pair:{pair.version}", close=conn.close)


def _load_areas(settings: ApiSettings, context: dict) -> LoadedComponent:
    from api.areas import AreaStats
    from ingestion.config import DbSettings

    stats = AreaStats.load(DbSettings.from_env())
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
