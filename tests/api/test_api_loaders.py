"""api.state.default_loaders(): the real component loaders, patched at their dependency
boundary so no real model loads and no real database connects."""

from types import SimpleNamespace

import psycopg2
import pytest

from api.settings import COMPONENTS, ApiSettings
from api.state import default_loaders

SETTINGS = ApiSettings.from_env({"API_KEYS": "k"})


def test_default_loaders_keys_are_exactly_components():
    assert set(default_loaders()) == set(COMPONENTS)


def test_price_loader_raises_when_no_champion(monkeypatch):
    from listings import fraud

    monkeypatch.setattr(fraud, "load_price_predictor", lambda uri: None)
    loaders = default_loaders()
    with pytest.raises(RuntimeError, match="price champion unavailable"):
        loaders["price"](SETTINGS, {})


class _DummyConn:
    def close(self):
        pass


class _DummyDbSettings:
    def connect(self):
        return _DummyConn()


class _FakeEmbedder:
    def __init__(self, device=None):
        self.device = device


def test_shared_embedder_is_built_only_when_absent(monkeypatch):
    from api import state
    from listings import embed

    def fail(device=None):
        raise AssertionError("the embedder must not be rebuilt")

    monkeypatch.setattr(embed, "SentenceTransformerEmbedder", fail)
    existing = object()
    assert state._shared_embedder(SETTINGS, {"embedder": existing}) is existing

    monkeypatch.setattr(embed, "SentenceTransformerEmbedder", _FakeEmbedder)
    context = {}
    built = state._shared_embedder(SETTINGS, context)
    assert isinstance(built, _FakeEmbedder) and context["embedder"] is built


def test_areas_loader_reads_shared_homes_from_the_context(monkeypatch):
    from api import areas
    from ingestion.config import DbSettings

    seen = {}

    def fake_load(settings, homes=None):
        seen["homes"] = homes
        return SimpleNamespace(data_end="2023-03-17")

    monkeypatch.setattr(DbSettings, "from_env", lambda: _DummyDbSettings())
    monkeypatch.setattr(areas.AreaStats, "load", staticmethod(fake_load))
    homes = object()
    context = {"homes": homes}
    loaded = default_loaders()["areas"](SETTINGS, context)
    assert seen["homes"] is homes
    assert "homes" not in context  # not kept alive by the state
    assert loaded.version == "data_end:2023-03-17"


class _ProbeCursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.conn.executed.append(sql)
        if self.conn.fail_execute:
            raise psycopg2.OperationalError("server closed the connection unexpectedly")

    def fetchone(self):
        return (1,)


class _ProbeConn:
    def __init__(self, closed=0, fail_execute=False, fail_rollback=False):
        self.closed = closed
        self.fail_execute = fail_execute
        self.fail_rollback = fail_rollback
        self.executed = []
        self.rollbacks = 0

    def cursor(self):
        return _ProbeCursor(self)

    def rollback(self):
        self.rollbacks += 1
        if self.fail_rollback:
            raise psycopg2.InterfaceError("connection already closed")


def test_connection_probe():
    from api.state import _connection_alive

    healthy = _ProbeConn()
    assert _connection_alive(healthy) is True
    assert healthy.executed == ["SELECT 1"] and healthy.rollbacks == 1

    closed = _ProbeConn(closed=1)
    assert _connection_alive(closed) is False
    assert closed.executed == []

    assert _connection_alive(_ProbeConn(fail_execute=True, fail_rollback=True)) is False


def test_listings_loader_raises_when_no_pair_model(monkeypatch):
    from ingestion.config import DbSettings
    from listings import embed, pairmodel

    monkeypatch.setattr(DbSettings, "from_env", lambda: _DummyDbSettings())
    monkeypatch.setattr(embed, "SentenceTransformerEmbedder", _FakeEmbedder)
    monkeypatch.setattr(pairmodel, "load_pair_model", lambda: None)
    loaders = default_loaders()
    with pytest.raises(RuntimeError, match="run python -m listings detect"):
        loaders["listings"](SETTINGS, {})
