"""api.state.default_loaders(): the real component loaders, patched at their dependency
boundary so no real model loads and no real database connects."""

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


def test_listings_loader_raises_when_no_pair_model(monkeypatch):
    from ingestion.config import DbSettings
    from listings import embed, pairmodel

    monkeypatch.setattr(DbSettings, "from_env", lambda: _DummyDbSettings())
    monkeypatch.setattr(embed, "SentenceTransformerEmbedder", _FakeEmbedder)
    monkeypatch.setattr(pairmodel, "load_pair_model", lambda: None)
    loaders = default_loaders()
    with pytest.raises(RuntimeError, match="run python -m listings detect"):
        loaders["listings"](SETTINGS, {})
