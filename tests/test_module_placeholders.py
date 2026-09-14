"""Confirms the Phase 1 placeholder module structure is importable."""

import importlib

import pytest


@pytest.mark.parametrize("module_name", ["ingestion", "models", "api", "demo"])
def test_module_importable(module_name):
    module = importlib.import_module(module_name)
    assert module.__doc__ is not None
