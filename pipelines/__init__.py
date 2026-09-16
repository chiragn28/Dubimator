"""Scheduled retraining pipeline: ingest, price, listings, search, forecast (Phase 9)."""

import models.price  # noqa: F401 — Windows DLL preload before pandas/pyarrow/xgboost
