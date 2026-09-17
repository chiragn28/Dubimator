"""JSON log lines on stdout, tagged with the current request id."""

import json
import logging
import re
import sys
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
_VALID_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")
_STANDARD = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime"}


def new_request_id(incoming: str | None) -> str:
    if incoming and _VALID_ID.fullmatch(incoming):
        return incoming
    return uuid.uuid4().hex


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        payload.update({k: v for k, v in vars(record).items() if k not in _STANDARD})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _StdoutHandler(logging.StreamHandler):
    """Writes to whatever `sys.stdout` is now, not the stream at setup time (which may
    have been replaced and closed since, as pytest's capture does)."""

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value):
        pass


def configure_logging(level: str = "INFO") -> None:
    logger = logging.getLogger("api")
    logger.setLevel(level)
    logger.propagate = False
    if not any(getattr(h, "_api_json", False) for h in logger.handlers):
        handler = _StdoutHandler()
        handler.setFormatter(JsonFormatter())
        handler._api_json = True
        logger.addHandler(handler)
