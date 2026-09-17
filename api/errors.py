"""The one error type routes raise; app.py turns it into the JSON error envelope.

Also `redact()`, which turns an exception into the short, credential-free text that 503s
and `/v1/ready` may show. The full error only ever goes to the logs.
"""

import re

MAX_MESSAGE = 120
# `key=value` pairs from libpq-style DSNs and connection errors (quoted or bare values).
_KEY_VALUE = re.compile(
    r"\b(host|hostaddr|password|passwd|user|username)\s*=\s*('[^']*'|\"[^\"]*\"|\S+)",
    re.IGNORECASE,
)
# `scheme://user[:password]@host...` URIs.
_CREDENTIAL_URI = re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s/@]*@\S*")
# libpq's prose form: `server at "db" (10.0.0.1)`, `for user "zest"`.
_QUOTED = re.compile(r'\b(server at|user)\s+"[^"]*"(\s*\([^)]*\))?', re.IGNORECASE)


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, field: str | None = None,
                 headers: dict[str, str] | None = None):  # fmt: skip
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.field = field
        self.headers = headers or {}


def redact(exc: BaseException) -> str:
    """`"<ExceptionType>: <message>"`, with hosts, users, passwords and credential URIs
    masked, and the message cut to its first 120 characters (after masking)."""
    message = str(exc)
    message = _CREDENTIAL_URI.sub("<uri>", message)
    message = _KEY_VALUE.sub(lambda m: f"{m.group(1)}=***", message)
    message = _QUOTED.sub(lambda m: f'{m.group(1)} "***"', message)
    message = " ".join(message.split())[:MAX_MESSAGE]
    return f"{type(exc).__name__}: {message}"
