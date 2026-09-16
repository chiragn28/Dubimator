"""The one error type routes raise; app.py turns it into the JSON error envelope."""


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, field: str | None = None,
                 headers: dict[str, str] | None = None):  # fmt: skip
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.field = field
        self.headers = headers or {}
