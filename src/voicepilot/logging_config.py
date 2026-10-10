"""Structured logging and privacy redaction for VoicePilot."""

import json
import logging
import re
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

request_id_ctx: ContextVar[str | None] = ContextVar("request_id_ctx", default=None)
turn_id_ctx: ContextVar[str | None] = ContextVar("turn_id_ctx", default=None)

# Patterns to sanitize
API_KEY_REGEX = re.compile(r"(gsk_[a-zA-Z0-9_-]{20,}|sk-[a-zA-Z0-9_-]{20,})")
BEARER_REGEX = re.compile(r"Bearer\s+([a-zA-Z0-9_\-\.]+)", re.IGNORECASE)
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")


def redact_sensitive(text: str) -> str:
    """Redact API keys, tokens, and emails from strings."""
    if not isinstance(text, str):
        return text
    text = API_KEY_REGEX.sub("[REDACTED_API_KEY]", text)
    text = BEARER_REGEX.sub("Bearer [REDACTED_TOKEN]", text)
    text = EMAIL_REGEX.sub("[REDACTED_EMAIL]", text)
    return text


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as structured JSON with privacy safeguards."""

    def __init__(self, redact: bool = True):
        super().__init__()
        self.redact = redact

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Context vars
        req_id = request_id_ctx.get()
        if req_id:
            log_entry["request_id"] = req_id

        turn_id = turn_id_ctx.get()
        if turn_id:
            log_entry["turn_id"] = turn_id

        # Attach custom extra fields
        for field in ("stage", "latency_ms", "error_type", "provider", "status_code"):
            if hasattr(record, field):
                log_entry[field] = getattr(record, field)

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        payload = json.dumps(log_entry)
        if self.redact:
            payload = redact_sensitive(payload)
        return payload


def setup_logging(level: str = "INFO", redact: bool = True) -> None:
    """Configure the root logger with structured JSON formatting."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())

    # Clear existing handlers
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredJsonFormatter(redact=redact))
    root_logger.addHandler(handler)


logger = logging.getLogger("voicepilot")
