"""Input validation and sanitization — WBS-MCP4 (GREEN).

Provides ``sanitize_string()`` for null-byte stripping and Unicode NFC
normalization, and ``format_validation_errors()`` for structured 422 output.

Reference: Strategy §4.3 (Input Validation Layer — P0), §7.1 Control #11,
           §8 Taxonomy: APP_SEC.API_SECURITY.INPUT_VALIDATION
"""

from __future__ import annotations

import unicodedata
from typing import Any


def sanitize_string(value: str) -> str:
    """Strip null bytes and normalize to Unicode NFC.

    Applied as a Pydantic ``field_validator`` on all user-facing string fields.
    """
    value = value.replace("\x00", "")
    value = unicodedata.normalize("NFC", value)
    return value


def format_validation_errors(exc: Any) -> list[dict[str, str]]:
    """Convert a Pydantic ``ValidationError`` into a structured list.

    Returns a list of ``{"field": ..., "message": ...}`` dicts suitable for
    a 422 JSON response.  Never includes stack traces or internal paths.
    """
    errors: list[dict[str, str]] = []
    for err in exc.errors():
        loc = err.get("loc", ())
        field = ".".join(str(part) for part in loc) if loc else "unknown"
        errors.append({
            "field": field,
            "message": err.get("msg", "Validation error"),
        })
    return errors
