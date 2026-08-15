"""Credential masking for mcp-gateway logging pipeline. ASCP.KS.2."""

from __future__ import annotations

import logging
import re

_CREDENTIAL_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9\-_]{8,}", re.IGNORECASE),
    re.compile(
        r"(api[_-]?key|apikey|bearer|authorization)[=:\s]+['\"]?([a-zA-Z0-9\-_\.]{16,})['\"]?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(password|passwd|secret|token)[=:\s]+['\"]?([^\s'\"]{8,})['\"]?",
        re.IGNORECASE,
    ),
]


def mask_credentials(message: str) -> str:
    """Replace credential patterns in message with [REDACTED]."""
    if not message:
        return message
    result = message
    for pattern in _CREDENTIAL_PATTERNS:
        result = pattern.sub("[REDACTED]", result)
    return result


class CredentialMaskingFilter(logging.Filter):
    """Logging filter that redacts credential patterns from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = mask_credentials(str(record.msg))
        if record.args:
            record.args = tuple(
                mask_credentials(str(a)) if isinstance(a, str) else a
                for a in (record.args if isinstance(record.args, tuple) else (record.args,))
            )
        return True


__all__ = ["CredentialMaskingFilter", "mask_credentials"]
