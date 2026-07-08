"""ServiceKey utilities for the mcp-gateway.

Provides ``make_service_key()`` which normalizes service names to
canonical hyphen form, with optional ``port`` suffix for URI format.

This complements the ``NewType("ServiceKey", str)`` in ``config.py``
which is used for type annotations.  The function uses a different name
(``make_service_key``) to avoid shadowing the NewType identity function.
"""

from __future__ import annotations

from src.core.keys import normalize_service_key


def make_service_key(name: str, port: int | None = None) -> str:
    """Build a canonical service key from *name*.

    Normalises underscores to hyphens (:func:`normalize_service_key`)
    and optionally appends ``:port`` for URI-style keys.

    Idempotent: ``make_service_key(make_service_key(x)) == make_service_key(x)``
    because the hyphen form passes through ``normalize_service_key`` unchanged.
    """
    normalized = normalize_service_key(name)
    if port is not None:
        return f"{normalized}:{port}"
    return normalized
