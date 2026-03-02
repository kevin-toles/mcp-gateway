"""Shared wait_for_service utility — PDW-6.11 REFACTOR.

Extracted into each repo's ``tests/integration/utils/`` to avoid
copy-pasting the same polling loop in every conftest.

Usage::

    from tests.integration.utils.health_waiter import wait_for_service

    if not await wait_for_service("http://localhost:8087/health"):
        pytest.skip("Service unavailable")
"""

from __future__ import annotations

import asyncio
import time

import httpx


async def wait_for_service(url: str, timeout: float = 30.0) -> bool:
    """Poll *url* until it returns HTTP 200 or *timeout* seconds elapse.

    Args:
        url:     Health-check URL to poll (e.g. ``"http://localhost:8087/health"``).
        timeout: Maximum seconds to wait before giving up (default 30 s).

    Returns:
        ``True`` if the service became healthy within the timeout window;
        ``False`` on timeout or persistent connection failure.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(url, timeout=3.0)
                if r.status_code == 200:
                    return True
        except Exception:  # noqa: S110 — intentional silent retry
            pass
        await asyncio.sleep(1.0)
    return False
