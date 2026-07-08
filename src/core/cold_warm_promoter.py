"""Cold / Warm Promoter

Promotes services from COLD → WARM tier on first successful health-check.

.. deprecated::
    Use the deque-based ``ColdWarmPromoter`` in ``idle_timeout.py``
    instead via ``record_dispatch_for_promotion()``.  This module is
    retained only for backward compatibility and will be removed in a
    future release.
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime, timezone
from typing import Optional

import httpx

from src.core.config import Settings
from src.core.keys import normalize_service_key

warnings.warn(
    "cold_warm_promoter is deprecated — use idle_timeout.record_dispatch_for_promotion instead",
    DeprecationWarning,
    stacklevel=2,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLD_WARMUP_TIMEOUT = 60.0   # seconds — generous for cold-start backends
WARM_HEALTH_TIMEOUT = 2.0    # seconds — tight; WARM backends should be fast
BOOT_WARMUP_TIMEOUT = 120.0  # seconds — reserved for BOOT tier


# ---------------------------------------------------------------------------
# Promoter
# ---------------------------------------------------------------------------

class ColdWarmPromoter:
    """Promotes COLD services to WARM after a successful health check.

    Tracks per-service state (tier, last-checked timestamp) and provides
    a ``warm_service()`` coroutine that health-checks the backend and
    mutates ``health_config.SERVICE_TIERS`` on success.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._service_states: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_state(self, service_key: str) -> dict:
        """Return the current tracked state for *service_key*.

        Returns a dict with keys ``tier``, ``last_checked`` (ISO str or
        ``None``), ``healthy`` (bool), and ``promotion_count`` (int).
        """
        service_key = normalize_service_key(service_key)
        state = self._service_states.get(service_key, {})
        return {
            "tier": state.get("tier", "unknown"),
            "last_checked": (
                state["last_checked"].isoformat()
                if state.get("last_checked")
                else None
            ),
            "healthy": state.get("healthy", False),
            "promotion_count": state.get("promotion_count", 0),
        }

    def get_all_states(self) -> dict[str, dict]:
        """Return state dicts for all tracked services."""
        return {sk: self.get_state(sk) for sk in self._service_states}

    async def warm_service(self, service_key: str) -> bool:
        """Health-check the backend for *service_key* and promote if healthy.

        Steps:
        1. Determine the backend URL from ``Settings`` attributes.
        2. Perform an HTTP GET ``{base_url}/health`` with an appropriate
           timeout (60 s for COLD, 2 s for WARM).
        3. If the backend responds with status 200, mutate
           ``health_config.SERVICE_TIERS[service_key] = "WARM"`` and
           record the promotion.
        4. Return ``True`` on success, ``False`` on failure.

        Args:
            service_key: Service identifier (e.g. ``"semantic-search"``).

        Returns:
            ``True`` if the backend is healthy and was promoted (or is
            already WARM/HOT); ``False`` if the health check failed.
        """
        service_key = normalize_service_key(service_key)

        # Determine health-check URL from the Settings object.
        health_url = self._resolve_health_url(service_key)
        if health_url is None:
            logger.warning("No health URL known for %s — cannot warm", service_key)
            self._update_state(service_key, healthy=False)
            return False

        # Determine the warm-up timeout based on the current tier.
        timeout = self._resolve_warmup_timeout(service_key)

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(health_url, timeout=timeout)
                if resp.status_code == 200:
                    self._promote(service_key)
                    return True

                logger.debug(
                    "Health check for %s returned %d — not promoting",
                    service_key,
                    resp.status_code,
                )
                self._update_state(service_key, healthy=False)
                return False

        except httpx.TimeoutException:
            logger.debug("Health check timed out for %s (timeout=%.1fs)", service_key, timeout)
            self._update_state(service_key, healthy=False)
            return False
        except httpx.ConnectError:
            logger.debug("Health check connection refused for %s", service_key)
            self._update_state(service_key, healthy=False)
            return False
        except Exception:
            logger.exception("Unexpected error during health check for %s", service_key)
            self._update_state(service_key, healthy=False)
            return False

    def warmup_timeout_for(self, service_key: str) -> float:
        """Return the warm-up timeout for *service_key* based on its tier.

        Args:
            service_key: Service identifier.

        Returns:
            Timeout in seconds:
                - 120 s for BOOT
                - 60 s for COLD (default)
                - 2 s for WARM / HOT / unknown
        """
        service_key = normalize_service_key(service_key)
        try:
            from src.core.health_config import SERVICE_TIERS  # noqa: PLC0415
        except ImportError:
            return COLD_WARMUP_TIMEOUT

        tier = SERVICE_TIERS.get(service_key, "COLD")
        return {
            "BOOT": BOOT_WARMUP_TIMEOUT,
            "COLD": COLD_WARMUP_TIMEOUT,
        }.get(tier, WARM_HEALTH_TIMEOUT)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_health_url(self, service_key: str) -> str | None:
        """Map a service key to its ``{base_url}/health`` endpoint.

        Looks up the service's base URL via ``Settings`` attribute and
        appends ``/health``.
        """
        attr = _SERVICE_URL_ATTRS.get(service_key)
        if attr is None:
            return None

        base_url: str | None = getattr(self._settings, attr, None)
        if not base_url:
            return None

        return f"{base_url.rstrip('/')}/health"

    def _resolve_warmup_timeout(self, service_key: str) -> float:
        """Return the warm-up timeout for the current tier of *service_key*."""
        service_key = normalize_service_key(service_key)
        return self.warmup_timeout_for(service_key)

    def _promote(self, service_key: str) -> None:
        """Flip *service_key* from COLD → WARM in ``SERVICE_TIERS``."""
        try:
            from src.core.health_config import SERVICE_TIERS  # noqa: PLC0415
        except ImportError:
            logger.warning("Cannot import SERVICE_TIERS — skipping promotion")
            self._update_state(service_key, healthy=True)
            return

        current = SERVICE_TIERS.get(service_key, "COLD")
        if current in ("WARM", "HOT"):
            logger.debug("%s is already %s — no promotion needed", service_key, current)
            self._update_state(service_key, healthy=True)
            return

        SERVICE_TIERS[service_key] = "WARM"
        self._update_state(service_key, healthy=True)
        logger.info("Promoted %s from %s → WARM", service_key, current)

    def _update_state(self, service_key: str, *, healthy: bool) -> None:
        """Update internal tracking state for *service_key*."""
        prev = self._service_states.get(service_key, {})
        try:
            from src.core.health_config import SERVICE_TIERS  # noqa: PLC0415
        except ImportError:
            tier = "unknown"
        else:
            tier = SERVICE_TIERS.get(service_key, "unknown")

        self._service_states[service_key] = {
            "tier": tier,
            "last_checked": datetime.now(timezone.utc),
            "healthy": healthy,
            "promotion_count": prev.get("promotion_count", 0) + (1 if healthy else 0),
        }


# ---------------------------------------------------------------------------
# Mapping: service key → Settings attribute name
# ---------------------------------------------------------------------------

_SERVICE_URL_ATTRS: dict[str, str] = {
    "semantic-search": "SEMANTIC_SEARCH_URL",
    "unified-search-service": "UNIFIED_SEARCH_URL",
    "unified-search-rs": "UNIFIED_SEARCH_RS_URL",
    "code-orchestrator": "CODE_ORCHESTRATOR_URL",
    "llm-gateway": "LLM_GATEWAY_URL",
    "ai-agents": "AI_AGENTS_URL",
    "audit-service": "AUDIT_SERVICE_URL",
    "inference-service-cpp": "INFERENCE_SERVICE_URL",
    "context-management-service": "CONTEXT_MANAGEMENT_URL",
    "amve": "AMVE_SERVICE_URL",
    "struct-analyzer-service": "STRUCT_ANALYZER_URL",
}


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_promoter: Optional[ColdWarmPromoter] = None


def get_promoter(settings: Settings | None = None) -> ColdWarmPromoter:
    """Return the global ``ColdWarmPromoter`` singleton.

    Args:
        settings: Optional ``Settings`` instance.  If omitted the default
                  ``Settings()`` is used internally.

    Returns:
        The shared promoter instance.
    """
    global _promoter
    if _promoter is None:
        _promoter = ColdWarmPromoter(settings=settings)
    return _promoter


def reset_promoter() -> None:
    """Reset the global ``ColdWarmPromoter`` singleton (for testing).

    This is only intended for test fixtures that need a fresh promoter
    instance between test cases.
    """
    global _promoter
    _promoter = None


# ---------------------------------------------------------------------------
# Re-exports from the new spec-based implementation
# ---------------------------------------------------------------------------
# These allow existing callers to migrate gradually without changing
# their import paths.  New code should import directly from idle_timeout.
# ---------------------------------------------------------------------------

from src.core.idle_timeout import record_dispatch_for_promotion  # noqa: F401, E402, PLC0415
