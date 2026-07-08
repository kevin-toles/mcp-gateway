"""
Tool Dispatch Tests — Item 36 (HWC F2)
========================================

RED tests that ``inference-service-cpp`` is properly registered in:

  - ``_TOOL_SERVICE_NAMES``
  - ``_build_routes()`` dispatch table
  - Health check COLD tier timeout (60s)
  - ``SERVICE_SHUTDOWN_COMMANDS`` binary pattern

Spec source: TDD_AMENDMENT_PLAN_CONSOLIDATED (3).md §Item 36
"""

from __future__ import annotations

import pytest

from src.tool_dispatcher import _TOOL_SERVICE_NAMES, _build_routes, DispatchRoute
from src.core.config import settings


# ── Test 1: Tool service name registration ─────────────────────────────


class TestInferenceCppRegistration:
    """inference-service-cpp is registered in _TOOL_SERVICE_NAMES."""

    def test_inference_cpp_in_tool_service_names(self) -> None:
        """_TOOL_SERVICE_NAMES has 'inference' -> 'inference-service-cpp'."""
        assert "inference" in _TOOL_SERVICE_NAMES, (
            "inference key missing from _TOOL_SERVICE_NAMES"
        )
        assert _TOOL_SERVICE_NAMES["inference"] == "inference-service-cpp", (
            f"Expected 'inference-service-cpp', got {_TOOL_SERVICE_NAMES['inference']!r}"
        )

    # ── Test 2: Dispatch route registration ────────────────────────────────

    def test_inference_cpp_tool_dispatch_routes(self) -> None:
        """Route table has 'inference' dispatching to inference-service-cpp."""
        routes = _build_routes(settings)
        assert "inference" in routes, "inference route missing from _build_routes()"

        route = routes["inference"]
        assert isinstance(route, DispatchRoute), (
            f"Expected DispatchRoute, got {type(route)}"
        )
        assert route.base_url == settings.INFERENCE_SERVICE_URL, (
            f"Expected base_url={settings.INFERENCE_SERVICE_URL}, "
            f"got {route.base_url}"
        )
        assert route.path == "/v1/models", (
            f"Expected path=/v1/models, got {route.path}"
        )

    # ── Test 3: Cold tier health check timeout ─────────────────────────────

    def test_inference_cpp_health_check_cold_timeout(self) -> None:
        """inference-service-cpp uses COLD tier (60s) health check timeout."""
        # The COLD tier in health_config uses 60s
        # inference-service-cpp is classified as COLD
        from src.config.health_config import TIER_HEALTH_TIMEOUTS, SERVICE_TIERS

        tier = SERVICE_TIERS.get("inference-service-cpp")
        assert tier is not None, (
            "inference-service-cpp not found in SERVICE_TIERS"
        )
        assert tier == "COLD", (
            f"Expected COLD tier, got {tier}"
        )
        timeout = TIER_HEALTH_TIMEOUTS.get(tier)
        assert timeout is not None, (
            f"No timeout defined for tier {tier}"
        )
        assert timeout == 60.0, (
            f"Expected COLD timeout 60.0s, got {timeout}s"
        )

    # ── Test 4: Shutdown command ───────────────────────────────────────────

    def test_inference_cpp_shutdown_command(self) -> None:
        """Shutdown command uses binary pattern, not uvicorn."""
        cmds = settings.SERVICE_SHUTDOWN_COMMANDS
        key = "inference-service-cpp"
        assert key in cmds, f"{key} missing from SERVICE_SHUTDOWN_COMMANDS"

        cmd = cmds[key]
        # Should NOT use uvicorn pattern (lsof -ti:... | xargs kill)
        assert "uvicorn" not in cmd, (
            f"Shutdown command should not be uvicorn pattern, got: {cmd}"
        )
        # Should use a binary kill pattern (pkill or kill)
        assert any(pat in cmd for pat in ("pkill", "kill ")), (
            f"Shutdown command should use binary pattern, got: {cmd}"
        )
