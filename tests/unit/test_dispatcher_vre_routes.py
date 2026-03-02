"""PDW-3: MCP VRE Tool Wiring Verification — dispatcher routes.

Proves that the ToolDispatcher reads audit-service URLs from
``settings.AUDIT_SERVICE_URL`` at route-table construction time, not from
hardcoded strings.  Changing ``AUDIT_SERVICE_URL`` must change every VRE
route's base_url — no hardcoded ``localhost:8084`` in ``_build_routes``.

PDW3.1 — AC-PDW3.1: ``audit_search_exploits`` base_url equals ``settings.AUDIT_SERVICE_URL``
PDW3.2 — AC-PDW3.1: ``audit_search_exploits`` path is ``/v1/audit/exploits``
PDW3.3 — AC-PDW3.2: ``audit_search_cves`` base_url equals ``settings.AUDIT_SERVICE_URL``
PDW3.4 — AC-PDW3.2: ``audit_search_cves`` path is ``/v1/audit/cves``
"""

from __future__ import annotations

import inspect

# ── URL constants ──────────────────────────────────────────────────────────────

_DEFAULT_AUDIT_URL = "http://localhost:8084"
_CUSTOM_AUDIT_URL = "http://custom-audit-service:9084"


# =============================================================================
# TestAuditSearchExploitsRoute
# =============================================================================


class TestAuditSearchExploitsRoute:
    """AC-PDW3.1: audit_search_exploits dispatches to settings.AUDIT_SERVICE_URL + /v1/audit/exploits."""

    def test_route_registered_in_dispatcher(self) -> None:
        """Smoke check: the route exists before testing URL details."""
        from src.core.config import Settings
        from src.tool_dispatcher import ToolDispatcher

        dispatcher = ToolDispatcher(Settings())
        assert dispatcher.get_route("audit_search_exploits") is not None

    def test_route_base_url_equals_default_audit_service_url(self) -> None:
        """With default Settings, base_url == AUDIT_SERVICE_URL default (http://localhost:8084)."""
        from src.core.config import Settings
        from src.tool_dispatcher import ToolDispatcher

        settings = Settings()
        dispatcher = ToolDispatcher(settings)
        route = dispatcher.get_route("audit_search_exploits")
        assert route.base_url == settings.AUDIT_SERVICE_URL

    def test_route_base_url_changes_with_settings_override(self, monkeypatch) -> None:
        """When MCP_GATEWAY_AUDIT_SERVICE_URL env var is set, route picks it up — not hardcoded."""
        monkeypatch.setenv("MCP_GATEWAY_AUDIT_SERVICE_URL", _CUSTOM_AUDIT_URL)

        from importlib import reload

        import src.core.config as config_mod
        import src.tool_dispatcher as dispatcher_mod

        reload(config_mod)
        reload(dispatcher_mod)
        try:
            from src.core.config import Settings
            from src.tool_dispatcher import ToolDispatcher

            settings = Settings()
            dispatcher = ToolDispatcher(settings)
            route = dispatcher.get_route("audit_search_exploits")
            assert route.base_url == _CUSTOM_AUDIT_URL
        finally:
            reload(config_mod)
            reload(dispatcher_mod)

    def test_route_path_is_v1_audit_exploits(self) -> None:
        """Path must be /v1/audit/exploits — matching audit-service endpoint."""
        from src.core.config import Settings
        from src.tool_dispatcher import ToolDispatcher

        dispatcher = ToolDispatcher(Settings())
        route = dispatcher.get_route("audit_search_exploits")
        assert route.path == "/v1/audit/exploits"

    def test_build_routes_source_references_audit_service_url(self) -> None:
        """``_build_routes`` source must reference ``AUDIT_SERVICE_URL`` — not a hardcoded host."""
        from src.tool_dispatcher import _build_routes

        src = inspect.getsource(_build_routes)
        assert "AUDIT_SERVICE_URL" in src, "_build_routes must use settings.AUDIT_SERVICE_URL, not a hardcoded URL"

    def test_no_hardcoded_audit_url_in_build_routes(self) -> None:
        """``_build_routes`` must not contain a hardcoded audit-service port string."""
        from src.tool_dispatcher import _build_routes

        src = inspect.getsource(_build_routes)
        # The URL should come from settings; no literal 'localhost:8084' in route builder
        assert "localhost:8084" not in src, "Hardcoded 'localhost:8084' found in _build_routes — must use settings"


# =============================================================================
# TestAuditSearchCVEsRoute
# =============================================================================


class TestAuditSearchCVEsRoute:
    """AC-PDW3.2: audit_search_cves dispatches to settings.AUDIT_SERVICE_URL + /v1/audit/cves."""

    def test_route_registered_in_dispatcher(self) -> None:
        """Smoke check: the route exists before testing URL details."""
        from src.core.config import Settings
        from src.tool_dispatcher import ToolDispatcher

        dispatcher = ToolDispatcher(Settings())
        assert dispatcher.get_route("audit_search_cves") is not None

    def test_route_base_url_equals_default_audit_service_url(self) -> None:
        """With default Settings, base_url == AUDIT_SERVICE_URL default (http://localhost:8084)."""
        from src.core.config import Settings
        from src.tool_dispatcher import ToolDispatcher

        settings = Settings()
        dispatcher = ToolDispatcher(settings)
        route = dispatcher.get_route("audit_search_cves")
        assert route.base_url == settings.AUDIT_SERVICE_URL

    def test_route_base_url_changes_with_settings_override(self, monkeypatch) -> None:
        """When MCP_GATEWAY_AUDIT_SERVICE_URL env var is set, route picks it up — not hardcoded."""
        monkeypatch.setenv("MCP_GATEWAY_AUDIT_SERVICE_URL", _CUSTOM_AUDIT_URL)

        from importlib import reload

        import src.core.config as config_mod
        import src.tool_dispatcher as dispatcher_mod

        reload(config_mod)
        reload(dispatcher_mod)
        try:
            from src.core.config import Settings
            from src.tool_dispatcher import ToolDispatcher

            settings = Settings()
            dispatcher = ToolDispatcher(settings)
            route = dispatcher.get_route("audit_search_cves")
            assert route.base_url == _CUSTOM_AUDIT_URL
        finally:
            reload(config_mod)
            reload(dispatcher_mod)

    def test_route_path_is_v1_audit_cves(self) -> None:
        """Path must be /v1/audit/cves — matching audit-service endpoint."""
        from src.core.config import Settings
        from src.tool_dispatcher import ToolDispatcher

        dispatcher = ToolDispatcher(Settings())
        route = dispatcher.get_route("audit_search_cves")
        assert route.path == "/v1/audit/cves"


# =============================================================================
# TestBothVRERoutesShareAuditServiceURL
# =============================================================================


class TestBothVRERoutesShareAuditServiceURL:
    """Both VRE routes must share the same AUDIT_SERVICE_URL base — no divergence."""

    def test_both_routes_have_same_base_url(self) -> None:
        """audit_search_exploits and audit_search_cves must resolve to the same base_url."""
        from src.core.config import Settings
        from src.tool_dispatcher import ToolDispatcher

        dispatcher = ToolDispatcher(Settings())
        exploits_route = dispatcher.get_route("audit_search_exploits")
        cves_route = dispatcher.get_route("audit_search_cves")
        assert exploits_route.base_url == cves_route.base_url

    def test_both_routes_base_url_equals_settings(self) -> None:
        """Both routes resolve to settings.AUDIT_SERVICE_URL — single source of truth."""
        from src.core.config import Settings
        from src.tool_dispatcher import ToolDispatcher

        settings = Settings()
        dispatcher = ToolDispatcher(settings)
        for tool in ("audit_search_exploits", "audit_search_cves"):
            route = dispatcher.get_route(tool)
            assert route.base_url == settings.AUDIT_SERVICE_URL, (
                f"{tool}: base_url {route.base_url!r} != AUDIT_SERVICE_URL {settings.AUDIT_SERVICE_URL!r}"
            )
