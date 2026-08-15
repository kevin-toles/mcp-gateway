"""ASCP.VS7 — RED tests for validate_wbs and validate_design_doc MCP tool registration.

These tests fail initially because the tool entries, input schemas, routes, and config
do not exist yet. GREEN phase adds each piece; REFACTOR verifies schema compliance.

Covers AC-ASCP.1 (validate_wbs registered) and AC-ASCP.2 (validate_design_doc registered).
"""

import pytest

from src.core.config import Settings
from src.tool_dispatcher import ToolDispatcher
from src.tool_registry import ToolRegistry
from pathlib import Path


TOOLS_CONFIG = Path(__file__).parent.parent.parent / "config" / "tools.yaml"
VALIDATION_SERVICE_DEFAULT_URL = "http://localhost:8091"


# ── ASCP.VS7.1 RED — validate_wbs tool in route table ───────────────────────


class TestValidateWBSToolRoute:
    """AC-ASCP.1: validate_wbs is registered in ToolDispatcher route table."""

    def test_validate_wbs_route_exists(self):
        settings = Settings()
        dispatcher = ToolDispatcher(settings)
        route = dispatcher.get_route("validate_wbs")
        assert route is not None, "validate_wbs route missing from ToolDispatcher"

    def test_validate_wbs_route_uses_validation_service_url(self):
        settings = Settings()
        dispatcher = ToolDispatcher(settings)
        route = dispatcher.get_route("validate_wbs")
        assert route is not None
        assert "8091" in route.base_url or "validation" in route.base_url.lower(), (
            f"Expected validate_wbs to route to validation-service, got: {route.base_url}"
        )

    def test_validate_wbs_route_path_is_validate_wbs(self):
        settings = Settings()
        dispatcher = ToolDispatcher(settings)
        route = dispatcher.get_route("validate_wbs")
        assert route is not None
        assert route.path == "/validate/wbs", f"Expected /validate/wbs, got: {route.path}"


# ── ASCP.VS7.1 RED — validate_wbs in ToolRegistry from tools.yaml ───────────


class TestValidateWBSToolRegistry:
    """AC-ASCP.1: validate_wbs appears in ToolRegistry loaded from tools.yaml."""

    def test_validate_wbs_in_registry(self):
        registry = ToolRegistry(TOOLS_CONFIG)
        tool = registry.get("validate_wbs")
        assert tool is not None, "validate_wbs not found in ToolRegistry"

    def test_validate_wbs_has_description(self):
        registry = ToolRegistry(TOOLS_CONFIG)
        tool = registry.get("validate_wbs")
        assert tool is not None
        assert tool.description, "validate_wbs description is empty"

    def test_validate_wbs_has_tier(self):
        registry = ToolRegistry(TOOLS_CONFIG)
        tool = registry.get("validate_wbs")
        assert tool is not None
        assert tool.tier in {"bronze", "silver", "gold", "enterprise"}, (
            f"validate_wbs has invalid tier: {tool.tier}"
        )

    def test_validate_wbs_input_model_accepts_content_field(self):
        registry = ToolRegistry(TOOLS_CONFIG)
        tool = registry.get("validate_wbs")
        assert tool is not None
        model = tool.input_model
        instance = model(content="## WBS-TEST: Example\n**Dependencies:** none")
        assert instance.content == "## WBS-TEST: Example\n**Dependencies:** none"


# ── ASCP.VS7.3 RED — validate_design_doc tool in route table ────────────────


class TestValidateDesignDocToolRoute:
    """AC-ASCP.2: validate_design_doc is registered in ToolDispatcher route table."""

    def test_validate_design_doc_route_exists(self):
        settings = Settings()
        dispatcher = ToolDispatcher(settings)
        route = dispatcher.get_route("validate_design_doc")
        assert route is not None, "validate_design_doc route missing from ToolDispatcher"

    def test_validate_design_doc_route_path(self):
        settings = Settings()
        dispatcher = ToolDispatcher(settings)
        route = dispatcher.get_route("validate_design_doc")
        assert route is not None
        assert route.path == "/validate/design_document", (
            f"Expected /validate/design_document, got: {route.path}"
        )

    def test_validate_design_doc_routes_to_validation_service(self):
        settings = Settings()
        dispatcher = ToolDispatcher(settings)
        route_wbs = dispatcher.get_route("validate_wbs")
        route_doc = dispatcher.get_route("validate_design_doc")
        assert route_wbs is not None and route_doc is not None
        assert route_wbs.base_url == route_doc.base_url, (
            "Both tools must route to the same validation-service base_url"
        )


# ── ASCP.VS7.3 RED — validate_design_doc in ToolRegistry ───────────────────


class TestValidateDesignDocToolRegistry:
    """AC-ASCP.2: validate_design_doc appears in ToolRegistry loaded from tools.yaml."""

    def test_validate_design_doc_in_registry(self):
        registry = ToolRegistry(TOOLS_CONFIG)
        tool = registry.get("validate_design_doc")
        assert tool is not None, "validate_design_doc not found in ToolRegistry"

    def test_validate_design_doc_input_model_accepts_content_and_phase(self):
        registry = ToolRegistry(TOOLS_CONFIG)
        tool = registry.get("validate_design_doc")
        assert tool is not None
        model = tool.input_model
        instance = model(content="# Design Doc", phase="ARCHITECT")
        assert instance.content == "# Design Doc"
        assert instance.phase == "ARCHITECT"


# ── ASCP.VS7.5 REFACTOR — schema compliance ─────────────────────────────────


class TestToolEntrySchemaCompliance:
    """ASCP.VS7.5 REFACTOR: Both entries follow existing tools.yaml schema exactly."""

    def test_both_tools_load_from_yaml_without_error(self):
        registry = ToolRegistry(TOOLS_CONFIG)
        names = registry.tool_names()
        assert "validate_wbs" in names
        assert "validate_design_doc" in names

    def test_no_duplicate_tool_names(self):
        import yaml
        raw = TOOLS_CONFIG.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
        names = [t["name"] for t in data["tools"] if "name" in t]
        assert len(names) == len(set(names)), f"Duplicate tool names found: {names}"

    def test_validate_wbs_uses_validation_service_env_var(self):
        import os
        os.environ["MCP_GATEWAY_VALIDATION_SERVICE_URL"] = "http://custom-vs:9999"
        try:
            settings = Settings()
            dispatcher = ToolDispatcher(settings)
            route = dispatcher.get_route("validate_wbs")
            assert route is not None
            assert "9999" in route.base_url, (
                f"Expected route to use VALIDATION_SERVICE_URL env override, got: {route.base_url}"
            )
        finally:
            del os.environ["MCP_GATEWAY_VALIDATION_SERVICE_URL"]
