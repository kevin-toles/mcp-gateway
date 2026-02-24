"""AEI-23 tests — audit_search_exploits + audit_search_cves MCP tools (RED phase).

TDD coverage for AC-AEI23.5:
  - Both tools registered in _INPUT_MODELS (tool_registry)
  - AuditSearchExploitsInput / AuditSearchCVEsInput schema validation
  - Dispatcher routes to /v1/audit/exploits and /v1/audit/cves on audit-service (:8084)
  - create_handler() returns callable for each tool
  - Handler dispatches POST with correct payload
  - tools.yaml registration

Tasks: AEI23.9 (RED), AEI23.10 (GREEN)
"""

from __future__ import annotations

import json

import httpx
import pytest

from src.core.config import Settings
from src.tool_dispatcher import ToolDispatcher


@pytest.fixture
def dispatcher():
    settings = Settings()
    return ToolDispatcher(settings)


# ---------------------------------------------------------------------------
# AC-AEI23.5a — Tools registered in _INPUT_MODELS
# ---------------------------------------------------------------------------


class TestVREToolRegistration:
    """Both audit_search_exploits and audit_search_cves are in _INPUT_MODELS."""

    def test_exploit_tool_in_input_models(self):
        from src.tool_registry import _INPUT_MODELS

        assert "audit_search_exploits" in _INPUT_MODELS, "audit_search_exploits is missing from _INPUT_MODELS"

    def test_cves_tool_in_input_models(self):
        from src.tool_registry import _INPUT_MODELS

        assert "audit_search_cves" in _INPUT_MODELS, "audit_search_cves is missing from _INPUT_MODELS"

    def test_exploit_input_model_is_pydantic(self):
        from pydantic import BaseModel

        from src.tool_registry import _INPUT_MODELS

        model = _INPUT_MODELS["audit_search_exploits"]
        assert issubclass(model, BaseModel)

    def test_cves_input_model_is_pydantic(self):
        from pydantic import BaseModel

        from src.tool_registry import _INPUT_MODELS

        model = _INPUT_MODELS["audit_search_cves"]
        assert issubclass(model, BaseModel)


# ---------------------------------------------------------------------------
# AC-AEI23.5b — AuditSearchExploitsInput schema
# ---------------------------------------------------------------------------


class TestAuditSearchExploitsInputSchema:
    """AuditSearchExploitsInput validates all required and optional fields."""

    def test_schema_has_query_field(self):
        from src.models.schemas import AuditSearchExploitsInput

        assert "query" in AuditSearchExploitsInput.model_fields

    def test_schema_has_cwe_ids_field(self):
        from src.models.schemas import AuditSearchExploitsInput

        assert "cwe_ids" in AuditSearchExploitsInput.model_fields

    def test_schema_has_top_k_field(self):
        from src.models.schemas import AuditSearchExploitsInput

        assert "top_k" in AuditSearchExploitsInput.model_fields

    def test_schema_has_min_similarity_field(self):
        from src.models.schemas import AuditSearchExploitsInput

        assert "min_similarity" in AuditSearchExploitsInput.model_fields

    def test_query_is_required(self):
        from pydantic import ValidationError

        from src.models.schemas import AuditSearchExploitsInput

        with pytest.raises(ValidationError):
            AuditSearchExploitsInput()

    def test_query_must_not_be_empty(self):
        from pydantic import ValidationError

        from src.models.schemas import AuditSearchExploitsInput

        with pytest.raises(ValidationError):
            AuditSearchExploitsInput(query="")

    def test_top_k_defaults_to_10(self):
        from src.models.schemas import AuditSearchExploitsInput

        obj = AuditSearchExploitsInput(query="sql injection")
        assert obj.top_k == 10

    def test_min_similarity_defaults_to_0_7(self):
        from src.models.schemas import AuditSearchExploitsInput

        obj = AuditSearchExploitsInput(query="sql injection")
        assert obj.min_similarity == 0.7

    def test_cwe_ids_defaults_to_none(self):
        from src.models.schemas import AuditSearchExploitsInput

        obj = AuditSearchExploitsInput(query="sql injection")
        assert obj.cwe_ids is None

    def test_valid_construction_all_fields(self):
        from src.models.schemas import AuditSearchExploitsInput

        obj = AuditSearchExploitsInput(
            query="buffer overflow exploit",
            cwe_ids=["CWE-120"],
            top_k=5,
            min_similarity=0.8,
        )
        assert obj.query == "buffer overflow exploit"
        assert obj.cwe_ids == ["CWE-120"]
        assert obj.top_k == 5
        assert obj.min_similarity == 0.8


# ---------------------------------------------------------------------------
# AC-AEI23.5c — AuditSearchCVEsInput schema
# ---------------------------------------------------------------------------


class TestAuditSearchCVEsInputSchema:
    """AuditSearchCVEsInput validates all optional fields."""

    def test_schema_has_cwe_id_field(self):
        from src.models.schemas import AuditSearchCVEsInput

        assert "cwe_id" in AuditSearchCVEsInput.model_fields

    def test_schema_has_severity_field(self):
        from src.models.schemas import AuditSearchCVEsInput

        assert "severity" in AuditSearchCVEsInput.model_fields

    def test_schema_has_ecosystem_field(self):
        from src.models.schemas import AuditSearchCVEsInput

        assert "ecosystem" in AuditSearchCVEsInput.model_fields

    def test_schema_has_limit_field(self):
        from src.models.schemas import AuditSearchCVEsInput

        assert "limit" in AuditSearchCVEsInput.model_fields

    def test_all_fields_default_to_none_or_50(self):
        from src.models.schemas import AuditSearchCVEsInput

        obj = AuditSearchCVEsInput()
        assert obj.cwe_id is None
        assert obj.severity is None
        assert obj.ecosystem is None
        assert obj.limit == 50

    def test_valid_construction_all_fields(self):
        from src.models.schemas import AuditSearchCVEsInput

        obj = AuditSearchCVEsInput(
            cwe_id="CWE-89",
            severity="critical",
            ecosystem="python",
            limit=25,
        )
        assert obj.cwe_id == "CWE-89"
        assert obj.severity == "critical"
        assert obj.ecosystem == "python"
        assert obj.limit == 25


# ---------------------------------------------------------------------------
# AC-AEI23.5d — Dispatcher routes to audit-service
# ---------------------------------------------------------------------------


class TestVREDispatchRoutes:
    """audit_search_exploits and audit_search_cves dispatch to audit-service."""

    def test_exploit_route_registered(self, dispatcher):
        route = dispatcher.get_route("audit_search_exploits")
        assert route is not None

    def test_cves_route_registered(self, dispatcher):
        route = dispatcher.get_route("audit_search_cves")
        assert route is not None

    def test_exploit_route_path_is_exploits(self, dispatcher):
        route = dispatcher.get_route("audit_search_exploits")
        assert route.path == "/v1/audit/exploits"

    def test_cves_route_path_is_cves(self, dispatcher):
        route = dispatcher.get_route("audit_search_cves")
        assert route.path == "/v1/audit/cves"

    def test_exploit_route_base_url_is_audit_service(self, dispatcher):
        route = dispatcher.get_route("audit_search_exploits")
        assert route.base_url == "http://localhost:8084"

    def test_cves_route_base_url_is_audit_service(self, dispatcher):
        route = dispatcher.get_route("audit_search_cves")
        assert route.base_url == "http://localhost:8084"

    @pytest.mark.asyncio
    async def test_dispatch_exploits_sends_post(self, dispatcher):
        captured: dict = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["method"] = request.method
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"success": True, "result": {}})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        await dispatcher.dispatch(
            "audit_search_exploits",
            {"query": "sql injection", "cwe_ids": None, "top_k": 10, "min_similarity": 0.7},
        )
        assert captured["url"] == "http://localhost:8084/v1/audit/exploits"
        assert captured["method"] == "POST"

    @pytest.mark.asyncio
    async def test_dispatch_cves_sends_post(self, dispatcher):
        captured: dict = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["method"] = request.method
            return httpx.Response(200, json={"success": True, "result": {}})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        await dispatcher.dispatch(
            "audit_search_cves",
            {"cwe_id": "CWE-89", "severity": None, "ecosystem": None, "limit": 50},
        )
        assert captured["url"] == "http://localhost:8084/v1/audit/cves"
        assert captured["method"] == "POST"


# ---------------------------------------------------------------------------
# AC-AEI23.5e — create_handler factories
# ---------------------------------------------------------------------------


class TestVREHandlers:
    """create_handler returns callable for both VRE tools."""

    def test_exploits_create_handler_is_callable(self):
        from src.tools.audit_search_exploits import create_handler

        assert callable(create_handler)

    def test_cves_create_handler_is_callable(self):
        from src.tools.audit_search_cves import create_handler

        assert callable(create_handler)

    def test_exploits_handler_returns_callable(self, dispatcher):
        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.audit_search_exploits import create_handler

        handler = create_handler(dispatcher, OutputSanitizer())
        assert callable(handler)

    def test_cves_handler_returns_callable(self, dispatcher):
        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.audit_search_cves import create_handler

        handler = create_handler(dispatcher, OutputSanitizer())
        assert callable(handler)

    @pytest.mark.asyncio
    async def test_exploits_handler_sends_query(self, dispatcher):
        captured: dict = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"success": True, "result": {}})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.audit_search_exploits import create_handler

        handler = create_handler(dispatcher, OutputSanitizer())
        await handler(query="buffer overflow attack vector")
        assert captured["body"]["query"] == "buffer overflow attack vector"

    @pytest.mark.asyncio
    async def test_cves_handler_sends_cwe_id(self, dispatcher):
        captured: dict = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"success": True, "result": {}})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.audit_search_cves import create_handler

        handler = create_handler(dispatcher, OutputSanitizer())
        await handler(cwe_id="CWE-120")
        assert captured["body"]["cwe_id"] == "CWE-120"


# ---------------------------------------------------------------------------
# AC-AEI23.5f — tools.yaml registration
# ---------------------------------------------------------------------------


class TestVREYamlRegistration:
    """Both VRE tools are registered in config/tools.yaml."""

    def test_exploit_tool_in_yaml_registry(self):
        from pathlib import Path

        from src.tool_registry import ToolRegistry

        repo_root = Path(__file__).parent.parent.parent.parent
        tools_yaml = repo_root / "config" / "tools.yaml"
        registry = ToolRegistry(config_path=tools_yaml)
        tool_names = [t.name for t in registry.list_all()]
        assert "audit_search_exploits" in tool_names, (
            f"audit_search_exploits not found in tools.yaml. Found: {tool_names}"
        )

    def test_cves_tool_in_yaml_registry(self):
        from pathlib import Path

        from src.tool_registry import ToolRegistry

        repo_root = Path(__file__).parent.parent.parent.parent
        tools_yaml = repo_root / "config" / "tools.yaml"
        registry = ToolRegistry(config_path=tools_yaml)
        tool_names = [t.name for t in registry.list_all()]
        assert "audit_search_cves" in tool_names, f"audit_search_cves not found in tools.yaml. Found: {tool_names}"

    def test_exploit_tool_tier_is_silver(self):
        from pathlib import Path

        from src.tool_registry import ToolRegistry

        repo_root = Path(__file__).parent.parent.parent.parent
        tools_yaml = repo_root / "config" / "tools.yaml"
        registry = ToolRegistry(config_path=tools_yaml)
        tool = next(
            (t for t in registry.list_all() if t.name == "audit_search_exploits"),
            None,
        )
        assert tool is not None
        assert tool.tier == "silver"

    def test_cves_tool_tier_is_silver(self):
        from pathlib import Path

        from src.tool_registry import ToolRegistry

        repo_root = Path(__file__).parent.parent.parent.parent
        tools_yaml = repo_root / "config" / "tools.yaml"
        registry = ToolRegistry(config_path=tools_yaml)
        tool = next(
            (t for t in registry.list_all() if t.name == "audit_search_cves"),
            None,
        )
        assert tool is not None
        assert tool.tier == "silver"

    def test_exploit_tool_description_mentions_exploit(self):
        from pathlib import Path

        from src.tool_registry import ToolRegistry

        repo_root = Path(__file__).parent.parent.parent.parent
        tools_yaml = repo_root / "config" / "tools.yaml"
        registry = ToolRegistry(config_path=tools_yaml)
        tool = next(
            (t for t in registry.list_all() if t.name == "audit_search_exploits"),
            None,
        )
        assert tool is not None and tool.description
        assert "exploit" in tool.description.lower()

    def test_cves_tool_description_mentions_cve(self):
        from pathlib import Path

        from src.tool_registry import ToolRegistry

        repo_root = Path(__file__).parent.parent.parent.parent
        tools_yaml = repo_root / "config" / "tools.yaml"
        registry = ToolRegistry(config_path=tools_yaml)
        tool = next(
            (t for t in registry.list_all() if t.name == "audit_search_cves"),
            None,
        )
        assert tool is not None and tool.description
        assert "cve" in tool.description.lower()
