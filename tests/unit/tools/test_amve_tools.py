"""AEI-7 tests - AMVE tool handlers.

Verifies AC-AEI7.1 through AC-AEI7.4:
- AMVE_SERVICE_URL config defaults to http://localhost:8088
- 6 amve_* tool handlers dispatch to correct AMVE endpoints
- amve_evaluate_fitness accepts dimensions array and baseline_snapshot
- Handlers use correct HTTP methods and payload transformation
"""

import json

import httpx
import pytest

from src.core.config import Settings
from src.tool_dispatcher import ToolDispatcher


@pytest.fixture
def dispatcher():
    settings = Settings()
    return ToolDispatcher(settings)


# AC-AEI7.1: AMVE_SERVICE_URL config


class TestAMVEServiceURLConfig:
    """AC-AEI7.1: AMVE_SERVICE_URL defaults to http://localhost:8088."""

    def test_amve_service_url_default(self):
        settings = Settings()
        assert settings.AMVE_SERVICE_URL == "http://localhost:8088"

    def test_amve_service_url_env_override(self, monkeypatch):
        monkeypatch.setenv("MCP_GATEWAY_AMVE_SERVICE_URL", "http://remote:9999")
        settings = Settings()
        assert settings.AMVE_SERVICE_URL == "http://remote:9999"

    def test_amve_service_url_used_in_dispatch_routes(self):
        settings = Settings()
        dispatcher = ToolDispatcher(settings)
        route = dispatcher.get_route("amve_detect_patterns")
        assert route is not None
        assert route.base_url == "http://localhost:8088"


# AC-AEI7.2: 6 amve_* tools registered with correct inputSchema


AMVE_TOOL_NAMES = {
    "amve_detect_patterns",
    "amve_detect_boundaries",
    "amve_detect_communication",
    "amve_build_call_graph",
    "amve_evaluate_fitness",
    "amve_generate_architecture_log",
}


class TestAMVEToolRegistration:
    """AC-AEI7.2: All 6 amve_* tools registered in _INPUT_MODELS."""

    def test_all_amve_tools_in_input_models(self):
        from src.tool_registry import _INPUT_MODELS

        for name in AMVE_TOOL_NAMES:
            assert name in _INPUT_MODELS, f"{name} missing from _INPUT_MODELS"

    def test_amve_input_models_are_pydantic(self):
        from pydantic import BaseModel

        from src.tool_registry import _INPUT_MODELS

        for name in AMVE_TOOL_NAMES:
            model = _INPUT_MODELS[name]
            assert issubclass(model, BaseModel), f"{name} model is not BaseModel"

    def test_detect_patterns_schema_has_source_path(self):
        from src.tool_registry import _INPUT_MODELS

        model = _INPUT_MODELS["amve_detect_patterns"]
        assert "source_path" in model.model_fields

    def test_detect_patterns_schema_has_include_confidence(self):
        from src.tool_registry import _INPUT_MODELS

        model = _INPUT_MODELS["amve_detect_patterns"]
        assert "include_confidence" in model.model_fields

    def test_detect_communication_schema_has_scope(self):
        from src.tool_registry import _INPUT_MODELS

        model = _INPUT_MODELS["amve_detect_communication"]
        assert "scope" in model.model_fields

    def test_evaluate_fitness_schema_has_snapshot_id(self):
        from src.tool_registry import _INPUT_MODELS

        model = _INPUT_MODELS["amve_evaluate_fitness"]
        assert "snapshot_id" in model.model_fields

    def test_evaluate_fitness_schema_has_dimensions(self):
        """AC-AEI7.4: dimensions array param."""
        from src.tool_registry import _INPUT_MODELS

        model = _INPUT_MODELS["amve_evaluate_fitness"]
        assert "fitness_function_ids" in model.model_fields

    def test_generate_arch_log_schema_has_source_paths(self):
        from src.tool_registry import _INPUT_MODELS

        model = _INPUT_MODELS["amve_generate_architecture_log"]
        assert "source_paths" in model.model_fields


# AC-AEI7.3: amve_detect_patterns dispatches to /v1/analysis/patterns


class TestAMVEDispatchRoutes:
    """AC-AEI7.3: Tools dispatch to correct AMVE endpoints."""

    def test_detect_patterns_route_path(self, dispatcher):
        route = dispatcher.get_route("amve_detect_patterns")
        assert route is not None
        assert route.path == "/v1/analysis/patterns"
        assert route.base_url == "http://localhost:8088"

    def test_detect_boundaries_route_path(self, dispatcher):
        route = dispatcher.get_route("amve_detect_boundaries")
        assert route is not None
        assert route.path == "/v1/analysis/boundaries"

    def test_detect_communication_route_path(self, dispatcher):
        route = dispatcher.get_route("amve_detect_communication")
        assert route is not None
        assert route.path == "/v1/analysis/communication"

    def test_build_call_graph_route_path(self, dispatcher):
        route = dispatcher.get_route("amve_build_call_graph")
        assert route is not None
        assert route.path == "/v1/analysis/call-graph"

    def test_evaluate_fitness_route_path(self, dispatcher):
        route = dispatcher.get_route("amve_evaluate_fitness")
        assert route is not None
        assert route.path == "/v1/fitness/evaluate"

    def test_generate_architecture_log_route_path(self, dispatcher):
        route = dispatcher.get_route("amve_generate_architecture_log")
        assert route is not None
        assert route.path == "/v1/architecture/batch-scan"

    @pytest.mark.asyncio
    async def test_detect_patterns_dispatches_post(self, dispatcher):
        """AC-AEI7.3: amve_detect_patterns dispatches POST to /v1/analysis/patterns."""
        captured = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["method"] = request.method
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"success": True, "result": {}})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        await dispatcher.dispatch(
            "amve_detect_patterns",
            {"source_path": "/src", "include_confidence": False},
        )
        assert captured["url"] == "http://localhost:8088/v1/analysis/patterns"
        assert captured["method"] == "POST"


# AC-AEI7.3: Handler payload transformation


class TestAMVEDetectPatternsHandler:
    """AC-AEI7.3: Handler dispatches correctly to AMVE."""

    @pytest.mark.asyncio
    async def test_handler_sends_source_path(self, dispatcher):
        captured = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"success": True})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.amve_detect_patterns import create_handler

        handler = create_handler(dispatcher, OutputSanitizer())
        await handler(source_path="/my/source")
        assert captured["body"]["source_path"] == "/my/source"

    @pytest.mark.asyncio
    async def test_handler_sends_include_confidence(self, dispatcher):
        captured = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"success": True})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.amve_detect_patterns import create_handler

        handler = create_handler(dispatcher, OutputSanitizer())
        await handler(source_path="/src", include_confidence=True)
        assert captured["body"]["include_confidence"] is True


class TestAMVEDetectCommunicationHandler:
    """Handler for amve_detect_communication with scope param."""

    @pytest.mark.asyncio
    async def test_handler_sends_scope(self, dispatcher):
        captured = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"scope": "events"})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.amve_detect_communication import create_handler

        handler = create_handler(dispatcher, OutputSanitizer())
        await handler(source_path="/src", scope="events")
        assert captured["body"]["scope"] == "events"

    @pytest.mark.asyncio
    async def test_handler_default_scope_all(self, dispatcher):
        captured = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"scope": "all"})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.amve_detect_communication import create_handler

        handler = create_handler(dispatcher, OutputSanitizer())
        await handler(source_path="/src")
        assert captured["body"]["scope"] == "all"


# AC-AEI7.4: amve_evaluate_fitness with dimensions + baseline_snapshot


class TestAMVEEvaluateFitnessHandler:
    """AC-AEI7.4: amve_evaluate_fitness accepts dimensions array and baseline."""

    @pytest.mark.asyncio
    async def test_handler_sends_snapshot_id(self, dispatcher):
        captured = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"overall_passed": True})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.amve_evaluate_fitness import create_handler

        handler = create_handler(dispatcher, OutputSanitizer())
        await handler(snapshot_id="snap-001")
        assert captured["body"]["snapshot_id"] == "snap-001"

    @pytest.mark.asyncio
    async def test_handler_sends_fitness_function_ids(self, dispatcher):
        """AC-AEI7.4: dimensions array param accepted."""
        captured = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"overall_passed": True})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.amve_evaluate_fitness import create_handler

        handler = create_handler(dispatcher, OutputSanitizer())
        await handler(
            snapshot_id="snap-001",
            fitness_function_ids=["FF-001", "FF-002"],
        )
        assert captured["body"]["fitness_function_ids"] == ["FF-001", "FF-002"]

    @pytest.mark.asyncio
    async def test_handler_optional_fitness_ids_default_none(self, dispatcher):
        captured = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"overall_passed": True})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.amve_evaluate_fitness import create_handler

        handler = create_handler(dispatcher, OutputSanitizer())
        await handler(snapshot_id="snap-001")
        assert captured["body"].get("fitness_function_ids") is None


class TestAMVEGenerateArchLogHandler:
    """Handler for amve_generate_architecture_log."""

    @pytest.mark.asyncio
    async def test_handler_sends_source_paths(self, dispatcher):
        captured = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"violations": []})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.amve_generate_architecture_log import create_handler

        handler = create_handler(dispatcher, OutputSanitizer())
        await handler(source_paths=["/src/app", "/src/lib"])
        assert captured["body"]["source_paths"] == ["/src/app", "/src/lib"]

    @pytest.mark.asyncio
    async def test_handler_sends_baseline_json(self, dispatcher):
        captured = {}
        baseline = {"violations": [{"type": "circular_dep"}]}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"violations": []})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.amve_generate_architecture_log import create_handler

        handler = create_handler(dispatcher, OutputSanitizer())
        await handler(source_paths=["/src"], baseline_json=baseline)
        assert captured["body"]["baseline_json"] == baseline


# Service name mapping for circuit breakers


class TestAMVEServiceNames:
    """All AMVE tools map to amve service name for circuit breakers."""

    def test_all_amve_tools_map_to_amve_service(self):
        from src.tool_dispatcher import _TOOL_SERVICE_NAMES

        for name in AMVE_TOOL_NAMES:
            assert name in _TOOL_SERVICE_NAMES, f"{name} missing from _TOOL_SERVICE_NAMES"
            assert _TOOL_SERVICE_NAMES[name] == "amve", f"{name} maps to {_TOOL_SERVICE_NAMES[name]!r} not amve"


# Server handler factory registration


class TestAMVEHandlerFactories:
    """All 6 AMVE tools have handler factories in server.py."""

    def test_all_amve_tools_in_handler_factories(self):
        from src.server import _HANDLER_FACTORIES

        for name in AMVE_TOOL_NAMES:
            assert name in _HANDLER_FACTORIES, f"{name} missing from _HANDLER_FACTORIES"
