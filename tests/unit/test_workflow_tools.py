"""Tests for Workflow MCP Tools — WBS-WF6 (RED).

Verifies the 5 workflow tools that bridge MCP protocol to backend endpoints:
- convert_pdf: POST /api/v1/workflows/convert-pdf → Code-Orchestrator :8083
- extract_book_metadata: POST /api/v1/workflows/extract-book → Code-Orchestrator :8083
- generate_taxonomy: POST /api/v1/workflows/generate-taxonomy → Code-Orchestrator :8083
- enrich_book_metadata: POST /v1/workflows/enrich-book → ai-agents :8082
- enhance_guideline: POST /v1/workflows/enhance-guideline → ai-agents :8082

AC-WF6.1: 5 tools registered in tools.yaml with tier=gold, tags=[workflow, ...]
AC-WF6.2: 5 Pydantic input schemas with validation & sanitization
AC-WF6.3: 5 dispatch routes with timeout=300s
AC-WF6.4: 5 tool handlers wired to correct backend endpoints
AC-WF6.5: tools/list returns 14 tools (9 existing + 5 workflow)
"""

from unittest.mock import AsyncMock

import httpx
import pytest
from fastmcp import Context
from pydantic import ValidationError

from src.core.config import Settings
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import DispatchResult, ToolDispatcher

# ── Helpers ─────────────────────────────────────────────────────────────


def _make_result(body: dict, status_code: int = 200) -> DispatchResult:
    return DispatchResult(
        status_code=status_code,
        body=body,
        headers={"content-type": "application/json"},
        elapsed_ms=150.0,
    )


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def dispatcher():
    settings = Settings()
    return ToolDispatcher(settings)


@pytest.fixture
def mock_dispatcher():
    d = AsyncMock(spec=ToolDispatcher)
    d.dispatch = AsyncMock(return_value=_make_result({"output_path": "/tmp/out.json", "status": "success"}))
    return d


@pytest.fixture
def sanitizer():
    return OutputSanitizer()


# ═══════════════════════════════════════════════════════════════════════
# AC-WF6.1: tools.yaml registration — 5 workflow tools with tier=gold
# ═══════════════════════════════════════════════════════════════════════


WORKFLOW_TOOLS = [
    "convert_pdf",
    "extract_book_metadata",
    "batch_extract_metadata",
    "generate_taxonomy",
    "enrich_book_metadata",
    "enhance_guideline",
    "analyze_taxonomy_coverage",
]


class TestToolRegistryWorkflow:
    """AC-WF6.1: 5 workflow tools in tools.yaml with correct metadata."""

    def test_all_workflow_tools_in_registry(self):
        from pathlib import Path

        from src.tool_registry import ToolRegistry

        config_path = Path(__file__).resolve().parents[2] / "config" / "tools.yaml"
        registry = ToolRegistry(config_path)

        for tool_name in WORKFLOW_TOOLS:
            tool = registry.get(tool_name)
            assert tool is not None, f"{tool_name} not in registry"

    def test_workflow_tools_have_gold_tier(self):
        from pathlib import Path

        from src.tool_registry import ToolRegistry

        config_path = Path(__file__).resolve().parents[2] / "config" / "tools.yaml"
        registry = ToolRegistry(config_path)

        for tool_name in WORKFLOW_TOOLS:
            tool = registry.get(tool_name)
            assert tool.tier == "gold", f"{tool_name} should be tier=gold, got {tool.tier}"

    def test_workflow_tools_have_workflow_tag(self):
        from pathlib import Path

        from src.tool_registry import ToolRegistry

        config_path = Path(__file__).resolve().parents[2] / "config" / "tools.yaml"
        registry = ToolRegistry(config_path)

        for tool_name in WORKFLOW_TOOLS:
            tool = registry.get(tool_name)
            assert "workflow" in tool.tags, f"{tool_name} missing 'workflow' tag"

    def test_registry_total_is_15(self):
        from pathlib import Path

        from src.tool_registry import ToolRegistry

        config_path = Path(__file__).resolve().parents[2] / "config" / "tools.yaml"
        registry = ToolRegistry(config_path)
        assert len(registry.list_all()) == 16


# ═══════════════════════════════════════════════════════════════════════
# AC-WF6.3: Dispatch routes — 5 workflow tools with timeout=300s
# ═══════════════════════════════════════════════════════════════════════


EXPECTED_WORKFLOW_ROUTES = {
    "convert_pdf": {
        "base_url": "http://localhost:8083",
        "path": "/api/v1/workflows/convert-pdf",
        "timeout": 300.0,
    },
    "extract_book_metadata": {
        "base_url": "http://localhost:8083",
        "path": "/api/v1/workflows/extract-book",
        "timeout": None,
    },
    "batch_extract_metadata": {
        "base_url": "http://localhost:8083",
        "path": "/api/v1/workflows/extract-book",
        "timeout": None,
    },
    "generate_taxonomy": {
        "base_url": "http://localhost:8083",
        "path": "/api/v1/workflows/generate-taxonomy",
        "timeout": 300.0,
    },
    "enrich_book_metadata": {
        "base_url": "http://localhost:8082",
        "path": "/v1/workflows/enrich-book",
        "timeout": 300.0,
    },
    "enhance_guideline": {
        "base_url": "http://localhost:8082",
        "path": "/v1/workflows/enhance-guideline",
        "timeout": 300.0,
    },
    "analyze_taxonomy_coverage": {
        "base_url": "http://localhost:8083",
        "path": "/api/v1/workflows/analyze-taxonomy-coverage",
        "timeout": 300.0,
    },
}


class TestWorkflowRouteTable:
    """AC-WF6.3: Correct dispatch routes for all 5 workflow tools."""

    @pytest.mark.parametrize("tool_name,expected", list(EXPECTED_WORKFLOW_ROUTES.items()))
    def test_route_exists(self, dispatcher, tool_name, expected):
        route = dispatcher.get_route(tool_name)
        assert route is not None, f"Missing route for {tool_name}"

    @pytest.mark.parametrize("tool_name,expected", list(EXPECTED_WORKFLOW_ROUTES.items()))
    def test_route_base_url(self, dispatcher, tool_name, expected):
        route = dispatcher.get_route(tool_name)
        assert route.base_url == expected["base_url"], (
            f"{tool_name}: expected base_url={expected['base_url']}, got {route.base_url}"
        )

    @pytest.mark.parametrize("tool_name,expected", list(EXPECTED_WORKFLOW_ROUTES.items()))
    def test_route_path(self, dispatcher, tool_name, expected):
        route = dispatcher.get_route(tool_name)
        assert route.path == expected["path"], f"{tool_name}: expected path={expected['path']}, got {route.path}"

    @pytest.mark.parametrize("tool_name,expected", list(EXPECTED_WORKFLOW_ROUTES.items()))
    def test_route_timeout_matches_expected(self, dispatcher, tool_name, expected):
        route = dispatcher.get_route(tool_name)
        assert route.timeout == expected["timeout"], (
            f"{tool_name}: expected timeout={expected['timeout']}, got {route.timeout}"
        )

    def test_total_route_count_is_16(self, dispatcher):
        assert len(dispatcher.routes) == 16

    @pytest.mark.parametrize("tool_name,expected", list(EXPECTED_WORKFLOW_ROUTES.items()))
    @pytest.mark.asyncio
    async def test_dispatches_to_correct_url(self, dispatcher, tool_name, expected):
        """Verify each workflow tool dispatches to the correct backend URL."""
        captured = {}

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["method"] = request.method
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(mock_handler)
        dispatcher._client = httpx.AsyncClient(transport=transport)

        await dispatcher.dispatch(tool_name, {"input_path": "/tmp/test.pdf"})
        expected_url = f"{expected['base_url']}{expected['path']}"
        assert captured["url"] == expected_url
        assert captured["method"] == "POST"


class TestWorkflowServiceNames:
    """AC-WF6.3: _TOOL_SERVICE_NAMES has entries for all 5 workflow tools."""

    @pytest.mark.parametrize(
        "tool_name,expected_service",
        [
            ("convert_pdf", "code-orchestrator"),
            ("extract_book_metadata", "code-orchestrator"),
            ("batch_extract_metadata", "code-orchestrator"),
            ("generate_taxonomy", "code-orchestrator"),
            ("enrich_book_metadata", "ai-agents"),
            ("enhance_guideline", "ai-agents"),
            ("analyze_taxonomy_coverage", "code-orchestrator"),
        ],
    )
    @pytest.mark.asyncio
    async def test_connection_error_includes_service_name(self, dispatcher, tool_name, expected_service):
        """BackendUnavailableError includes the correct service name."""
        from src.core.errors import BackendUnavailableError

        async def fail_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused")

        transport = httpx.MockTransport(fail_handler)
        dispatcher._client = httpx.AsyncClient(transport=transport)

        with pytest.raises(BackendUnavailableError) as exc_info:
            await dispatcher.dispatch(tool_name, {"input": "data"})

        assert expected_service in exc_info.value.service_name.lower().replace("_", "-")


# ═══════════════════════════════════════════════════════════════════════
# AC-WF6.2: Pydantic input schemas — validation & sanitization
# ═══════════════════════════════════════════════════════════════════════


class TestConvertPDFSchema:
    """Input validation for convert_pdf tool."""

    def test_valid_minimal(self):
        from src.models.schemas import ConvertPDFInput

        m = ConvertPDFInput(input_path="/tmp/book.pdf")
        assert m.input_path == "/tmp/book.pdf"
        assert m.enable_ocr is True  # default

    def test_valid_full(self):
        from src.models.schemas import ConvertPDFInput

        m = ConvertPDFInput(
            input_path="/tmp/book.pdf",
            output_path="/tmp/out.json",
            enable_ocr=False,
        )
        assert m.output_path == "/tmp/out.json"
        assert m.enable_ocr is False

    def test_rejects_empty_input_path(self):
        from src.models.schemas import ConvertPDFInput

        with pytest.raises(ValidationError):
            ConvertPDFInput(input_path="")

    def test_rejects_missing_input_path(self):
        from src.models.schemas import ConvertPDFInput

        with pytest.raises(ValidationError):
            ConvertPDFInput()


class TestExtractBookMetadataSchema:
    """Input validation for extract_book_metadata tool."""

    def test_valid_minimal(self):
        from src.models.schemas import ExtractBookMetadataInput

        m = ExtractBookMetadataInput(input_path="/tmp/book.json")
        assert m.input_path == "/tmp/book.json"
        assert m.output_path is None
        assert m.chapters is None
        assert m.options is None

    def test_valid_full(self):
        from src.models.schemas import ExtractBookMetadataInput

        m = ExtractBookMetadataInput(
            input_path="/tmp/book.json",
            output_path="/tmp/meta.json",
            chapters=[{"number": 1, "title": "Ch1", "start_page": 1, "end_page": 10}],
            options={"extract_code": True},
        )
        assert m.output_path == "/tmp/meta.json"
        assert len(m.chapters) == 1
        assert m.options["extract_code"] is True

    def test_rejects_empty_input_path(self):
        from src.models.schemas import ExtractBookMetadataInput

        with pytest.raises(ValidationError):
            ExtractBookMetadataInput(input_path="")

    def test_rejects_missing_input_path(self):
        from src.models.schemas import ExtractBookMetadataInput

        with pytest.raises(ValidationError):
            ExtractBookMetadataInput()


class TestBatchExtractMetadataSchema:
    """Input validation for batch_extract_metadata tool."""

    def test_valid_minimal(self):
        from src.models.schemas import BatchExtractMetadataInput

        m = BatchExtractMetadataInput(input_dir="/tmp/books")
        assert m.input_dir == "/tmp/books"
        assert m.output_dir is None
        assert m.file_pattern == "*.json"
        assert m.skip_existing is True

    def test_valid_full(self):
        from src.models.schemas import BatchExtractMetadataInput

        m = BatchExtractMetadataInput(
            input_dir="/tmp/books",
            output_dir="/tmp/metadata",
            file_pattern="*.json",
            skip_existing=False,
        )
        assert m.input_dir == "/tmp/books"
        assert m.output_dir == "/tmp/metadata"
        assert m.file_pattern == "*.json"
        assert m.skip_existing is False

    def test_rejects_empty_input_dir(self):
        from src.models.schemas import BatchExtractMetadataInput

        with pytest.raises(ValidationError):
            BatchExtractMetadataInput(input_dir="")

    def test_rejects_missing_input_dir(self):
        from src.models.schemas import BatchExtractMetadataInput

        with pytest.raises(ValidationError):
            BatchExtractMetadataInput()

    def test_defaults_file_pattern(self):
        from src.models.schemas import BatchExtractMetadataInput

        m = BatchExtractMetadataInput(input_dir="/tmp/books")
        assert m.file_pattern == "*.json"

    def test_defaults_skip_existing_true(self):
        from src.models.schemas import BatchExtractMetadataInput

        m = BatchExtractMetadataInput(input_dir="/tmp/books")
        assert m.skip_existing is True

    def test_accepts_custom_file_pattern(self):
        from src.models.schemas import BatchExtractMetadataInput

        m = BatchExtractMetadataInput(input_dir="/tmp/books", file_pattern="book_*.json")
        assert m.file_pattern == "book_*.json"


class TestGenerateTaxonomySchema:
    """Input validation for generate_taxonomy tool."""

    def test_valid_minimal(self):
        from src.models.schemas import GenerateTaxonomyInput

        m = GenerateTaxonomyInput(
            tier_books={"beginner": ["/tmp/b1.json"]},
        )
        assert m.tier_books == {"beginner": ["/tmp/b1.json"]}
        assert m.domain == "auto"

    def test_valid_full(self):
        from src.models.schemas import GenerateTaxonomyInput

        m = GenerateTaxonomyInput(
            tier_books={"beginner": ["/tmp/b1.json"], "advanced": ["/tmp/b2.json"]},
            output_path="/tmp/taxonomy.json",
            concepts=["OOP", "testing"],
            domain="python",
        )
        assert m.output_path == "/tmp/taxonomy.json"
        assert len(m.concepts) == 2
        assert m.domain == "python"

    def test_rejects_empty_tier_books(self):
        from src.models.schemas import GenerateTaxonomyInput

        with pytest.raises(ValidationError):
            GenerateTaxonomyInput(tier_books={})

    def test_rejects_missing_tier_books(self):
        from src.models.schemas import GenerateTaxonomyInput

        with pytest.raises(ValidationError):
            GenerateTaxonomyInput()

    def test_rejects_invalid_domain(self):
        from src.models.schemas import GenerateTaxonomyInput

        with pytest.raises(ValidationError):
            GenerateTaxonomyInput(
                tier_books={"beginner": ["/tmp/b1.json"]},
                domain="invalid_domain",
            )

    def test_accepts_valid_domains(self):
        from src.models.schemas import GenerateTaxonomyInput

        for domain in ("python", "architecture", "data_science", "auto"):
            m = GenerateTaxonomyInput(
                tier_books={"t": ["/tmp/b.json"]},
                domain=domain,
            )
            assert m.domain == domain


class TestEnrichBookMetadataSchema:
    """Input validation for enrich_book_metadata tool."""

    def test_valid_minimal(self):
        from src.models.schemas import EnrichBookMetadataInput

        m = EnrichBookMetadataInput(input_path="/tmp/book.json")
        assert m.input_path == "/tmp/book.json"
        assert m.mode == "msep"

    def test_valid_full(self):
        from src.models.schemas import EnrichBookMetadataInput

        m = EnrichBookMetadataInput(
            input_path="/tmp/book.json",
            output_path="/tmp/enriched.json",
            taxonomy_path="/tmp/taxonomy.json",
            mode="msep",
        )
        assert m.output_path == "/tmp/enriched.json"
        assert m.taxonomy_path == "/tmp/taxonomy.json"

    def test_rejects_empty_input_path(self):
        from src.models.schemas import EnrichBookMetadataInput

        with pytest.raises(ValidationError):
            EnrichBookMetadataInput(input_path="")

    def test_rejects_missing_input_path(self):
        from src.models.schemas import EnrichBookMetadataInput

        with pytest.raises(ValidationError):
            EnrichBookMetadataInput()


class TestEnhanceGuidelineSchema:
    """Input validation for enhance_guideline tool."""

    def test_valid_minimal(self):
        from src.models.schemas import EnhanceGuidelineInput

        m = EnhanceGuidelineInput(
            aggregate_path="/tmp/agg.json",
            guideline_path="/tmp/guide.json",
        )
        assert m.aggregate_path == "/tmp/agg.json"
        assert m.guideline_path == "/tmp/guide.json"
        assert m.output_dir == "output"
        assert m.provider == "gateway"
        assert m.max_tokens == 4096
        assert m.temperature == 0.7

    def test_valid_full(self):
        from src.models.schemas import EnhanceGuidelineInput

        m = EnhanceGuidelineInput(
            aggregate_path="/tmp/agg.json",
            guideline_path="/tmp/guide.json",
            output_dir="/tmp/enhanced",
            provider="anthropic",
            max_tokens=8192,
            temperature=0.5,
        )
        assert m.output_dir == "/tmp/enhanced"
        assert m.provider == "anthropic"
        assert m.max_tokens == 8192

    def test_rejects_empty_aggregate_path(self):
        from src.models.schemas import EnhanceGuidelineInput

        with pytest.raises(ValidationError):
            EnhanceGuidelineInput(
                aggregate_path="",
                guideline_path="/tmp/guide.json",
            )

    def test_rejects_empty_guideline_path(self):
        from src.models.schemas import EnhanceGuidelineInput

        with pytest.raises(ValidationError):
            EnhanceGuidelineInput(
                aggregate_path="/tmp/agg.json",
                guideline_path="",
            )

    def test_rejects_missing_required_paths(self):
        from src.models.schemas import EnhanceGuidelineInput

        with pytest.raises(ValidationError):
            EnhanceGuidelineInput()

    def test_rejects_invalid_provider(self):
        from src.models.schemas import EnhanceGuidelineInput

        with pytest.raises(ValidationError):
            EnhanceGuidelineInput(
                aggregate_path="/tmp/agg.json",
                guideline_path="/tmp/guide.json",
                provider="invalid_provider",
            )

    def test_accepts_valid_providers(self):
        from src.models.schemas import EnhanceGuidelineInput

        for provider in ("gateway", "anthropic", "local"):
            m = EnhanceGuidelineInput(
                aggregate_path="/tmp/agg.json",
                guideline_path="/tmp/guide.json",
                provider=provider,
            )
            assert m.provider == provider

    def test_rejects_temperature_out_of_range(self):
        from src.models.schemas import EnhanceGuidelineInput

        with pytest.raises(ValidationError):
            EnhanceGuidelineInput(
                aggregate_path="/tmp/agg.json",
                guideline_path="/tmp/guide.json",
                temperature=3.0,
            )

    def test_rejects_max_tokens_too_high(self):
        from src.models.schemas import EnhanceGuidelineInput

        with pytest.raises(ValidationError):
            EnhanceGuidelineInput(
                aggregate_path="/tmp/agg.json",
                guideline_path="/tmp/guide.json",
                max_tokens=100000,
            )


class TestAnalyzeTaxonomyCoverageSchema:
    """Input validation for analyze_taxonomy_coverage tool (WBS-TAP9)."""

    def test_valid_minimal(self):
        from src.models.schemas import AnalyzeTaxonomyCoverageInput

        m = AnalyzeTaxonomyCoverageInput(taxonomy_path="/tmp/taxonomy.json")
        assert m.taxonomy_path == "/tmp/taxonomy.json"
        assert m.output_path is None
        assert m.collection == "all"
        assert m.top_k == 10
        assert m.threshold == 0.3
        assert m.max_leaf_nodes == 500
        assert m.subtree_root is None
        assert m.concurrency == 10
        assert m.include_evidence is True
        assert m.scoring_weights is None

    def test_valid_full(self):
        from src.models.schemas import AnalyzeTaxonomyCoverageInput

        m = AnalyzeTaxonomyCoverageInput(
            taxonomy_path="/tmp/taxonomy.json",
            output_path="/tmp/out.json",
            collection="code_chunks",
            top_k=20,
            threshold=0.5,
            max_leaf_nodes=1000,
            subtree_root="algorithms",
            concurrency=5,
            include_evidence=False,
            scoring_weights={"semantic": 0.7, "keyword": 0.3},
        )
        assert m.output_path == "/tmp/out.json"
        assert m.collection == "code_chunks"
        assert m.top_k == 20
        assert m.threshold == 0.5
        assert m.max_leaf_nodes == 1000
        assert m.subtree_root == "algorithms"
        assert m.concurrency == 5
        assert m.include_evidence is False
        assert m.scoring_weights == {"semantic": 0.7, "keyword": 0.3}

    def test_rejects_empty_taxonomy_path(self):
        from src.models.schemas import AnalyzeTaxonomyCoverageInput

        with pytest.raises(ValidationError):
            AnalyzeTaxonomyCoverageInput(taxonomy_path="")

    def test_rejects_missing_taxonomy_path(self):
        from src.models.schemas import AnalyzeTaxonomyCoverageInput

        with pytest.raises(ValidationError):
            AnalyzeTaxonomyCoverageInput()

    def test_rejects_top_k_out_of_range(self):
        from src.models.schemas import AnalyzeTaxonomyCoverageInput

        with pytest.raises(ValidationError):
            AnalyzeTaxonomyCoverageInput(taxonomy_path="/tmp/t.json", top_k=0)
        with pytest.raises(ValidationError):
            AnalyzeTaxonomyCoverageInput(taxonomy_path="/tmp/t.json", top_k=101)

    def test_rejects_threshold_out_of_range(self):
        from src.models.schemas import AnalyzeTaxonomyCoverageInput

        with pytest.raises(ValidationError):
            AnalyzeTaxonomyCoverageInput(taxonomy_path="/tmp/t.json", threshold=-0.1)
        with pytest.raises(ValidationError):
            AnalyzeTaxonomyCoverageInput(taxonomy_path="/tmp/t.json", threshold=1.1)

    def test_rejects_concurrency_out_of_range(self):
        from src.models.schemas import AnalyzeTaxonomyCoverageInput

        with pytest.raises(ValidationError):
            AnalyzeTaxonomyCoverageInput(taxonomy_path="/tmp/t.json", concurrency=0)
        with pytest.raises(ValidationError):
            AnalyzeTaxonomyCoverageInput(taxonomy_path="/tmp/t.json", concurrency=51)

    def test_rejects_max_leaf_nodes_out_of_range(self):
        from src.models.schemas import AnalyzeTaxonomyCoverageInput

        with pytest.raises(ValidationError):
            AnalyzeTaxonomyCoverageInput(taxonomy_path="/tmp/t.json", max_leaf_nodes=0)
        with pytest.raises(ValidationError):
            AnalyzeTaxonomyCoverageInput(taxonomy_path="/tmp/t.json", max_leaf_nodes=5001)


# ═══════════════════════════════════════════════════════════════════════
# AC-WF6.4: Tool handlers — validate → dispatch → sanitize
# ═══════════════════════════════════════════════════════════════════════


class TestConvertPDFHandler:
    """Test the convert_pdf tool handler."""

    @pytest.mark.asyncio
    async def test_dispatches_correct_tool_name(self, mock_dispatcher, sanitizer):
        from src.tools.convert_pdf import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        await handler(input_path="/tmp/book.pdf")

        mock_dispatcher.dispatch.assert_called_once()
        assert mock_dispatcher.dispatch.call_args[0][0] == "convert_pdf"

    @pytest.mark.asyncio
    async def test_dispatches_validated_payload(self, mock_dispatcher, sanitizer):
        from src.tools.convert_pdf import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        await handler(input_path="/tmp/book.pdf", output_path="/tmp/out.json", enable_ocr=False)

        payload = mock_dispatcher.dispatch.call_args[0][1]
        assert payload["input_path"] == "/tmp/book.pdf"
        assert payload["output_path"] == "/tmp/out.json"
        assert payload["enable_ocr"] is False

    @pytest.mark.asyncio
    async def test_returns_sanitized_body(self, mock_dispatcher, sanitizer):
        from src.tools.convert_pdf import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        result = await handler(input_path="/tmp/book.pdf")
        assert "output_path" in result

    @pytest.mark.asyncio
    async def test_rejects_empty_input_path(self, mock_dispatcher, sanitizer):
        from src.tools.convert_pdf import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        with pytest.raises(ValidationError):
            await handler(input_path="")


class TestExtractBookMetadataHandler:
    """Test the extract_book_metadata tool handler."""

    @pytest.mark.asyncio
    async def test_dispatches_correct_tool_name(self, mock_dispatcher, sanitizer):
        from src.tools.extract_book_metadata import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        await handler(input_path="/tmp/book.json")

        mock_dispatcher.dispatch.assert_called_once()
        assert mock_dispatcher.dispatch.call_args[0][0] == "extract_book_metadata"

    @pytest.mark.asyncio
    async def test_dispatches_validated_payload(self, mock_dispatcher, sanitizer):
        from src.tools.extract_book_metadata import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        await handler(
            input_path="/tmp/book.json",
            output_path="/tmp/meta.json",
            chapters=[{"number": 1, "title": "Ch1", "start_page": 1, "end_page": 10}],
            options={"extract_code": True},
        )

        payload = mock_dispatcher.dispatch.call_args[0][1]
        assert payload["input_path"] == "/tmp/book.json"
        assert payload["output_path"] == "/tmp/meta.json"
        assert len(payload["chapters"]) == 1
        assert payload["options"]["extract_code"] is True

    @pytest.mark.asyncio
    async def test_returns_sanitized_body(self, mock_dispatcher, sanitizer):
        from src.tools.extract_book_metadata import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        result = await handler(input_path="/tmp/book.json")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_rejects_empty_input_path(self, mock_dispatcher, sanitizer):
        from src.tools.extract_book_metadata import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        with pytest.raises(ValidationError):
            await handler(input_path="")


class TestBatchExtractMetadataHandler:
    """Test the batch_extract_metadata tool handler — 295-line dispatch loop."""

    @pytest.fixture
    def books_dir(self, tmp_path):
        """Create a temp directory with 3 book JSON files.

        Uses a nested subdir so sibling 'metadata' dir is unique per test.
        """
        d = tmp_path / "books"
        d.mkdir()
        for name in ["book_alpha.json", "book_beta.json", "book_gamma.json"]:
            (d / name).write_text('{"title": "test"}')
        return d

    @pytest.fixture
    def co_success_result(self):
        """CO response for a successful single-book extraction."""
        return _make_result({
            "total_chapters": 5,
            "unique_keywords": 42,
            "unique_concepts": 18,
            "total_code_blocks": 7,
            "total_ascii_diagrams": 2,
            "output_path": "/tmp/meta/book_metadata.json",
            "chapter_results": [
                {
                    "chapter_number": 1,
                    "title": "Introduction",
                    "keywords_count": 10,
                    "concepts_count": 4,
                    "summary_length": 300,
                    "code_blocks_count": 2,
                    "status": "success",
                },
                {
                    "chapter_number": 2,
                    "title": "Basics",
                    "keywords_count": 8,
                    "concepts_count": 3,
                    "summary_length": 250,
                    "code_blocks_count": 1,
                    "status": "success",
                },
            ],
        })

    # ── Discovery & dispatching ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_discovers_books_in_directory(self, books_dir, co_success_result, sanitizer):
        """Handler discovers all .json files in input_dir."""
        d = AsyncMock(spec=ToolDispatcher)
        d.dispatch = AsyncMock(return_value=co_success_result)

        from src.tools.batch_extract_metadata import create_handler

        handler = create_handler(d, sanitizer)
        result = await handler(input_dir=str(books_dir))

        assert result["total_processed"] == 3
        assert d.dispatch.call_count == 3

    @pytest.mark.asyncio
    async def test_dispatches_extract_book_metadata_tool(self, books_dir, co_success_result, sanitizer):
        """Each book dispatches with tool_name='extract_book_metadata'."""
        d = AsyncMock(spec=ToolDispatcher)
        d.dispatch = AsyncMock(return_value=co_success_result)

        from src.tools.batch_extract_metadata import create_handler

        handler = create_handler(d, sanitizer)
        await handler(input_dir=str(books_dir))

        for call in d.dispatch.call_args_list:
            assert call[0][0] == "extract_book_metadata"

    @pytest.mark.asyncio
    async def test_dispatches_correct_payload_per_book(self, books_dir, co_success_result, sanitizer):
        """Each dispatch sends the correct input_path and output_path."""
        d = AsyncMock(spec=ToolDispatcher)
        d.dispatch = AsyncMock(return_value=co_success_result)

        from src.tools.batch_extract_metadata import create_handler

        handler = create_handler(d, sanitizer)
        out_dir = str(books_dir / "output")
        await handler(input_dir=str(books_dir), output_dir=out_dir)

        dispatched_inputs = [call[0][1]["input_path"] for call in d.dispatch.call_args_list]
        dispatched_outputs = [call[0][1]["output_path"] for call in d.dispatch.call_args_list]

        for inp in dispatched_inputs:
            assert inp.startswith(str(books_dir))
            assert inp.endswith(".json")

        for out in dispatched_outputs:
            assert out.startswith(out_dir)
            assert out.endswith("_metadata.json")

    # ── Empty directory ─────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_empty_directory_returns_no_files(self, tmp_path, sanitizer):
        """Empty dir returns status='no_files'."""
        d = AsyncMock(spec=ToolDispatcher)

        from src.tools.batch_extract_metadata import create_handler

        handler = create_handler(d, sanitizer)
        result = await handler(input_dir=str(tmp_path))

        assert result["status"] == "no_files"
        d.dispatch.assert_not_called()

    # ── Skip existing ───────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_skip_existing_skips_processed_books(self, books_dir, co_success_result, sanitizer):
        """Books with existing metadata files are skipped."""
        import os

        # Default output dir is sibling "metadata" dir
        meta_dir = books_dir.parent / "metadata"
        meta_dir.mkdir(exist_ok=True)
        # Pre-create metadata for book_alpha
        (meta_dir / "book_alpha_metadata.json").write_text("{}")

        d = AsyncMock(spec=ToolDispatcher)
        d.dispatch = AsyncMock(return_value=co_success_result)

        from src.tools.batch_extract_metadata import create_handler

        handler = create_handler(d, sanitizer)
        result = await handler(input_dir=str(books_dir), skip_existing=True)

        assert result["skipped"] == 1
        assert result["total_processed"] == 2
        assert d.dispatch.call_count == 2

    @pytest.mark.asyncio
    async def test_skip_existing_false_processes_all(self, books_dir, co_success_result, sanitizer):
        """skip_existing=False processes all books even if metadata exists."""
        meta_dir = books_dir.parent / "metadata"
        meta_dir.mkdir(exist_ok=True)
        (meta_dir / "book_alpha_metadata.json").write_text("{}")

        d = AsyncMock(spec=ToolDispatcher)
        d.dispatch = AsyncMock(return_value=co_success_result)

        from src.tools.batch_extract_metadata import create_handler

        handler = create_handler(d, sanitizer)
        result = await handler(input_dir=str(books_dir), skip_existing=False)

        assert result["skipped"] == 0
        assert result["total_processed"] == 3

    @pytest.mark.asyncio
    async def test_all_skipped_returns_all_skipped(self, books_dir, sanitizer):
        """When all books have existing metadata, return status='all_skipped'."""
        meta_dir = books_dir.parent / "metadata"
        meta_dir.mkdir(exist_ok=True)
        for name in ["book_alpha", "book_beta", "book_gamma"]:
            (meta_dir / f"{name}_metadata.json").write_text("{}")

        d = AsyncMock(spec=ToolDispatcher)

        from src.tools.batch_extract_metadata import create_handler

        handler = create_handler(d, sanitizer)
        result = await handler(input_dir=str(books_dir), skip_existing=True)

        assert result["status"] == "all_skipped"
        assert result["skipped"] == 3
        d.dispatch.assert_not_called()

    # ── Output directory resolution ─────────────────────────────────

    @pytest.mark.asyncio
    async def test_default_output_dir_is_sibling_metadata(self, books_dir, co_success_result, sanitizer):
        """Without output_dir, handler creates a sibling 'metadata' dir."""
        import os

        d = AsyncMock(spec=ToolDispatcher)
        d.dispatch = AsyncMock(return_value=co_success_result)

        from src.tools.batch_extract_metadata import create_handler

        handler = create_handler(d, sanitizer)
        await handler(input_dir=str(books_dir))

        expected_meta_dir = os.path.join(os.path.dirname(str(books_dir).rstrip("/")), "metadata")
        assert os.path.isdir(expected_meta_dir)

    @pytest.mark.asyncio
    async def test_explicit_output_dir_is_created(self, books_dir, co_success_result, sanitizer):
        """Explicit output_dir is created if it doesn't exist."""
        import os

        custom_out = str(books_dir / "custom_output")

        d = AsyncMock(spec=ToolDispatcher)
        d.dispatch = AsyncMock(return_value=co_success_result)

        from src.tools.batch_extract_metadata import create_handler

        handler = create_handler(d, sanitizer)
        await handler(input_dir=str(books_dir), output_dir=custom_out)

        assert os.path.isdir(custom_out)

    # ── Success result structure ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_success_result_structure(self, books_dir, co_success_result, sanitizer):
        """Verify the complete result dict on successful batch."""
        d = AsyncMock(spec=ToolDispatcher)
        d.dispatch = AsyncMock(return_value=co_success_result)

        from src.tools.batch_extract_metadata import create_handler

        handler = create_handler(d, sanitizer)
        result = await handler(input_dir=str(books_dir))

        assert result["status"] == "complete"
        assert result["total_processed"] == 3
        assert result["succeeded"] == 3
        assert result["failed"] == 0
        assert result["skipped"] == 0
        assert isinstance(result["duration_s"], float)
        assert len(result["results"]) == 3

    @pytest.mark.asyncio
    async def test_per_book_result_fields(self, books_dir, co_success_result, sanitizer):
        """Each book result has expected fields from CO response."""
        d = AsyncMock(spec=ToolDispatcher)
        d.dispatch = AsyncMock(return_value=co_success_result)

        from src.tools.batch_extract_metadata import create_handler

        handler = create_handler(d, sanitizer)
        result = await handler(input_dir=str(books_dir))

        book_result = result["results"][0]
        assert book_result["status"] == "success"
        assert book_result["total_chapters"] == 5
        assert book_result["unique_keywords"] == 42
        assert book_result["unique_concepts"] == 18
        assert book_result["total_code_blocks"] == 7
        assert book_result["total_ascii_diagrams"] == 2
        assert isinstance(book_result["elapsed_s"], float)

    # ── Error handling ──────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_dispatch_error_counted_as_failed(self, books_dir, sanitizer):
        """A book that raises on dispatch is counted as failed."""
        d = AsyncMock(spec=ToolDispatcher)
        d.dispatch = AsyncMock(side_effect=RuntimeError("CO connection refused"))

        from src.tools.batch_extract_metadata import create_handler

        handler = create_handler(d, sanitizer)
        result = await handler(input_dir=str(books_dir))

        assert result["status"] == "complete"
        assert result["succeeded"] == 0
        assert result["failed"] == 3
        assert all(r["status"] == "failed" for r in result["results"])
        assert all("CO connection refused" in r["error"] for r in result["results"])

    @pytest.mark.asyncio
    async def test_partial_failure_mixed_results(self, books_dir, co_success_result, sanitizer):
        """Mix of success and failure: counts are correct."""
        call_count = 0

        async def alternating_dispatch(tool_name, payload):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Timeout on book 2")
            return co_success_result

        d = AsyncMock(spec=ToolDispatcher)
        d.dispatch = AsyncMock(side_effect=alternating_dispatch)

        from src.tools.batch_extract_metadata import create_handler

        handler = create_handler(d, sanitizer)
        result = await handler(input_dir=str(books_dir))

        assert result["succeeded"] == 2
        assert result["failed"] == 1
        assert result["total_processed"] == 3

    @pytest.mark.asyncio
    async def test_unexpected_body_type_counted_as_failed(self, books_dir, sanitizer):
        """Non-dict CO response body is counted as failed."""
        bad_result = _make_result({"output_path": "/tmp/out.json"})
        # Simulate sanitizer returning a non-dict (e.g., a list)
        s = OutputSanitizer()

        d = AsyncMock(spec=ToolDispatcher)
        d.dispatch = AsyncMock(return_value=_make_result({}))

        from src.tools.batch_extract_metadata import create_handler

        # Monkey-patch the sanitizer to return a string
        original_sanitize = s.sanitize
        call_idx = [0]

        def bad_sanitize(body):
            call_idx[0] += 1
            if call_idx[0] == 1:
                return "not a dict"  # type: ignore
            return original_sanitize(body)

        s.sanitize = bad_sanitize

        handler = create_handler(d, s)
        result = await handler(input_dir=str(books_dir))

        # First book gets bad response, rest are fine
        assert result["failed"] >= 1

    # ── Custom file pattern ─────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_custom_file_pattern_filters_correctly(self, tmp_path, co_success_result, sanitizer):
        """Only files matching file_pattern are discovered."""
        (tmp_path / "book_one.json").write_text("{}")
        (tmp_path / "book_two.json").write_text("{}")
        (tmp_path / "notes.txt").write_text("ignore me")
        (tmp_path / "readme.md").write_text("ignore me")

        d = AsyncMock(spec=ToolDispatcher)
        d.dispatch = AsyncMock(return_value=co_success_result)

        from src.tools.batch_extract_metadata import create_handler

        handler = create_handler(d, sanitizer)
        result = await handler(input_dir=str(tmp_path), file_pattern="*.json")

        assert result["total_processed"] == 2

    @pytest.mark.asyncio
    async def test_wildcard_pattern_matches_all(self, tmp_path, co_success_result, sanitizer):
        """file_pattern='*' discovers all files."""
        (tmp_path / "a.json").write_text("{}")
        (tmp_path / "b.txt").write_text("{}")

        d = AsyncMock(spec=ToolDispatcher)
        d.dispatch = AsyncMock(return_value=co_success_result)

        from src.tools.batch_extract_metadata import create_handler

        handler = create_handler(d, sanitizer)
        result = await handler(input_dir=str(tmp_path), file_pattern="*")

        assert result["total_processed"] == 2

    # ── Input validation ────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_rejects_empty_input_dir(self, sanitizer):
        """Empty input_dir is rejected by Pydantic validation."""
        d = AsyncMock(spec=ToolDispatcher)

        from src.tools.batch_extract_metadata import create_handler

        handler = create_handler(d, sanitizer)
        with pytest.raises(ValidationError):
            await handler(input_dir="")

    # ── Progress context reporting ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_ctx_receives_progress_reports(self, books_dir, co_success_result, sanitizer):
        """When ctx is provided, it receives info() and report_progress() calls."""
        d = AsyncMock(spec=ToolDispatcher)
        d.dispatch = AsyncMock(return_value=co_success_result)

        ctx = AsyncMock()
        ctx.info = AsyncMock()
        ctx.report_progress = AsyncMock()

        from src.tools.batch_extract_metadata import create_handler

        handler = create_handler(d, sanitizer)
        await handler(input_dir=str(books_dir), ctx=ctx)

        # At minimum: batch header + per-book start + per-chapter + completion + final summary
        assert ctx.info.call_count > 0
        assert ctx.report_progress.call_count > 0

    @pytest.mark.asyncio
    async def test_runs_without_ctx(self, books_dir, co_success_result, sanitizer):
        """Handler works fine when ctx=None (no MCP context)."""
        d = AsyncMock(spec=ToolDispatcher)
        d.dispatch = AsyncMock(return_value=co_success_result)

        from src.tools.batch_extract_metadata import create_handler

        handler = create_handler(d, sanitizer)
        result = await handler(input_dir=str(books_dir), ctx=None)

        assert result["status"] == "complete"
        assert result["succeeded"] == 3


class TestGenerateTaxonomyHandler:
    """Test the generate_taxonomy tool handler."""

    @pytest.mark.asyncio
    async def test_dispatches_correct_tool_name(self, mock_dispatcher, sanitizer):
        from src.tools.generate_taxonomy import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        await handler(tier_books={"beginner": ["/tmp/b1.json"]})

        mock_dispatcher.dispatch.assert_called_once()
        assert mock_dispatcher.dispatch.call_args[0][0] == "generate_taxonomy"

    @pytest.mark.asyncio
    async def test_dispatches_validated_payload(self, mock_dispatcher, sanitizer):
        from src.tools.generate_taxonomy import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        await handler(
            tier_books={"beginner": ["/tmp/b1.json"]},
            output_path="/tmp/tax.json",
            concepts=["OOP"],
            domain="python",
        )

        payload = mock_dispatcher.dispatch.call_args[0][1]
        assert payload["tier_books"] == {"beginner": ["/tmp/b1.json"]}
        assert payload["output_path"] == "/tmp/tax.json"
        assert payload["concepts"] == ["OOP"]
        assert payload["domain"] == "python"

    @pytest.mark.asyncio
    async def test_returns_sanitized_body(self, mock_dispatcher, sanitizer):
        from src.tools.generate_taxonomy import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        result = await handler(tier_books={"t": ["/tmp/b.json"]})
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_rejects_empty_tier_books(self, mock_dispatcher, sanitizer):
        from src.tools.generate_taxonomy import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        with pytest.raises(ValidationError):
            await handler(tier_books={})


class TestEnrichBookMetadataHandler:
    """Test the enrich_book_metadata tool handler."""

    @pytest.mark.asyncio
    async def test_dispatches_correct_tool_name(self, mock_dispatcher, sanitizer):
        from src.tools.enrich_book_metadata import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        await handler(input_path="/tmp/book.json")

        mock_dispatcher.dispatch.assert_called_once()
        assert mock_dispatcher.dispatch.call_args[0][0] == "enrich_book_metadata"

    @pytest.mark.asyncio
    async def test_dispatches_validated_payload(self, mock_dispatcher, sanitizer):
        from src.tools.enrich_book_metadata import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        await handler(
            input_path="/tmp/book.json",
            output_path="/tmp/enriched.json",
            taxonomy_path="/tmp/taxonomy.json",
            mode="msep",
        )

        payload = mock_dispatcher.dispatch.call_args[0][1]
        assert payload["input_path"] == "/tmp/book.json"
        assert payload["output_path"] == "/tmp/enriched.json"
        assert payload["taxonomy_path"] == "/tmp/taxonomy.json"
        assert payload["mode"] == "msep"

    @pytest.mark.asyncio
    async def test_returns_sanitized_body(self, mock_dispatcher, sanitizer):
        from src.tools.enrich_book_metadata import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        result = await handler(input_path="/tmp/book.json")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_rejects_empty_input_path(self, mock_dispatcher, sanitizer):
        from src.tools.enrich_book_metadata import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        with pytest.raises(ValidationError):
            await handler(input_path="")


class TestEnhanceGuidelineHandler:
    """Test the enhance_guideline tool handler."""

    @pytest.mark.asyncio
    async def test_dispatches_correct_tool_name(self, mock_dispatcher, sanitizer):
        from src.tools.enhance_guideline import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        await handler(
            aggregate_path="/tmp/agg.json",
            guideline_path="/tmp/guide.json",
        )

        mock_dispatcher.dispatch.assert_called_once()
        assert mock_dispatcher.dispatch.call_args[0][0] == "enhance_guideline"

    @pytest.mark.asyncio
    async def test_dispatches_validated_payload(self, mock_dispatcher, sanitizer):
        from src.tools.enhance_guideline import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        await handler(
            aggregate_path="/tmp/agg.json",
            guideline_path="/tmp/guide.json",
            output_dir="/tmp/enhanced",
            provider="anthropic",
            max_tokens=8192,
            temperature=0.5,
        )

        payload = mock_dispatcher.dispatch.call_args[0][1]
        assert payload["aggregate_path"] == "/tmp/agg.json"
        assert payload["guideline_path"] == "/tmp/guide.json"
        assert payload["output_dir"] == "/tmp/enhanced"
        assert payload["provider"] == "anthropic"
        assert payload["max_tokens"] == 8192
        assert payload["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_returns_sanitized_body(self, mock_dispatcher, sanitizer):
        from src.tools.enhance_guideline import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        result = await handler(
            aggregate_path="/tmp/agg.json",
            guideline_path="/tmp/guide.json",
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_rejects_empty_aggregate_path(self, mock_dispatcher, sanitizer):
        from src.tools.enhance_guideline import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        with pytest.raises(ValidationError):
            await handler(aggregate_path="", guideline_path="/tmp/guide.json")

    @pytest.mark.asyncio
    async def test_rejects_empty_guideline_path(self, mock_dispatcher, sanitizer):
        from src.tools.enhance_guideline import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        with pytest.raises(ValidationError):
            await handler(aggregate_path="/tmp/agg.json", guideline_path="")


class TestAnalyzeTaxonomyCoverageHandler:
    """Test the analyze_taxonomy_coverage tool handler (WBS-TAP9)."""

    @pytest.mark.asyncio
    async def test_dispatches_correct_tool_name(self, mock_dispatcher, sanitizer):
        from src.tools.analyze_taxonomy_coverage import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        await handler(taxonomy_path="/tmp/taxonomy.json")

        mock_dispatcher.dispatch.assert_called_once()
        assert mock_dispatcher.dispatch.call_args[0][0] == "analyze_taxonomy_coverage"

    @pytest.mark.asyncio
    async def test_dispatches_validated_payload(self, mock_dispatcher, sanitizer):
        from src.tools.analyze_taxonomy_coverage import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        await handler(
            taxonomy_path="/tmp/taxonomy.json",
            output_path="/tmp/out.json",
            collection="code_chunks",
            top_k=20,
            threshold=0.5,
            max_leaf_nodes=1000,
            subtree_root="algorithms",
            concurrency=5,
            include_evidence=False,
            scoring_weights={"semantic": 0.7, "keyword": 0.3},
        )

        payload = mock_dispatcher.dispatch.call_args[0][1]
        assert payload["taxonomy_path"] == "/tmp/taxonomy.json"
        assert payload["output_path"] == "/tmp/out.json"
        assert payload["collection"] == "code_chunks"
        assert payload["top_k"] == 20
        assert payload["threshold"] == 0.5
        assert payload["max_leaf_nodes"] == 1000
        assert payload["subtree_root"] == "algorithms"
        assert payload["concurrency"] == 5
        assert payload["include_evidence"] is False
        assert payload["scoring_weights"] == {"semantic": 0.7, "keyword": 0.3}

    @pytest.mark.asyncio
    async def test_returns_sanitized_body(self, mock_dispatcher, sanitizer):
        from src.tools.analyze_taxonomy_coverage import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        result = await handler(taxonomy_path="/tmp/taxonomy.json")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_rejects_empty_taxonomy_path(self, mock_dispatcher, sanitizer):
        from src.tools.analyze_taxonomy_coverage import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        with pytest.raises(ValidationError):
            await handler(taxonomy_path="")


# ═══════════════════════════════════════════════════════════════════════
# AC-WF6.5: tools/list — 14 tools via FastMCP Client
# ═══════════════════════════════════════════════════════════════════════


EXPECTED_ALL_TOOL_NAMES = {
    # Original 9
    "semantic_search",
    "hybrid_search",
    "code_analyze",
    "code_pattern_audit",
    "graph_query",
    "llm_complete",
    "a2a_send_message",
    "a2a_get_task",
    "a2a_cancel_task",
    # 6 workflow tools
    "convert_pdf",
    "extract_book_metadata",
    "batch_extract_metadata",
    "generate_taxonomy",
    "enrich_book_metadata",
    "enhance_guideline",
    # Taxonomy Analysis (WBS-TAP9)
    "analyze_taxonomy_coverage",
}


class TestToolsListWorkflow:
    """AC-WF6.5: tools/list returns all 14 tools."""

    @pytest.fixture()
    def registry(self, tmp_path):
        """Create a registry with all 14 tools."""
        from pathlib import Path

        from src.tool_registry import ToolRegistry

        config_path = Path(__file__).resolve().parents[2] / "config" / "tools.yaml"
        return ToolRegistry(config_path)

    @pytest.fixture()
    def mcp_server(self, registry, mock_dispatcher, sanitizer):
        from src.server import create_mcp_server

        return create_mcp_server(registry, mock_dispatcher, sanitizer)

    async def test_returns_16_tools(self, mcp_server):
        from fastmcp import Client

        async with Client(mcp_server) as client:
            tools = await client.list_tools()
        assert len(tools) == 16

    async def test_all_tool_names_present(self, mcp_server):
        from fastmcp import Client

        async with Client(mcp_server) as client:
            tools = await client.list_tools()
        names = {t.name for t in tools}
        assert names == EXPECTED_ALL_TOOL_NAMES

    async def test_workflow_tools_have_descriptions(self, mcp_server):
        from fastmcp import Client

        async with Client(mcp_server) as client:
            tools = await client.list_tools()
        workflow_tools = [t for t in tools if t.name in WORKFLOW_TOOLS]
        assert len(workflow_tools) == 7
        for tool in workflow_tools:
            assert tool.description, f"{tool.name} missing description"

    async def test_convert_pdf_schema_fields(self, mcp_server):
        from fastmcp import Client

        async with Client(mcp_server) as client:
            tools = await client.list_tools()
        tool = next(t for t in tools if t.name == "convert_pdf")
        props = tool.inputSchema["properties"]
        assert "input_path" in props
        assert "enable_ocr" in props

    async def test_extract_book_metadata_schema_fields(self, mcp_server):
        from fastmcp import Client

        async with Client(mcp_server) as client:
            tools = await client.list_tools()
        tool = next(t for t in tools if t.name == "extract_book_metadata")
        props = tool.inputSchema["properties"]
        assert "input_path" in props

    async def test_batch_extract_metadata_schema_fields(self, mcp_server):
        from fastmcp import Client

        async with Client(mcp_server) as client:
            tools = await client.list_tools()
        tool = next(t for t in tools if t.name == "batch_extract_metadata")
        props = tool.inputSchema["properties"]
        assert "input_dir" in props
        assert "output_dir" in props
        assert "file_pattern" in props
        assert "skip_existing" in props

    async def test_generate_taxonomy_schema_fields(self, mcp_server):
        from fastmcp import Client

        async with Client(mcp_server) as client:
            tools = await client.list_tools()
        tool = next(t for t in tools if t.name == "generate_taxonomy")
        props = tool.inputSchema["properties"]
        assert "tier_books" in props
        assert "domain" in props

    async def test_enrich_book_metadata_schema_fields(self, mcp_server):
        from fastmcp import Client

        async with Client(mcp_server) as client:
            tools = await client.list_tools()
        tool = next(t for t in tools if t.name == "enrich_book_metadata")
        props = tool.inputSchema["properties"]
        assert "input_path" in props
        assert "mode" in props

    async def test_enhance_guideline_schema_fields(self, mcp_server):
        from fastmcp import Client

        async with Client(mcp_server) as client:
            tools = await client.list_tools()
        tool = next(t for t in tools if t.name == "enhance_guideline")
        props = tool.inputSchema["properties"]
        assert "aggregate_path" in props
        assert "guideline_path" in props
        assert "provider" in props
        assert "max_tokens" in props

    async def test_analyze_taxonomy_coverage_schema_fields(self, mcp_server):
        from fastmcp import Client

        async with Client(mcp_server) as client:
            tools = await client.list_tools()
        tool = next(t for t in tools if t.name == "analyze_taxonomy_coverage")
        props = tool.inputSchema["properties"]
        assert "taxonomy_path" in props
        assert "collection" in props
        assert "top_k" in props
        assert "threshold" in props
        assert "max_leaf_nodes" in props
        assert "concurrency" in props
        assert "include_evidence" in props

    async def test_workflow_tools_call_succeeds(self, mcp_server, mock_dispatcher):
        """tools/call for each workflow tool should succeed."""
        from fastmcp import Client

        test_cases = [
            ("convert_pdf", {"input_path": "/tmp/book.pdf"}),
            ("extract_book_metadata", {"input_path": "/tmp/book.json"}),
            ("generate_taxonomy", {"tier_books": {"t": ["/tmp/b.json"]}}),
            ("enrich_book_metadata", {"input_path": "/tmp/book.json"}),
            ("enhance_guideline", {"aggregate_path": "/tmp/agg.json", "guideline_path": "/tmp/guide.json"}),
            ("analyze_taxonomy_coverage", {"taxonomy_path": "/tmp/taxonomy.json"}),
        ]

        async with Client(mcp_server) as client:
            for tool_name, args in test_cases:
                result = await client.call_tool(tool_name, args)
                assert not result.is_error, f"{tool_name} call failed"
