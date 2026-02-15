"""Tests for analyze_taxonomy_coverage handler — WBS-TAP9 (RED).

AC-TAP9.5: Handler factory follows create_handler() pattern.
AC-TAP9.2: Input validation via AnalyzeTaxonomyCoverageInput schema.
"""

from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

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
def mock_dispatcher():
    d = AsyncMock(spec=ToolDispatcher)
    d.dispatch = AsyncMock(
        return_value=_make_result(
            {
                "taxonomy_id": "test-taxonomy",
                "overall_score": 0.72,
                "overall_label": "moderate",
                "total_nodes_analyzed": 10,
                "leaf_nodes_analyzed": 5,
                "output_path": "/tmp/report.json",
            }
        )
    )
    return d


@pytest.fixture
def sanitizer():
    return OutputSanitizer()


# ═══════════════════════════════════════════════════════════════════════
# AC-TAP9.5: create_handler() factory
# ═══════════════════════════════════════════════════════════════════════


class TestAnalyzeTaxonomyCoverageHandler:
    """AC-TAP9.5: Handler factory follows create_handler() pattern."""

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
            output_path="/tmp/report.json",
            collection="textbooks",
            top_k=20,
            threshold=0.5,
            max_leaf_nodes=200,
            subtree_root="PRINCIPLE_A",
            concurrency=5,
            include_evidence=False,
            scoring_weights={"breadth": 0.5, "depth": 0.3, "spread": 0.2},
        )

        payload = mock_dispatcher.dispatch.call_args[0][1]
        assert payload["taxonomy_path"] == "/tmp/taxonomy.json"
        assert payload["output_path"] == "/tmp/report.json"
        assert payload["collection"] == "textbooks"
        assert payload["top_k"] == 20
        assert payload["threshold"] == 0.5
        assert payload["max_leaf_nodes"] == 200
        assert payload["subtree_root"] == "PRINCIPLE_A"
        assert payload["concurrency"] == 5
        assert payload["include_evidence"] is False
        assert payload["scoring_weights"] == {"breadth": 0.5, "depth": 0.3, "spread": 0.2}

    @pytest.mark.asyncio
    async def test_returns_sanitized_body(self, mock_dispatcher, sanitizer):
        from src.tools.analyze_taxonomy_coverage import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        result = await handler(taxonomy_path="/tmp/taxonomy.json")
        assert isinstance(result, dict)
        assert "taxonomy_id" in result

    @pytest.mark.asyncio
    async def test_rejects_empty_taxonomy_path(self, mock_dispatcher, sanitizer):
        from src.tools.analyze_taxonomy_coverage import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        with pytest.raises(ValidationError):
            await handler(taxonomy_path="")

    @pytest.mark.asyncio
    async def test_rejects_invalid_top_k(self, mock_dispatcher, sanitizer):
        from src.tools.analyze_taxonomy_coverage import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        with pytest.raises(ValidationError):
            await handler(taxonomy_path="/tmp/tax.json", top_k=0)

    @pytest.mark.asyncio
    async def test_rejects_invalid_threshold(self, mock_dispatcher, sanitizer):
        from src.tools.analyze_taxonomy_coverage import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        with pytest.raises(ValidationError):
            await handler(taxonomy_path="/tmp/tax.json", threshold=1.5)

    @pytest.mark.asyncio
    async def test_rejects_invalid_concurrency(self, mock_dispatcher, sanitizer):
        from src.tools.analyze_taxonomy_coverage import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        with pytest.raises(ValidationError):
            await handler(taxonomy_path="/tmp/tax.json", concurrency=0)

    @pytest.mark.asyncio
    async def test_defaults_applied(self, mock_dispatcher, sanitizer):
        from src.tools.analyze_taxonomy_coverage import create_handler

        handler = create_handler(mock_dispatcher, sanitizer)
        await handler(taxonomy_path="/tmp/taxonomy.json")

        payload = mock_dispatcher.dispatch.call_args[0][1]
        assert payload["collection"] == "all"
        assert payload["top_k"] == 10
        assert payload["threshold"] == 0.3
        assert payload["max_leaf_nodes"] == 500
        assert payload["concurrency"] == 10
        assert payload["include_evidence"] is True

    @pytest.mark.asyncio
    async def test_tool_name_constant(self):
        from src.tools.analyze_taxonomy_coverage import TOOL_NAME

        assert TOOL_NAME == "analyze_taxonomy_coverage"
