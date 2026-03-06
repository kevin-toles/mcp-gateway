"""
Unit tests for MCP gateway tier parameter exposure — WBS-TXS5.

TDD RED phase: tests written before implementation.
All tests in this file should FAIL until TXS5 GREEN phase adds:
    - bloom_tier_filter, quality_tier_filter, bloom_tier_boost fields to HybridSearchInput
    - These params to the hybrid_search() handler signature
    - Conditional forwarding in the dispatch payload

Acceptance Criteria:
- AC-TXS5.1: hybrid_search MCP tool accepts bloom_tier_filter (list[int] 0-6)
- AC-TXS5.2: hybrid_search MCP tool accepts quality_tier_filter (list[int] 1-3)
- AC-TXS5.3: hybrid_search MCP tool accepts bloom_tier_boost (bool, default True)
- AC-TXS5.4: All three params forwarded to semantic-search dispatch payload
- AC-TXS5.5: Omitting all new params produces backward-compatible payload
- AC-TXS5.6: Invalid bloom_tier_filter values ([7]) rejected with validation error
"""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import ValidationError

from src.core.config import Settings
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def dispatcher() -> ToolDispatcher:
    return ToolDispatcher(Settings())


@pytest.fixture
def sanitizer() -> OutputSanitizer:
    return OutputSanitizer()


# ─────────────────────────────────────────────────────────────────────────────
# AC-TXS5.1  bloom_tier_filter accepted by HybridSearchInput
# ─────────────────────────────────────────────────────────────────────────────


class TestBloomTierFilterSchema:
    """AC-TXS5.1: HybridSearchInput accepts bloom_tier_filter (list[int] 0-6)."""

    def test_bloom_tier_filter_single_valid_value_accepted(self) -> None:
        from src.models.schemas import HybridSearchInput

        m = HybridSearchInput(query="test", bloom_tier_filter=[2])
        assert m.bloom_tier_filter == [2]

    def test_bloom_tier_filter_multiple_valid_values_accepted(self) -> None:
        from src.models.schemas import HybridSearchInput

        m = HybridSearchInput(query="test", bloom_tier_filter=[2, 3, 4])
        assert m.bloom_tier_filter == [2, 3, 4]

    def test_bloom_tier_filter_full_range_0_to_6_accepted(self) -> None:
        from src.models.schemas import HybridSearchInput

        m = HybridSearchInput(query="test", bloom_tier_filter=[0, 1, 2, 3, 4, 5, 6])
        assert len(m.bloom_tier_filter) == 7

    def test_bloom_tier_filter_none_by_default(self) -> None:
        from src.models.schemas import HybridSearchInput

        m = HybridSearchInput(query="test")
        assert m.bloom_tier_filter is None


# ─────────────────────────────────────────────────────────────────────────────
# AC-TXS5.2  quality_tier_filter accepted by HybridSearchInput
# ─────────────────────────────────────────────────────────────────────────────


class TestQualityTierFilterSchema:
    """AC-TXS5.2: HybridSearchInput accepts quality_tier_filter (list[int] 1-3)."""

    def test_quality_tier_filter_valid_values_accepted(self) -> None:
        from src.models.schemas import HybridSearchInput

        m = HybridSearchInput(query="test", quality_tier_filter=[1, 2])
        assert m.quality_tier_filter == [1, 2]

    def test_quality_tier_filter_none_by_default(self) -> None:
        from src.models.schemas import HybridSearchInput

        m = HybridSearchInput(query="test")
        assert m.quality_tier_filter is None


# ─────────────────────────────────────────────────────────────────────────────
# AC-TXS5.3  bloom_tier_boost accepted by HybridSearchInput
# ─────────────────────────────────────────────────────────────────────────────


class TestBloomTierBoostSchema:
    """AC-TXS5.3: HybridSearchInput accepts bloom_tier_boost (bool, default True)."""

    def test_bloom_tier_boost_true_accepted(self) -> None:
        from src.models.schemas import HybridSearchInput

        m = HybridSearchInput(query="test", bloom_tier_boost=True)
        assert m.bloom_tier_boost is True

    def test_bloom_tier_boost_false_accepted(self) -> None:
        from src.models.schemas import HybridSearchInput

        m = HybridSearchInput(query="test", bloom_tier_boost=False)
        assert m.bloom_tier_boost is False

    def test_bloom_tier_boost_defaults_to_true(self) -> None:
        from src.models.schemas import HybridSearchInput

        m = HybridSearchInput(query="test")
        assert m.bloom_tier_boost is True


# ─────────────────────────────────────────────────────────────────────────────
# AC-TXS5.4  All three params forwarded in dispatch payload
# ─────────────────────────────────────────────────────────────────────────────


class TestTierParamsForwardedInPayload:
    """AC-TXS5.4: bloom_tier_filter, quality_tier_filter, bloom_tier_boost
    are included in the payload dispatched to semantic-search-service."""

    @pytest.mark.asyncio
    async def test_bloom_tier_filter_in_dispatch_payload(
        self,
        dispatcher: ToolDispatcher,
        sanitizer: OutputSanitizer,
    ) -> None:
        from src.tools.hybrid_search import create_handler

        captured: dict = {}

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"results": [], "total": 0, "alpha": 0.7, "latency_ms": 1.0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
        handler = create_handler(dispatcher, sanitizer)
        await handler(query="design patterns", bloom_tier_filter=[2, 3])

        assert "bloom_tier_filter" in captured
        assert captured["bloom_tier_filter"] == [2, 3]

    @pytest.mark.asyncio
    async def test_quality_tier_filter_in_dispatch_payload(
        self,
        dispatcher: ToolDispatcher,
        sanitizer: OutputSanitizer,
    ) -> None:
        from src.tools.hybrid_search import create_handler

        captured: dict = {}

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"results": [], "total": 0, "alpha": 0.7, "latency_ms": 1.0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
        handler = create_handler(dispatcher, sanitizer)
        await handler(query="dependency injection", quality_tier_filter=[1])

        assert "quality_tier_filter" in captured
        assert captured["quality_tier_filter"] == [1]

    @pytest.mark.asyncio
    async def test_bloom_tier_boost_false_in_dispatch_payload(
        self,
        dispatcher: ToolDispatcher,
        sanitizer: OutputSanitizer,
    ) -> None:
        from src.tools.hybrid_search import create_handler

        captured: dict = {}

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"results": [], "total": 0, "alpha": 0.7, "latency_ms": 1.0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
        handler = create_handler(dispatcher, sanitizer)
        await handler(query="patterns", bloom_tier_boost=False)

        assert "tier_boost" in captured
        assert captured["tier_boost"] is False


# ─────────────────────────────────────────────────────────────────────────────
# AC-TXS5.5  Backward compatibility: omitting new params works identically
# ─────────────────────────────────────────────────────────────────────────────


class TestOmittingTierParameters:
    """AC-TXS5.5: Omitting new tier params produces backward-compatible payload."""

    @pytest.mark.asyncio
    async def test_payload_without_tier_params_does_not_include_bloom_filter(
        self,
        dispatcher: ToolDispatcher,
        sanitizer: OutputSanitizer,
    ) -> None:
        from src.tools.hybrid_search import create_handler

        captured: dict = {}

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"results": [], "total": 0, "alpha": 0.7, "latency_ms": 1.0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
        handler = create_handler(dispatcher, sanitizer)
        await handler(query="design patterns")

        # bloom_tier_filter must NOT appear in payload when not provided
        assert "bloom_tier_filter" not in captured

    @pytest.mark.asyncio
    async def test_payload_without_tier_params_does_not_include_quality_filter(
        self,
        dispatcher: ToolDispatcher,
        sanitizer: OutputSanitizer,
    ) -> None:
        from src.tools.hybrid_search import create_handler

        captured: dict = {}

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"results": [], "total": 0, "alpha": 0.7, "latency_ms": 1.0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
        handler = create_handler(dispatcher, sanitizer)
        await handler(query="design patterns")

        assert "quality_tier_filter" not in captured

    @pytest.mark.asyncio
    async def test_basic_payload_fields_still_present_without_tier_params(
        self,
        dispatcher: ToolDispatcher,
        sanitizer: OutputSanitizer,
    ) -> None:
        from src.tools.hybrid_search import create_handler

        captured: dict = {}

        async def mock_transport(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json={"results": [], "total": 0, "alpha": 0.7, "latency_ms": 1.0})

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_transport))
        handler = create_handler(dispatcher, sanitizer)
        await handler(query="design patterns")

        # Core fields must still be present (backward compat)
        assert "query" in captured
        assert "limit" in captured
        assert "alpha" in captured


# ─────────────────────────────────────────────────────────────────────────────
# AC-TXS5.6  Invalid bloom_tier_filter values rejected
# ─────────────────────────────────────────────────────────────────────────────


class TestTierFilterValidation:
    """AC-TXS5.6: Out-of-range tier filter values are rejected."""

    def test_bloom_tier_filter_value_7_raises_validation_error(self) -> None:
        from src.models.schemas import HybridSearchInput

        with pytest.raises((ValidationError, ValueError)):
            HybridSearchInput(query="test", bloom_tier_filter=[7])

    def test_bloom_tier_filter_value_negative_raises_validation_error(self) -> None:
        from src.models.schemas import HybridSearchInput

        with pytest.raises((ValidationError, ValueError)):
            HybridSearchInput(query="test", bloom_tier_filter=[-1])

    def test_quality_tier_filter_value_0_raises_validation_error(self) -> None:
        from src.models.schemas import HybridSearchInput

        with pytest.raises((ValidationError, ValueError)):
            HybridSearchInput(query="test", quality_tier_filter=[0])

    def test_quality_tier_filter_value_4_raises_validation_error(self) -> None:
        from src.models.schemas import HybridSearchInput

        with pytest.raises((ValidationError, ValueError)):
            HybridSearchInput(query="test", quality_tier_filter=[4])
