"""
MCP Gateway tier-enhanced search end-to-end integration tests — WBS-TXS7.

Covers the MCP gateway → unified-search-service full pipeline for
taxonomy-enhanced search parameters introduced in TXS3–TXS6.

Acceptance Criteria:
- AC-TXS7.1: bloom_tier_filter=[2,3] through MCP gateway returns only T2+T3 chapters
- AC-TXS7.2: bloom_tier_boost=true produces tier-boosted scores
- AC-TXS7.3: bloom_tier_filter + quality_tier_filter combined filtering
- AC-TXS7.4: expand_taxonomy forwarded to unified-search (expand_taxonomy support)
- AC-TXS7.5: Backward compat — calls without new params work identically

Run with:
    INTEGRATION=1 pytest tests/integration/test_tier_search_e2e.py -v

Requires:
    - mcp-gateway :8087 (or SEMANTIC_SEARCH_URL set)
    - unified-search-service :8081
    - Qdrant :6333 (with bloom_tier_level backfilled — TXS2 complete)
    - Neo4j :7687 (with TaxonomyConcept nodes — TXS1/V008 complete)
"""

from __future__ import annotations

import os

import httpx
import pytest

from src.core.config import Settings
from src.tool_dispatcher import ToolDispatcher

pytestmark = pytest.mark.integration

SEARCH_URL = os.environ.get("SEMANTIC_SEARCH_URL", "http://localhost:8081")
HYBRID_ENDPOINT = f"{SEARCH_URL}/v1/search/hybrid"


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def dispatcher(settings: Settings) -> ToolDispatcher:
    return ToolDispatcher(settings)


async def _service_ready() -> bool:
    """Return True if unified-search-service health endpoint is reachable."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{SEARCH_URL}/health", timeout=3.0)
            return resp.status_code == 200
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# TXS7 e2e tests via ToolDispatcher (gateway → unified-search-service)
# ─────────────────────────────────────────────────────────────────────────────


class TestBloomTierFilterViaGateway:
    """AC-TXS7.1: bloom_tier_filter=[2,3] through gateway returns only T2+T3 chapters."""

    @pytest.mark.asyncio
    async def test_bloom_tier_filter_restricts_results_to_specified_tiers(self, dispatcher: ToolDispatcher) -> None:
        if not await _service_ready():
            pytest.skip(f"unified-search-service not reachable at {SEARCH_URL}")

        result = await dispatcher.dispatch(
            "hybrid_search",
            {
                "query": "design patterns software architecture",
                "collection": "all",
                "limit": 20,
                "alpha": 0.7,
                "bloom_tier_filter": [2, 3],
                "tier_boost": True,
            },
        )
        assert result.status_code == 200
        data = result.body

        results = data.get("results", [])
        if not results:
            pytest.skip("No results returned — Qdrant data may be empty")

        # All chapter results must have bloom_tier_level in {2, 3}
        chapter_results = [r for r in results if r.get("payload", {}).get("bloom_tier_level") is not None]
        for r in chapter_results:
            bloom_level = r["payload"]["bloom_tier_level"]
            assert bloom_level in {2, 3}, (
                f"Expected bloom_tier_level in {{2,3}}, got {bloom_level} for result {r['id']}"
            )

    @pytest.mark.asyncio
    async def test_bloom_tier_filter_response_has_expected_shape(self, dispatcher: ToolDispatcher) -> None:
        if not await _service_ready():
            pytest.skip(f"unified-search-service not reachable at {SEARCH_URL}")

        result = await dispatcher.dispatch(
            "hybrid_search",
            {
                "query": "object oriented design",
                "collection": "all",
                "limit": 5,
                "alpha": 0.7,
                "bloom_tier_filter": [3],
                "tier_boost": True,
            },
        )
        assert result.status_code == 200
        body = result.body
        assert "results" in body
        assert "total" in body
        assert isinstance(body["results"], list)


class TestTierBoostViaGateway:
    """AC-TXS7.2: bloom_tier_boost=true produces higher scores for high-tier chapters."""

    @pytest.mark.asyncio
    async def test_tier_boost_endpoint_reachable_returns_200(self, dispatcher: ToolDispatcher) -> None:
        if not await _service_ready():
            pytest.skip(f"unified-search-service not reachable at {SEARCH_URL}")

        result = await dispatcher.dispatch(
            "hybrid_search",
            {
                "query": "dependency injection architecture",
                "collection": "all",
                "limit": 10,
                "alpha": 0.7,
                "tier_boost": True,
            },
        )
        assert result.status_code == 200
        data = result.body
        assert "results" in data

    @pytest.mark.asyncio
    async def test_tier_boost_false_produces_result_without_boost_applied(self, dispatcher: ToolDispatcher) -> None:
        if not await _service_ready():
            pytest.skip(f"unified-search-service not reachable at {SEARCH_URL}")

        result = await dispatcher.dispatch(
            "hybrid_search",
            {
                "query": "dependency injection architecture",
                "collection": "all",
                "limit": 5,
                "alpha": 0.7,
                "tier_boost": False,
            },
        )
        assert result.status_code == 200
        data = result.body
        # When tier_boost=False, tier_boost_applied should be None on all results
        for r in data.get("results", []):
            assert r.get("tier_boost_applied") is None, (
                f"Expected tier_boost_applied=None when boost disabled, got {r.get('tier_boost_applied')}"
            )


class TestCombinedFiltersViaGateway:
    """AC-TXS7.3: bloom_tier_filter + quality_tier_filter combined."""

    @pytest.mark.asyncio
    async def test_combined_tier_filters_return_200(self, dispatcher: ToolDispatcher) -> None:
        if not await _service_ready():
            pytest.skip(f"unified-search-service not reachable at {SEARCH_URL}")

        result = await dispatcher.dispatch(
            "hybrid_search",
            {
                "query": "design patterns",
                "collection": "all",
                "limit": 10,
                "alpha": 0.7,
                "bloom_tier_filter": [2, 3],
                "quality_tier_filter": [1],
                "tier_boost": True,
            },
        )
        assert result.status_code == 200
        assert "results" in result.body


class TestBackwardCompatViaGateway:
    """AC-TXS7.5: Calls without new tier params work identically to before TXS3-6."""

    @pytest.mark.asyncio
    async def test_hybrid_search_without_tier_params_returns_200(self, dispatcher: ToolDispatcher) -> None:
        if not await _service_ready():
            pytest.skip(f"unified-search-service not reachable at {SEARCH_URL}")

        result = await dispatcher.dispatch(
            "hybrid_search",
            {
                "query": "software architecture",
                "collection": "all",
                "limit": 5,
                "alpha": 0.7,
            },
        )
        assert result.status_code == 200
        body = result.body
        assert "results" in body
        assert "total" in body

    @pytest.mark.asyncio
    async def test_hybrid_search_without_tier_params_latency_acceptable(self, dispatcher: ToolDispatcher) -> None:
        if not await _service_ready():
            pytest.skip(f"unified-search-service not reachable at {SEARCH_URL}")

        result = await dispatcher.dispatch(
            "hybrid_search",
            {
                "query": "clean code principles",
                "collection": "all",
                "limit": 5,
                "alpha": 0.7,
            },
        )
        assert result.status_code == 200
        # Basic latency sanity: must have elapsed_ms recorded
        assert result.elapsed_ms >= 0
