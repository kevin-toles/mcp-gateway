"""ETDI-1.0.1071/.1072 (v0.48, AC-287; S3/S4 bug fix): the gateway never sends a retired collection name.

Under ETDI every level of book text lives in the `chapters` collection, reached by `granularity` (design §5.1.8;
§6 "USS-RS hybrid fans out to `chapters` (granularity-filtered)"); `chunks`, `passages` and `code_textbook_bridge` are
retired. Broad search (hybrid_search, semantic_search) asks for `chapter`; targeted refining (search_in,
knowledge_refine) asks for `passage`. Requests are captured as USS-RS /v1/search/hybrid receives them.
"""
from __future__ import annotations

import json

import httpx
import pytest

from src.core.config import Settings
from src.tool_dispatcher import ToolDispatcher

RETIRED = {"chunks", "passages", "code_textbook_bridge"}


def _capturing_dispatcher() -> tuple[ToolDispatcher, list[dict]]:
    seen: list[dict] = []

    async def transport(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"results": [], "total": 0, "latency_ms": 1.0})

    d = ToolDispatcher(Settings())
    d._client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    return d, seen


@pytest.mark.asyncio
async def test_search_in_textbooks_asks_for_chapters_at_passage_level() -> None:
    from src.tools.search_in import create_handler
    d, seen = _capturing_dispatcher()
    await create_handler(d)(query="Kafka consumer group rebalancing", source="textbooks")
    assert (seen[0]["collection"], seen[0].get("granularity")) == ("chapters", "passage")


@pytest.mark.asyncio
async def test_hybrid_search_textbooks_asks_for_chapters_at_chapter_level() -> None:
    from src.security.output_sanitizer import OutputSanitizer
    from src.tools.hybrid_search import create_handler
    d, seen = _capturing_dispatcher()
    await create_handler(d, OutputSanitizer())(query="Kafka consumer group rebalancing", source="textbooks")
    assert (seen[0]["collection"], seen[0].get("granularity")) == ("chapters", "chapter")


@pytest.mark.asyncio
async def test_knowledge_refine_on_book_text_asks_for_passages() -> None:
    from src.security.output_sanitizer import OutputSanitizer
    from src.tools.knowledge_refine import create_handler
    d, seen = _capturing_dispatcher()
    await create_handler(d, OutputSanitizer())(query="Kafka consumer group rebalancing", collection="textbooks")
    assert (seen[0]["collection"], seen[0].get("granularity")) == ("chapters", "passage")


def test_no_resolver_maps_to_a_retired_collection() -> None:
    from src.tools import _resolvers
    targets = set(_resolvers._SEARCH_COLLECTION_MAP.values()) | set(_resolvers._SHELF_COLLECTION_MAP.values())
    assert not targets & RETIRED, targets & RETIRED


@pytest.mark.asyncio
async def test_semantic_search_textbooks_asks_for_chapters_at_chapter_level() -> None:
    from src.security.output_sanitizer import OutputSanitizer
    from src.tools.semantic_search import create_handler
    d, seen = _capturing_dispatcher()
    await create_handler(d, OutputSanitizer())(query="Kafka consumer group rebalancing", source="textbooks")
    assert (seen[0]["collection"], seen[0].get("granularity")) == ("chapters", "chapter")
