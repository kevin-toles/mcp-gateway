"""Tests for audit_codebase_scan tool dedupe and route settings."""

from __future__ import annotations

import asyncio

import pytest

TEST_SOURCE_NODE = "/Users/example/amplitude/Amplitude-Node"
TEST_SOURCE_PYTHON = "/Users/example/amplitude/Amplitude-Python"

from src.core.config import Settings
from src.tool_dispatcher import ToolDispatcher
from src.tools import audit_codebase_scan as tool_module


class _FakeDispatchResult:
    def __init__(self, body: dict) -> None:
        self.body = body


class _FakeDispatcher:
    def __init__(self) -> None:
        self.calls = 0

    async def dispatch(self, tool_name: str, payload: dict) -> _FakeDispatchResult:
        self.calls += 1
        await asyncio.sleep(0.05)
        return _FakeDispatchResult(
            {
                "source_path": payload["source_path"],
                "findings": [],
                "stats": {
                    "files_scanned": 1,
                    "files_skipped": 0,
                    "total_findings": 0,
                    "scan_time_ms": 42,
                    "files_with_findings": 0,
                    "findings_by_priority": {},
                },
            }
        )


class _FakeSanitizer:
    def sanitize(self, body: dict) -> dict:
        return body


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    tool_module._INFLIGHT.clear()
    tool_module._RECENT_RESULTS.clear()


@pytest.mark.asyncio
async def test_single_flight_dedupes_concurrent_identical_requests() -> None:
    dispatcher = _FakeDispatcher()
    sanitizer = _FakeSanitizer()
    handler = tool_module.create_handler(dispatcher, sanitizer)

    first, second = await asyncio.gather(
        handler(source_path=TEST_SOURCE_NODE),
        handler(source_path=TEST_SOURCE_NODE),
    )

    assert dispatcher.calls == 1
    assert first["source_path"] == TEST_SOURCE_NODE
    assert second["source_path"] == TEST_SOURCE_NODE


@pytest.mark.asyncio
async def test_recent_result_cache_serves_immediate_retries() -> None:
    dispatcher = _FakeDispatcher()
    sanitizer = _FakeSanitizer()
    handler = tool_module.create_handler(dispatcher, sanitizer)

    first = await handler(source_path=TEST_SOURCE_PYTHON)
    second = await handler(source_path=TEST_SOURCE_PYTHON)

    assert dispatcher.calls == 1
    assert first == second


def test_audit_codebase_scan_route_has_bounded_timeout() -> None:
    dispatcher = ToolDispatcher(Settings())
    route = dispatcher.get_route("audit_codebase_scan")
    assert route is not None
    assert route.timeout == pytest.approx(600.0)
