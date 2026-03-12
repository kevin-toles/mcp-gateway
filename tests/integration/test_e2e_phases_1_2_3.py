"""Phase 4 E2E integration tests — SHA-drift cycle & tenant isolation (mcp-gateway).

AC-4.2: E2E extract→SHA→drift cycle — snapshot SHA resolved from Redis XRANGE,
        drift endpoint dispatched with resolved snapshots.
AC-4.3: Tenant isolation — SHA written under tenant-A NOT retrievable from tenant-B.

These tests run entirely in-process:
  - fakeredis backend (no live Redis required)
  - mock ToolDispatcher (no live AMVE backend required)

Run with:
    INTEGRATION=1 pytest tests/integration/test_e2e_phases_1_2_3.py -m integration -v
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis.aioredis as fakeredis_async
import pytest
import pytest_asyncio

pytestmark = pytest.mark.integration


# =============================================================================
# Helpers
# =============================================================================


def _sha(snapshot: dict) -> str:
    """Compute the canonical SHA-256 for a snapshot dict."""
    canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _xadd_fields(snapshot: dict, tool_name: str = "amve_extract_architecture") -> dict:
    """Build the Redis XADD field dict as written by AMVE's EventPublisher."""
    return {
        "tool_name": tool_name,
        "timestamp": "2025-01-01T00:00:00+00:00",
        "request": json.dumps({}),
        "response": json.dumps({"record_count": len(snapshot)}),
        "latency_ms": "10.0",
        "snapshot_sha": _sha(snapshot),
        "snapshot": json.dumps(snapshot, sort_keys=True, separators=(",", ":")),
    }


# =============================================================================
# Fixtures
# =============================================================================


@pytest_asyncio.fixture
async def fake_redis():
    """In-process Redis backed by fakeredis (no live Redis required)."""
    client = fakeredis_async.FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


@pytest.fixture
def drift_response():
    """Canonical fake drift response matching AMVE's /v1/architecture/drift schema."""
    return {
        "has_drift": True,
        "added_count": 1,
        "removed_count": 0,
        "modified_count": 0,
        "added": [{"id": "svc-new"}],
        "removed": [],
        "modified": [],
    }


def _make_dispatcher(
    *,
    identity_propagation: bool = False,
    redis_url: str = "redis://localhost:6379",
    drift_body: dict | None = None,
) -> MagicMock:
    """Build a mock ToolDispatcher for use in unit-style integration tests."""
    from src.tool_dispatcher import DispatchResult

    settings = MagicMock()
    settings.IDENTITY_PROPAGATION = identity_propagation
    settings.REDIS_URL = redis_url

    result = DispatchResult(
        status_code=200,
        body=drift_body or {"has_drift": False},
        headers={},
        elapsed_ms=5.0,
    )

    dispatcher = MagicMock()
    dispatcher._settings = settings
    dispatcher.dispatch = AsyncMock(return_value=result)
    return dispatcher


# =============================================================================
# TestExtractSHADriftCycleE2E — G4.3 / AC-4.2
# =============================================================================


class TestExtractSHADriftCycleE2E:
    """AC-4.2: SHA-pair mode end-to-end — snapshots pre-loaded in fakeredis,
    drift handler resolves them with XRANGE, dispatches drift call."""

    @pytest.mark.asyncio
    async def test_drift_resolves_both_snapshots_from_stream(
        self, fake_redis: fakeredis_async.FakeRedis, drift_response: dict
    ) -> None:
        """Handler reads snap_a and snap_b from Redis stream, calls dispatcher.dispatch."""
        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.amve_detect_drift import create_handler

        snap_a = {"nodes": [{"id": "svc-A"}], "edges": []}
        snap_b = {"nodes": [{"id": "svc-A"}, {"id": "svc-B"}], "edges": [{"from": "svc-A", "to": "svc-B"}]}

        sha_a = _sha(snap_a)
        sha_b = _sha(snap_b)

        # Pre-populate fakeredis with the stream entries AMVE would write
        await fake_redis.xadd("amve:findings:anonymous", _xadd_fields(snap_a))
        await fake_redis.xadd("amve:findings:anonymous", _xadd_fields(snap_b))

        dispatcher = _make_dispatcher(drift_body=drift_response)
        sanitizer = OutputSanitizer()
        handler = create_handler(dispatcher, sanitizer)

        with patch(
            "src.tools.amve_detect_drift.redis_async.from_url",
            return_value=fake_redis,
        ):
            result = await handler(snapshot_a_sha=sha_a, snapshot_b_sha=sha_b)

        assert "has_drift" in result, f"Unexpected result: {result}"
        # Verify snapshots were resolved and drift was dispatched
        dispatcher.dispatch.assert_awaited_once()
        call_args = dispatcher.dispatch.call_args
        assert call_args.args[0] == "amve_detect_drift"
        dispatched_payload = call_args.args[1]
        assert dispatched_payload["snapshot_a"] == snap_a
        assert dispatched_payload["snapshot_b"] == snap_b

    @pytest.mark.asyncio
    async def test_drift_returns_error_when_sha_not_found(
        self, fake_redis: fakeredis_async.FakeRedis
    ) -> None:
        """When the SHA is absent from the stream, handler returns snapshot_not_found error."""
        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.amve_detect_drift import create_handler

        ghost_sha = "a" * 64  # SHA that was never written to the stream

        dispatcher = _make_dispatcher()
        sanitizer = OutputSanitizer()
        handler = create_handler(dispatcher, sanitizer)

        with patch(
            "src.tools.amve_detect_drift.redis_async.from_url",
            return_value=fake_redis,
        ):
            result = await handler(snapshot_a_sha=ghost_sha, snapshot_b_sha="b" * 64)

        assert result.get("error") == "snapshot_not_found"
        assert result.get("sha") == ghost_sha

    @pytest.mark.asyncio
    async def test_identical_snapshots_yield_same_sha(self) -> None:
        """Content-addressing: two equal dicts produce the same SHA — deterministic key."""
        snap = {"nodes": [{"id": "X"}, {"id": "Y"}], "edges": [{"from": "X", "to": "Y"}]}
        assert _sha(snap) == _sha(snap)
        # Different dict object, same content
        snap2 = dict(snap)
        assert _sha(snap) == _sha(snap2)


# =============================================================================
# TestTenantIsolationE2E — G4.5 / AC-4.3
# =============================================================================


class TestTenantIsolationE2E:
    """AC-4.3: tenant-A snapshot SHA is NOT retrievable under tenant-B's stream."""

    @pytest.mark.asyncio
    async def test_sha_written_to_tenant_a_not_found_under_tenant_b(
        self, fake_redis: fakeredis_async.FakeRedis
    ) -> None:
        """XRANGE on tenant-B's stream finds no entry after writing SHA to tenant-A."""
        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.amve_detect_drift import create_handler

        snap_a = {"nodes": [{"id": "svc-Alpha"}], "edges": []}
        sha_a = _sha(snap_a)

        # Pre-populate tenant-A's stream only
        await fake_redis.xadd("amve:findings:tenant-A", _xadd_fields(snap_a))

        # Dispatcher with IDENTITY_PROPAGATION enabled
        dispatcher = _make_dispatcher(identity_propagation=True)
        sanitizer = OutputSanitizer()
        handler = create_handler(dispatcher, sanitizer)

        # Call drift handler as tenant-B — should NOT find sha_a
        with patch(
            "src.tools.amve_detect_drift.redis_async.from_url",
            return_value=fake_redis,
        ):
            result = await handler(
                snapshot_a_sha=sha_a,
                snapshot_b_sha="b" * 64,
                tenant_id="tenant-B",
            )

        # Must return not-found when identity propagation is enabled
        assert result.get("error") == "snapshot_not_found", (
            f"Expected snapshot_not_found for tenant-B query, got: {result}"
        )
        assert result.get("sha") == sha_a

    @pytest.mark.asyncio
    async def test_sha_written_to_tenant_a_found_under_tenant_a(
        self, fake_redis: fakeredis_async.FakeRedis, drift_response: dict
    ) -> None:
        """Positive path: XRANGE on tenant-A correctly resolves both SHAs."""
        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.amve_detect_drift import create_handler

        snap_a = {"nodes": [{"id": "svc-Alpha"}], "edges": []}
        snap_b = {"nodes": [{"id": "svc-Alpha"}, {"id": "svc-Beta"}], "edges": []}
        sha_a = _sha(snap_a)
        sha_b = _sha(snap_b)

        # Pre-populate tenant-A's stream with both snapshots
        await fake_redis.xadd("amve:findings:tenant-A", _xadd_fields(snap_a))
        await fake_redis.xadd("amve:findings:tenant-A", _xadd_fields(snap_b))

        dispatcher = _make_dispatcher(identity_propagation=True, drift_body=drift_response)
        sanitizer = OutputSanitizer()
        handler = create_handler(dispatcher, sanitizer)

        with patch(
            "src.tools.amve_detect_drift.redis_async.from_url",
            return_value=fake_redis,
        ):
            result = await handler(
                snapshot_a_sha=sha_a,
                snapshot_b_sha=sha_b,
                tenant_id="tenant-A",
            )

        assert "has_drift" in result, f"Expected drift result for tenant-A, got: {result}"
        dispatcher.dispatch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cross_tenant_streams_are_independent(
        self, fake_redis: fakeredis_async.FakeRedis, drift_response: dict
    ) -> None:
        """tenant-A and tenant-B can each hold their own snapshots independently."""
        from src.security.output_sanitizer import OutputSanitizer
        from src.tools.amve_detect_drift import create_handler

        snap_a1 = {"nodes": [{"id": "alpha-svc"}], "edges": []}
        snap_b1 = {"nodes": [{"id": "beta-svc"}], "edges": []}

        sha_a1 = _sha(snap_a1)
        sha_b1 = _sha(snap_b1)

        await fake_redis.xadd("amve:findings:tenant-A", _xadd_fields(snap_a1))
        await fake_redis.xadd("amve:findings:tenant-B", _xadd_fields(snap_b1))

        sanitizer = OutputSanitizer()

        # tenant-A querying its own SHA succeeds
        disp_a = _make_dispatcher(identity_propagation=True, drift_body=drift_response)
        handler_a = create_handler(disp_a, sanitizer)
        snap_a2 = {"nodes": [{"id": "alpha-svc"}, {"id": "alpha-new"}], "edges": []}
        sha_a2 = _sha(snap_a2)
        await fake_redis.xadd("amve:findings:tenant-A", _xadd_fields(snap_a2))

        with patch("src.tools.amve_detect_drift.redis_async.from_url", return_value=fake_redis):
            result_a = await handler_a(
                snapshot_a_sha=sha_a1, snapshot_b_sha=sha_a2, tenant_id="tenant-A"
            )
        assert "has_drift" in result_a

        # tenant-B querying tenant-A's SHA returns not-found
        disp_b = _make_dispatcher(identity_propagation=True)
        handler_b = create_handler(disp_b, sanitizer)
        with patch("src.tools.amve_detect_drift.redis_async.from_url", return_value=fake_redis):
            result_b = await handler_b(
                snapshot_a_sha=sha_a1, snapshot_b_sha=sha_b1, tenant_id="tenant-B"
            )
        assert result_b.get("error") == "snapshot_not_found"
