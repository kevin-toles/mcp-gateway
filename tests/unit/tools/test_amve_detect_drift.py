"""G2.7 / G2.9 (RED) — amve_detect_drift MCP tool.

TDD coverage for AC-2.5, 2.6, 2.7:
  - Registered in _INPUT_MODELS with AMVEDetectDriftInput schema
  - SHA-pair mode: XRANGE lookup → POST /v1/architecture/drift
  - SHA not-found returns structured error
  - Raw-dict passthrough mode: direct POST /v1/architecture/drift (no Redis)

Tasks: G2.7 (RED), G2.8 (GREEN), G2.9 (RED), G2.10 (GREEN)
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.core.config import Settings
from src.tool_dispatcher import ToolDispatcher

_DUMMY_SHA_A = "a" * 64
_DUMMY_SHA_B = "b" * 64
_SNAPSHOT_A = {"services": ["auth", "payments"], "edges": [], "version": 1}
_SNAPSHOT_B = {"services": ["auth", "payments", "billing"], "edges": [], "version": 2}
_DRIFT_RESPONSE = {
    "has_drift": True,
    "added_count": 1,
    "removed_count": 0,
    "modified_count": 0,
    "items": [{"record_id": "billing", "drift_type": "added", "significance": "high"}],
}


@pytest.fixture
def dispatcher():
    settings = Settings()
    return ToolDispatcher(settings)


# ---------------------------------------------------------------------------
# AC-2.5 — Tool registration
# ---------------------------------------------------------------------------


class TestAmveDetectDriftRegistration:
    """AC-2.5: amve_detect_drift is registered in the tool registry."""

    def test_tool_in_input_models(self):
        from src.tool_registry import _INPUT_MODELS

        assert "amve_detect_drift" in _INPUT_MODELS, (
            "amve_detect_drift is missing from _INPUT_MODELS"
        )

    def test_input_model_is_importable(self):
        from src.models.schemas import AMVEDetectDriftInput  # noqa: F401

    def test_input_model_has_snapshot_sha_fields(self):
        from src.models.schemas import AMVEDetectDriftInput

        assert "snapshot_a_sha" in AMVEDetectDriftInput.model_fields
        assert "snapshot_b_sha" in AMVEDetectDriftInput.model_fields

    def test_input_model_has_snapshot_dict_fields(self):
        from src.models.schemas import AMVEDetectDriftInput

        assert "snapshot_a" in AMVEDetectDriftInput.model_fields
        assert "snapshot_b" in AMVEDetectDriftInput.model_fields


# ---------------------------------------------------------------------------
# AC-2.5 — Dispatch route (to AMVE /v1/architecture/drift)
# ---------------------------------------------------------------------------


class TestAmveDetectDriftDispatchRoute:
    """amve_detect_drift has a dispatch route pointing to AMVE /v1/architecture/drift."""

    def test_route_registered(self, dispatcher):
        route = dispatcher.get_route("amve_detect_drift")
        assert route is not None, "amve_detect_drift has no dispatch route"

    def test_route_path(self, dispatcher):
        route = dispatcher.get_route("amve_detect_drift")
        assert route.path == "/v1/architecture/drift"

    def test_route_base_url_is_amve(self, dispatcher):
        route = dispatcher.get_route("amve_detect_drift")
        assert route.base_url == "http://localhost:8088"


# ---------------------------------------------------------------------------
# G2.7 (RED) — AC-2.5: SHA-pair mode
# ---------------------------------------------------------------------------


class TestDetectDriftSHAPairMode:
    """AC-2.5: SHA-pair mode resolves snapshots from Redis stream then calls AMVE drift."""

    def test_handler_factory_is_importable(self):
        """Handler module can be imported."""
        from src.tools import amve_detect_drift  # noqa: F401

    def test_create_handler_returns_callable(self, dispatcher):
        """create_handler() returns a callable handler."""
        from src.security.output_sanitizer import OutputSanitizer
        from src.tools import amve_detect_drift

        sanitizer = OutputSanitizer()
        handler = amve_detect_drift.create_handler(dispatcher, sanitizer)
        assert callable(handler)

    @pytest.mark.asyncio
    async def test_sha_pair_mode_performs_xrange_lookup(self, dispatcher):
        """AC-2.5: SHA-pair mode queries XRANGE amve:findings:anonymous stream."""
        from src.security.output_sanitizer import OutputSanitizer
        from src.tools import amve_detect_drift

        captured_xrange_calls: list = []

        # Mock Redis client: returns stream entries containing snapshot data
        mock_redis = AsyncMock()

        async def fake_xrange(stream_key, start, end):
            captured_xrange_calls.append(stream_key)
            return [
                (b"1-0", {
                    "snapshot_sha": _DUMMY_SHA_A,
                    "snapshot": json.dumps(_SNAPSHOT_A),
                }),
                (b"2-0", {
                    "snapshot_sha": _DUMMY_SHA_B,
                    "snapshot": json.dumps(_SNAPSHOT_B),
                }),
            ]

        mock_redis.xrange = fake_xrange

        # Mock httpx for the POST /v1/architecture/drift call
        async def mock_http(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_DRIFT_RESPONSE)

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_http))

        with patch("src.tools.amve_detect_drift.redis_async") as mock_redis_mod:
            mock_redis_mod.from_url.return_value = mock_redis

            sanitizer = OutputSanitizer()
            handler = amve_detect_drift.create_handler(dispatcher, sanitizer)

            await handler(
                snapshot_a_sha=_DUMMY_SHA_A,
                snapshot_b_sha=_DUMMY_SHA_B,
            )

        assert len(captured_xrange_calls) > 0, "XRANGE must be called for SHA-pair lookup"

    @pytest.mark.asyncio
    async def test_sha_pair_mode_calls_amve_drift_endpoint(self, dispatcher):
        """AC-2.5: After resolving snapshots, makes POST to /v1/architecture/drift."""
        from src.security.output_sanitizer import OutputSanitizer
        from src.tools import amve_detect_drift

        captured_http: dict = {}

        mock_redis = AsyncMock()
        mock_redis.xrange = AsyncMock(
            return_value=[
                (b"1-0", {
                    "snapshot_sha": _DUMMY_SHA_A,
                    "snapshot": json.dumps(_SNAPSHOT_A),
                }),
                (b"2-0", {
                    "snapshot_sha": _DUMMY_SHA_B,
                    "snapshot": json.dumps(_SNAPSHOT_B),
                }),
            ]
        )

        async def mock_http(request: httpx.Request) -> httpx.Response:
            captured_http["url"] = str(request.url)
            captured_http["method"] = request.method
            return httpx.Response(200, json=_DRIFT_RESPONSE)

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_http))

        with patch("src.tools.amve_detect_drift.redis_async") as mock_redis_mod:
            mock_redis_mod.from_url.return_value = mock_redis

            sanitizer = OutputSanitizer()
            handler = amve_detect_drift.create_handler(dispatcher, sanitizer)
            await handler(
                snapshot_a_sha=_DUMMY_SHA_A,
                snapshot_b_sha=_DUMMY_SHA_B,
            )

        assert captured_http.get("url") == "http://localhost:8088/v1/architecture/drift"
        assert captured_http.get("method") == "POST"

    @pytest.mark.asyncio
    async def test_sha_pair_mode_returns_drift_response(self, dispatcher):
        """AC-2.5: SHA-pair mode returns the drift response from AMVE."""
        from src.security.output_sanitizer import OutputSanitizer
        from src.tools import amve_detect_drift

        mock_redis = AsyncMock()
        mock_redis.xrange = AsyncMock(
            return_value=[
                (b"1-0", {
                    "snapshot_sha": _DUMMY_SHA_A,
                    "snapshot": json.dumps(_SNAPSHOT_A),
                }),
                (b"2-0", {
                    "snapshot_sha": _DUMMY_SHA_B,
                    "snapshot": json.dumps(_SNAPSHOT_B),
                }),
            ]
        )

        async def mock_http(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_DRIFT_RESPONSE)

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_http))

        with patch("src.tools.amve_detect_drift.redis_async") as mock_redis_mod:
            mock_redis_mod.from_url.return_value = mock_redis

            sanitizer = OutputSanitizer()
            handler = amve_detect_drift.create_handler(dispatcher, sanitizer)
            result = await handler(
                snapshot_a_sha=_DUMMY_SHA_A,
                snapshot_b_sha=_DUMMY_SHA_B,
            )

        # Result should contain drift detection fields
        assert "has_drift" in result or "added_count" in result or result.get("success")

    @pytest.mark.asyncio
    async def test_sha_pair_returns_structured_error_when_sha_not_found(self, dispatcher):
        """AC-2.7: Returns structured error dict (not exception) when SHA not in stream."""
        from src.security.output_sanitizer import OutputSanitizer
        from src.tools import amve_detect_drift

        mock_redis = AsyncMock()
        # Return empty stream — SHA won't be found
        mock_redis.xrange = AsyncMock(return_value=[])

        with patch("src.tools.amve_detect_drift.redis_async") as mock_redis_mod:
            mock_redis_mod.from_url.return_value = mock_redis

            sanitizer = OutputSanitizer()
            handler = amve_detect_drift.create_handler(dispatcher, sanitizer)

            # Must NOT raise; must return a structured error dict
            result = await handler(
                snapshot_a_sha=_DUMMY_SHA_A,
                snapshot_b_sha=_DUMMY_SHA_B,
            )

        assert isinstance(result, dict), "Must return a dict, not raise an exception"
        # Must have an error indicator
        has_error = (
            "error" in result
            or result.get("success") is False
            or "not_found" in str(result).lower()
            or "not found" in str(result).lower()
        )
        assert has_error, f"Must contain error indicator, got: {result}"


# ---------------------------------------------------------------------------
# G2.9 (RED) — AC-2.6: Raw-dict passthrough mode
# ---------------------------------------------------------------------------


class TestDetectDriftPassthroughMode:
    """AC-2.6: When snapshot_a and snapshot_b dicts provided, calls AMVE directly (no Redis)."""

    @pytest.mark.asyncio
    async def test_passthrough_does_not_call_redis(self, dispatcher):
        """AC-2.6: Passthrough mode does NOT perform Redis XRANGE lookup."""
        from src.security.output_sanitizer import OutputSanitizer
        from src.tools import amve_detect_drift

        redis_xrange_called = False

        mock_redis = AsyncMock()

        async def fake_xrange(*args, **kwargs):
            nonlocal redis_xrange_called
            redis_xrange_called = True
            return []

        mock_redis.xrange = fake_xrange

        async def mock_http(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_DRIFT_RESPONSE)

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_http))

        with patch("src.tools.amve_detect_drift.redis_async") as mock_redis_mod:
            mock_redis_mod.from_url.return_value = mock_redis

            sanitizer = OutputSanitizer()
            handler = amve_detect_drift.create_handler(dispatcher, sanitizer)
            await handler(
                snapshot_a=_SNAPSHOT_A,
                snapshot_b=_SNAPSHOT_B,
            )

        assert not redis_xrange_called, (
            "Passthrough mode must NOT call Redis XRANGE"
        )

    @pytest.mark.asyncio
    async def test_passthrough_calls_amve_drift_directly(self, dispatcher):
        """AC-2.6: Passthrough mode sends snapshot dicts directly to POST /v1/architecture/drift."""
        from src.security.output_sanitizer import OutputSanitizer
        from src.tools import amve_detect_drift

        captured: dict = {}

        async def mock_http(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            captured["url"] = str(request.url)
            return httpx.Response(200, json=_DRIFT_RESPONSE)

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_http))

        with patch("src.tools.amve_detect_drift.redis_async"):
            sanitizer = OutputSanitizer()
            handler = amve_detect_drift.create_handler(dispatcher, sanitizer)
            await handler(
                snapshot_a=_SNAPSHOT_A,
                snapshot_b=_SNAPSHOT_B,
            )

        assert captured.get("url") == "http://localhost:8088/v1/architecture/drift"
        body = captured.get("body", {})
        assert "snapshot_a" in body or "snapshot_b" in body, (
            "POST body must contain snapshot_a and snapshot_b for passthrough mode"
        )

    @pytest.mark.asyncio
    async def test_passthrough_returns_drift_response(self, dispatcher):
        """AC-2.6: Passthrough mode returns the same drift response schema as SHA-pair mode."""
        from src.security.output_sanitizer import OutputSanitizer
        from src.tools import amve_detect_drift

        async def mock_http(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_DRIFT_RESPONSE)

        dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(mock_http))

        with patch("src.tools.amve_detect_drift.redis_async"):
            sanitizer = OutputSanitizer()
            handler = amve_detect_drift.create_handler(dispatcher, sanitizer)
            result = await handler(
                snapshot_a=_SNAPSHOT_A,
                snapshot_b=_SNAPSHOT_B,
            )

        assert isinstance(result, dict)
        # Should contain drift detection fields (or success wrapper)
        assert "has_drift" in result or "added_count" in result or result.get("success")


# =============================================================================
# TestDetectDriftTenantStreamKey — G3.9/G3.11 RED
# =============================================================================


class TestDetectDriftTenantStreamKey:
    """AC-3.6/3.7/3.8: amve_detect_drift uses tenant-scoped stream key."""

    @pytest.fixture
    def identity_dispatcher(self):
        """ToolDispatcher with IDENTITY_PROPAGATION=true."""
        from src.core.config import Settings
        from src.tool_dispatcher import ToolDispatcher

        settings = Settings(IDENTITY_PROPAGATION=True)
        return ToolDispatcher(settings)

    @pytest.mark.asyncio
    async def test_sha_pair_queries_tenant_scoped_stream_when_flag_on(
        self, identity_dispatcher
    ):
        """AC-3.6: handler queries amve:findings:{tenant_id} when IDENTITY_PROPAGATION=true."""
        from src.security.output_sanitizer import OutputSanitizer
        from src.tools import amve_detect_drift

        queried_streams: list[str] = []

        async def fake_xrange(stream, start, end):
            queried_streams.append(stream)
            if stream == "amve:findings:tenant-foo":
                return [("1-0", {"snapshot_sha": "sha_a", "snapshot": "{}"})]
            return []

        mock_redis = AsyncMock()
        mock_redis.xrange.side_effect = fake_xrange
        mock_redis.aclose = AsyncMock()

        with patch(
            "src.tools.amve_detect_drift.redis_async"
        ) as mock_redis_mod:
            mock_redis_mod.from_url.return_value = mock_redis
            sanitizer = OutputSanitizer()
            handler = amve_detect_drift.create_handler(identity_dispatcher, sanitizer)
            await handler(
                snapshot_a_sha="sha_a",
                snapshot_b_sha="sha_b",
                tenant_id="tenant-foo",
            )

        assert any("tenant-foo" in s for s in queried_streams), (
            f"Expected tenant-scoped stream query; got: {queried_streams}"
        )

    @pytest.mark.asyncio
    async def test_sha_pair_uses_anonymous_stream_when_flag_off(
        self, dispatcher
    ):
        """AC-3.6: handler queries amve:findings:anonymous when IDENTITY_PROPAGATION=false."""
        from src.security.output_sanitizer import OutputSanitizer
        from src.tools import amve_detect_drift

        queried_streams: list[str] = []

        async def fake_xrange(stream, start, end):
            queried_streams.append(stream)
            return []

        mock_redis = AsyncMock()
        mock_redis.xrange.side_effect = fake_xrange
        mock_redis.aclose = AsyncMock()

        with patch(
            "src.tools.amve_detect_drift.redis_async"
        ) as mock_redis_mod:
            mock_redis_mod.from_url.return_value = mock_redis
            sanitizer = OutputSanitizer()
            handler = amve_detect_drift.create_handler(dispatcher, sanitizer)
            await handler(
                snapshot_a_sha="sha_a",
                snapshot_b_sha="sha_b",
            )

        assert all("anonymous" in s for s in queried_streams), (
            f"Expected amve:findings:anonymous; got: {queried_streams}"
        )

    @pytest.mark.asyncio
    async def test_cross_tenant_isolation(self, identity_dispatcher):
        """AC-3.7: SHA written under tenant-A stream is NOT visible under tenant-B."""
        from src.security.output_sanitizer import OutputSanitizer
        from src.tools import amve_detect_drift

        # tenant-A stream has sha_a; tenant-B stream is empty
        streams: dict[str, list] = {
            "amve:findings:tenant-A": [("1-0", {"snapshot_sha": "sha_a", "snapshot": "{}"})],
            "amve:findings:tenant-B": [],
        }

        async def fake_xrange(stream, start, end):
            return streams.get(stream, [])

        mock_redis = AsyncMock()
        mock_redis.xrange.side_effect = fake_xrange
        mock_redis.aclose = AsyncMock()

        with patch(
            "src.tools.amve_detect_drift.redis_async"
        ) as mock_redis_mod:
            mock_redis_mod.from_url.return_value = mock_redis
            sanitizer = OutputSanitizer()
            handler = amve_detect_drift.create_handler(identity_dispatcher, sanitizer)
            # Query as tenant-B — sha_a lives only in tenant-A's stream
            result = await handler(
                snapshot_a_sha="sha_a",
                snapshot_b_sha="sha_b",
                tenant_id="tenant-B",
            )

        # sha_a was not found in tenant-B's stream → snapshot_not_found
        assert result.get("error") == "snapshot_not_found"
        assert result.get("sha") == "sha_a"
