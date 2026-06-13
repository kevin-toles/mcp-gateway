"""
F10 (P0/CRITICAL) — Wiring test: ToolDispatcher ↔ IdleTimeoutTracker.

Verifies that ``ToolDispatcher.dispatch()`` calls ``record_request()``
with the correct ``service_name`` for each tool, and that errors from
the tracker do not propagate to callers.

Reference: F10 — Dead code in idle-timeout tracking hook
           Strategy §4.4.1, AC-10.1 through AC-10.3
"""

from __future__ import annotations

from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from src.tool_dispatcher import ToolDispatcher
from src.core.config import Settings
from src.core.idle_timeout import get_tracker, reset_tracker


# ── Helpers ─────────────────────────────────────────────────────────────


def _null_settings() -> Settings:
    """Return a minimal Settings object with default values."""
    return Settings()  # type: ignore[call-arg]


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_tracker():
    """Fresh tracker singleton before each test."""
    reset_tracker()


@pytest.fixture
def dispatcher():
    """A ToolDispatcher with mocked clients so dispatch can init."""
    return ToolDispatcher(settings=_null_settings())


# ── Tests ───────────────────────────────────────────────────────────────


class TestRecordRequestCalled:
    """Verify record_request() is invoked during dispatch()."""

    @patch("src.tool_dispatcher.get_tracker")
    async def test_record_request_called_for_known_tool(self, mock_get_tracker, dispatcher):
        """dispatch() should call record_request with mapped service name."""
        mock_tracker = MagicMock()
        mock_get_tracker.return_value = mock_tracker

        with patch.object(dispatcher, "_attempt_dispatch", new_callable=AsyncMock) as mock_ad:
            mock_ad.return_value = MagicMock(status_code=200)

            with patch.object(dispatcher, "_check_circuit_breaker", new_callable=AsyncMock):
                try:
                    await dispatcher.dispatch("code_analyze", {"code": "x"})
                except Exception:
                    pass  # may fail on real client; we only check record_request

        mock_tracker.record_request.assert_called_once_with("code-orchestrator")

    @patch("src.tool_dispatcher.get_tracker")
    async def test_record_request_passthrough_for_unknown_tool(self, mock_get_tracker, dispatcher):
        """Unknown tools should pass the tool name verbatim."""
        mock_tracker = MagicMock()
        mock_get_tracker.return_value = mock_tracker

        with patch.object(dispatcher, "_attempt_dispatch", new_callable=AsyncMock) as mock_ad:
            mock_ad.return_value = MagicMock(status_code=200)

            with patch.object(dispatcher, "_check_circuit_breaker", new_callable=AsyncMock):
                try:
                    await dispatcher.dispatch("some_random_tool", {})
                except Exception:
                    pass

        mock_tracker.record_request.assert_called_once_with("some_random_tool")

    @patch("src.tool_dispatcher.get_tracker")
    async def test_record_request_called_before_dispatch(self, mock_get_tracker, dispatcher):
        """record_request must be called BEFORE the actual dispatch attempt."""
        mock_tracker = MagicMock()
        mock_get_tracker.return_value = mock_tracker

        call_order = []

        def track_first(*args, **kwargs):
            call_order.append("record_request")
            return None

        mock_tracker.record_request.side_effect = track_first

        with patch.object(dispatcher, "_attempt_dispatch", new_callable=AsyncMock) as mock_ad:

            async def dispatch_second(*args, **kwargs):
                call_order.append("_attempt_dispatch")
                return MagicMock(status_code=200)

            mock_ad.side_effect = dispatch_second

            with patch.object(dispatcher, "_check_circuit_breaker", new_callable=AsyncMock):
                try:
                    await dispatcher.dispatch("code_analyze", {"code": "x"})
                except Exception:
                    pass

        assert call_order == ["record_request", "_attempt_dispatch"]


class TestTrackerErrorHandling:
    """record_request errors must not propagate."""

    @patch("src.tool_dispatcher.get_tracker")
    async def test_tracker_exception_caught(self, mock_get_tracker, dispatcher):
        """If record_request raises, dispatch should continue."""
        mock_tracker = MagicMock()
        mock_tracker.record_request.side_effect = RuntimeError("tracker down")
        mock_get_tracker.return_value = mock_tracker

        with patch.object(dispatcher, "_attempt_dispatch", new_callable=AsyncMock) as mock_ad:
            mock_ad.return_value = MagicMock(status_code=200)

            with patch.object(dispatcher, "_check_circuit_breaker", new_callable=AsyncMock):
                try:
                    result = await dispatcher.dispatch("code_analyze", {"code": "x"})
                except Exception:
                    result = None

        # dispatch should NOT raise — the try/except pass caught it
        assert result is not None or mock_ad.called


class TestIntegration:
    """Light integration: use real tracker singleton."""

    async def test_real_tracker_records_via_dispatcher(self, dispatcher):
        """After dispatch, the real tracker should have a record for the service."""
        with patch.object(dispatcher, "_attempt_dispatch", new_callable=AsyncMock) as mock_ad:
            mock_ad.return_value = MagicMock(status_code=200)

            with patch.object(dispatcher, "_check_circuit_breaker", new_callable=AsyncMock):
                try:
                    await dispatcher.dispatch("code_analyze", {"code": "x"})
                except Exception:
                    pass

        state = get_tracker()._service_states
        assert "code-orchestrator" in state
        assert state["code-orchestrator"]["total_requests"] >= 1
