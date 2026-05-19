#!/usr/bin/env python3
"""
Health-Aware Proxy Test Suite
==============================

TDD tests for the health-aware proxy that auto-restarts dead services.

Tests cover:
- Service health detection
- Auto-restart on demand
- Retry logic with exponential backoff
- Idle timeout integration
- Graceful degradation when proxy fails
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone, timedelta

import httpx

from src.middleware.health_proxy import (
    HealthAwareProxy,
    ServiceRestartError,
    get_proxy,
    reset_proxy,
)
from src.core.idle_timeout import IdleTimeoutTracker, get_tracker, reset_tracker


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_http_client():
    """Create a mock httpx.AsyncClient."""
    client = AsyncMock(spec=httpx.AsyncClient)
    return client


@pytest.fixture
def proxy(mock_http_client):
    """Create a HealthAwareProxy with mock client."""
    proxy = HealthAwareProxy(http_client=mock_http_client)
    # Reset tracker for clean state
    reset_tracker()
    return proxy


@pytest.fixture
def tracker():
    """Create a fresh IdleTimeoutTracker."""
    reset_tracker()
    return IdleTimeoutTracker(default_timeout_seconds=600)


# =============================================================================
# RED Phase Tests - Health Detection
# =============================================================================

class TestHealthDetection:
    """Tests for service health detection."""
    
    @pytest.mark.asyncio
    async def test_is_service_dead_returns_true_when_unreachable(self, proxy):
        """Service should be detected as dead when health check fails."""
        with patch('httpx.AsyncClient.get', side_effect=Exception("Connection refused")):
            is_dead = await proxy._is_service_dead("semantic_search")
            assert is_dead is True
    
    @pytest.mark.asyncio
    async def test_is_service_dead_returns_false_when_healthy(self, proxy):
        """Service should be detected as alive when health check succeeds."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        with patch('httpx.AsyncClient.get', return_value=mock_response):
            is_dead = await proxy._is_service_dead("semantic_search")
            assert is_dead is False
    
    @pytest.mark.asyncio
    async def test_is_service_dead_returns_true_on_503(self, proxy):
        """Service should be detected as dead when returning 503."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        
        with patch('httpx.AsyncClient.get', return_value=mock_response):
            is_dead = await proxy._is_service_dead("semantic_search")
            assert is_dead is True
    
    @pytest.mark.asyncio
    async def test_is_service_dead_returns_false_for_unknown_service(self, proxy):
        """Unknown service should not be detected as dead."""
        is_dead = await proxy._is_service_dead("unknown-service")
        assert is_dead is False


# =============================================================================
# RED Phase Tests - Auto-Restart
# =============================================================================

class TestAutoRestart:
    """Tests for service auto-restart functionality."""
    
    @pytest.mark.asyncio
    async def test_restart_service_starts_process(self, proxy):
        """Restart should start the service process."""
        mock_process = AsyncMock()
        mock_process.returncode = None  # Process is running
        
        with patch('asyncio.create_subprocess_shell', return_value=mock_process) as mock_shell:
            with patch.object(proxy, '_wait_for_service', new_callable=AsyncMock):
                await proxy._restart_service("semantic_search")
                
                # Verify command was executed
                mock_shell.assert_called_once()
                assert "unified-search-service" in mock_shell.call_args[0][0]
    
    @pytest.mark.asyncio
    async def test_restart_service_raises_on_immediate_exit(self, proxy):
        """Restart should fail if process exits immediately."""
        mock_process = AsyncMock()
        mock_process.returncode = 1  # Process failed
        
        with patch('asyncio.create_subprocess_shell', return_value=mock_process):
            with pytest.raises(ServiceRestartError):
                await proxy._restart_service("semantic_search")
    
    @pytest.mark.asyncio
    async def test_restart_service_prevents_concurrent_restarts(self, proxy):
        """Concurrent restarts should be prevented."""
        proxy._restarting.add("semantic_search")
        
        # Should wait instead of starting another restart
        with patch.object(proxy, '_wait_for_service', new_callable=AsyncMock) as mock_wait:
            await proxy._restart_service("semantic_search")
            mock_wait.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_wait_for_service_succeeds_when_healthy(self, proxy):
        """Wait should succeed when service becomes healthy."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        with patch('httpx.AsyncClient.get', return_value=mock_response):
            # Should return immediately
            await proxy._wait_for_service("semantic_search", timeout=1.0)
    
    @pytest.mark.asyncio
    async def test_wait_for_service_times_out(self, proxy):
        """Wait should timeout if service doesn't become healthy."""
        with patch('httpx.AsyncClient.get', side_effect=Exception("Connection refused")):
            with pytest.raises(ServiceRestartError, match="did not become healthy"):
                await proxy._wait_for_service("semantic_search", timeout=0.5)


# =============================================================================
# RED Phase Tests - Tool Call Integration
# =============================================================================

class TestToolCallIntegration:
    """Tests for tool call with health-aware proxy."""
    
    @pytest.mark.asyncio
    async def test_call_tool_records_request(self, proxy):
        """Tool call should record request in idle tracker."""
        # Use the proxy's tracker directly
        tracker = proxy._tracker
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        
        with patch.object(proxy, '_is_service_dead', return_value=False):
            with patch.object(proxy, '_call_backend', return_value={"result": "success"}):
                await proxy.call_tool(
                    "semantic_search",
                    {"query": "test"},
                    "http://localhost:8081/v1/search",
                )
                
                # Verify request was recorded
                status = tracker.get_status("semantic_search")
                assert status["total_requests"] == 1
    
    @pytest.mark.asyncio
    async def test_call_tool_restarts_dead_service(self, proxy):
        """Tool call should restart service if dead."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        
        with patch.object(proxy, '_is_service_dead', return_value=True):
            with patch.object(proxy, '_restart_service', new_callable=AsyncMock) as mock_restart:
                with patch.object(proxy, '_call_backend', return_value={"result": "success"}):
                    await proxy.call_tool(
                        "semantic_search",
                        {"query": "test"},
                        "http://localhost:8081/v1/search",
                    )
                    
                    # Verify restart was called with timeout
                    mock_restart.assert_called_once_with("semantic_search", timeout=2.0)
    
    @pytest.mark.asyncio
    async def test_call_tool_retries_on_connect_error(self, proxy):
        """Tool call should retry on connection errors."""
        call_count = 0
        
        async def mock_call_backend(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ConnectError("Connection refused")
            return {"result": "success"}
        
        with patch.object(proxy, '_is_service_dead', return_value=False):
            with patch.object(proxy, '_call_backend', side_effect=mock_call_backend):
                with patch.object(proxy, '_restart_service', new_callable=AsyncMock):
                    result = await proxy.call_tool(
                        "semantic_search",
                        {"query": "test"},
                        "http://localhost:8081/v1/search",
                    )
                    
                    # Should have been called twice (initial + retry)
                    assert call_count == 2
                    assert result == {"result": "success"}
    
    @pytest.mark.asyncio
    async def test_call_tool_fails_after_max_retries(self, proxy):
        """Tool call should fail after max retries."""
        with patch.object(proxy, '_is_service_dead', return_value=False):
            with patch.object(proxy, '_call_backend', side_effect=httpx.ConnectError("Connection refused")):
                with patch.object(proxy, '_restart_service', new_callable=AsyncMock):
                    with pytest.raises(httpx.ConnectError):
                        await proxy.call_tool(
                            "semantic_search",
                            {"query": "test"},
                            "http://localhost:8081/v1/search",
                        )


# =============================================================================
# RED Phase Tests - Idle Timeout Integration
# =============================================================================

class TestIdleTimeoutIntegration:
    """Tests for idle timeout integration with health proxy."""
    
    def test_tracker_records_request(self, tracker):
        """Tracker should record request timestamp."""
        tracker.record_request("test-service")
        
        status = tracker.get_status("test-service")
        assert status["total_requests"] == 1
        assert status["last_request"] is not None
    
    def test_tracker_detects_idle_service(self, tracker):
        """Tracker should detect idle services."""
        tracker.record_request("test-service")
        
        # Simulate time passing
        tracker._service_states["test-service"]["last_request"] = (
            datetime.now(timezone.utc) - timedelta(seconds=700)
        )
        
        assert tracker.is_idle("test-service") is True
    
    def test_tracker_detects_active_service(self, tracker):
        """Tracker should detect active services."""
        tracker.record_request("test-service")
        
        assert tracker.is_idle("test-service") is False
    
    def test_tracker_returns_services_needing_shutdown(self, tracker):
        """Tracker should return list of idle services."""
        tracker.record_request("idle-service")
        tracker._service_states["idle-service"]["last_request"] = (
            datetime.now(timezone.utc) - timedelta(seconds=700)
        )
        
        tracker.record_request("active-service")
        
        idle_services = tracker.get_services_needing_shutdown()
        
        assert "idle-service" in idle_services
        assert "active-service" not in idle_services
    
    def test_tracker_custom_timeout(self, tracker):
        """Tracker should support custom timeouts per service."""
        tracker.set_custom_timeout("fast-service", 300)
        
        tracker.record_request("fast-service")
        tracker._service_states["fast-service"]["last_request"] = (
            datetime.now(timezone.utc) - timedelta(seconds=400)
        )
        
        assert tracker.is_idle("fast-service") is True
        assert tracker.get_timeout_config("fast-service") == 300


# =============================================================================
# GREEN Phase - Implementation Verification
# =============================================================================

class TestImplementationVerification:
    """Verify implementation meets requirements."""
    
    def test_default_timeout_is_10_minutes(self):
        """Default timeout should be 600 seconds (10 minutes)."""
        tracker = IdleTimeoutTracker()
        assert tracker.default_timeout == 600
    
    def test_service_config_has_all_services(self):
        """Health proxy should have all services configured."""
        proxy = HealthAwareProxy()
        
        expected_services = [
            "semantic_search",
            "code_analyze",
            "llm_complete",
            "run_agent_function",
            "audit_quality_scan",
            "context_management",
            "amve_evaluate_fitness",
        ]
        
        for service in expected_services:
            assert service in proxy.SERVICE_CONFIG, f"Missing service: {service}"
    
    def test_tool_to_service_mapping(self):
        """Tool names should map to service keys."""
        proxy = HealthAwareProxy()
        
        mappings = {
            "semantic_search": "semantic_search",
            "code_analyze": "code_analyze",
            "llm_complete": "llm_complete",
            "audit_quality_scan": "audit_quality_scan",
        }
        
        for tool_name, expected_service in mappings.items():
            assert proxy._tool_to_service(tool_name) == expected_service
    
    def test_get_service_status(self):
        """Should return complete service status."""
        proxy = HealthAwareProxy()
        
        status = proxy.get_service_status("semantic_search")
        
        assert "name" in status
        assert "url" in status
        assert "is_restarting" in status
        assert "is_idle" in status
        assert "idle_seconds" in status
        assert "timeout" in status


# =============================================================================
# REFACTOR Phase - Will add after GREEN passes
# =============================================================================

# Will add:
# - Performance tests
# - Integration tests with real services
# - Metrics export
# - Configuration validation
