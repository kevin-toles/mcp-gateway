"""ResourceBudget tests — WBS-MCP6 (RED).

Covers AC-6.4 (token budget + tool call count), AC-6.5 (per-tool timeout),
AC-6.7 (memory limit enforcement).
"""

import pytest

from src.core.errors import ResourceExhaustedError
from src.models.budget import ResourceBudget


# ── AC-6.4: can_proceed / deduct ────────────────────────────────────────


class TestCanProceed:
    """ResourceBudget.can_proceed() gating."""

    def test_fresh_budget_can_proceed(self) -> None:
        b = ResourceBudget()
        assert b.can_proceed() is True

    def test_tokens_exhausted_cannot_proceed(self) -> None:
        b = ResourceBudget(max_tokens=100, tokens_used=100)
        assert b.can_proceed() is False

    def test_tools_exhausted_cannot_proceed(self) -> None:
        b = ResourceBudget(max_tool_calls=5, tools_called=5)
        assert b.can_proceed() is False

    def test_under_both_limits_can_proceed(self) -> None:
        b = ResourceBudget(max_tokens=100, max_tool_calls=10, tokens_used=50, tools_called=3)
        assert b.can_proceed() is True

    def test_tokens_over_cannot_proceed(self) -> None:
        b = ResourceBudget(max_tokens=100, tokens_used=101)
        assert b.can_proceed() is False


class TestDeduct:
    """ResourceBudget.deduct() atomic deduction."""

    def test_deduct_tokens_succeeds(self) -> None:
        b = ResourceBudget(max_tokens=1000)
        b.deduct(tokens=100)
        assert b.tokens_used == 100
        assert b.tools_called == 1

    def test_deduct_multiple_times(self) -> None:
        b = ResourceBudget(max_tokens=1000, max_tool_calls=10)
        b.deduct(tokens=200)
        b.deduct(tokens=300)
        assert b.tokens_used == 500
        assert b.tools_called == 2

    def test_deduct_exceeding_tokens_raises(self) -> None:
        b = ResourceBudget(max_tokens=100, tokens_used=90)
        with pytest.raises(ResourceExhaustedError, match="token"):
            b.deduct(tokens=20)

    def test_deduct_exceeding_tools_raises(self) -> None:
        b = ResourceBudget(max_tool_calls=2, tools_called=2)
        with pytest.raises(ResourceExhaustedError, match="tool"):
            b.deduct(tokens=0)

    def test_deduct_zero_tokens_still_increments_tools(self) -> None:
        b = ResourceBudget(max_tokens=1000, max_tool_calls=10)
        b.deduct(tokens=0)
        assert b.tokens_used == 0
        assert b.tools_called == 1

    def test_deduct_does_not_mutate_on_failure(self) -> None:
        b = ResourceBudget(max_tokens=100, tokens_used=90, tools_called=3)
        with pytest.raises(ResourceExhaustedError):
            b.deduct(tokens=20)
        assert b.tokens_used == 90
        assert b.tools_called == 3


class TestAllocate:
    """ResourceBudget.allocate() depth-based subdivision."""

    def test_allocate_depth_1_halves_budget(self) -> None:
        b = ResourceBudget(max_tokens=10000, max_tool_calls=20)
        child = b.allocate(depth=1)
        assert child.max_tokens == 5000
        assert child.max_tool_calls == 10

    def test_allocate_depth_2_quarters_budget(self) -> None:
        b = ResourceBudget(max_tokens=10000, max_tool_calls=20)
        child = b.allocate(depth=2)
        assert child.max_tokens == 2500
        assert child.max_tool_calls == 5

    def test_allocate_preserves_timeout(self) -> None:
        b = ResourceBudget(tool_timeout_seconds=60.0)
        child = b.allocate(depth=1)
        assert child.tool_timeout_seconds == 60.0

    def test_allocate_resets_usage(self) -> None:
        b = ResourceBudget(max_tokens=10000, tokens_used=5000, tools_called=3)
        child = b.allocate(depth=1)
        assert child.tokens_used == 0
        assert child.tools_called == 0

    def test_allocate_depth_0_returns_full_budget(self) -> None:
        b = ResourceBudget(max_tokens=10000, max_tool_calls=20)
        child = b.allocate(depth=0)
        assert child.max_tokens == 10000
        assert child.max_tool_calls == 20

    def test_allocate_floor_prevents_zero(self) -> None:
        b = ResourceBudget(max_tokens=3, max_tool_calls=1)
        child = b.allocate(depth=1)
        assert child.max_tokens >= 1
        assert child.max_tool_calls >= 1


# ── AC-6.5: Per-tool timeout ────────────────────────────────────────────


class TestToolTimeout:
    """Per-tool timeout configuration."""

    def test_default_timeout_30s(self) -> None:
        b = ResourceBudget()
        assert b.tool_timeout_seconds == 30.0

    def test_custom_timeout(self) -> None:
        b = ResourceBudget(tool_timeout_seconds=60.0)
        assert b.tool_timeout_seconds == 60.0


# ── AC-6.7: Memory limit ────────────────────────────────────────────────


class TestMemoryLimit:
    """Memory limit enforcement."""

    def test_default_memory_512mb(self) -> None:
        b = ResourceBudget()
        assert b.max_memory_bytes == 536_870_912

    def test_custom_memory_limit(self) -> None:
        b = ResourceBudget(max_memory_bytes=1_073_741_824)
        assert b.max_memory_bytes == 1_073_741_824

    def test_check_memory_under_limit_passes(self) -> None:
        b = ResourceBudget(max_memory_bytes=536_870_912)
        # Should not raise when under limit
        b.check_memory(current_bytes=100_000_000)

    def test_check_memory_over_limit_raises(self) -> None:
        b = ResourceBudget(max_memory_bytes=536_870_912)
        with pytest.raises(ResourceExhaustedError, match="memory"):
            b.check_memory(current_bytes=600_000_000)

    def test_check_memory_at_limit_raises(self) -> None:
        b = ResourceBudget(max_memory_bytes=536_870_912)
        with pytest.raises(ResourceExhaustedError, match="memory"):
            b.check_memory(current_bytes=536_870_912)


# ── Dataclass defaults ──────────────────────────────────────────────────


class TestBudgetDefaults:
    """Verify all defaults match the WBS schema."""

    def test_defaults(self) -> None:
        b = ResourceBudget()
        assert b.max_tokens == 50_000
        assert b.max_tool_calls == 20
        assert b.tool_timeout_seconds == 30.0
        assert b.max_concurrent_agents == 5
        assert b.max_output_bytes == 1_048_576
        assert b.max_memory_bytes == 536_870_912
        assert b.tokens_used == 0
        assert b.tools_called == 0
