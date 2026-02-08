"""ResourceBudget — WBS-MCP6 (GREEN).

Pre-execution token budget and tool-call limiter with depth-based
allocation for nested agent calls.

Reference: Strategy §5.3 (Resource Exhaustion — P1), §6.5 (Usage Billing),
           §7.1 Controls #4, #5, #13
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.errors import ResourceExhaustedError


@dataclass
class ResourceBudget:
    """Per-request resource budget.

    Tracks token usage and tool call count against configurable limits.
    Supports depth-based subdivision for nested agent calls.
    """

    max_tokens: int = 50_000
    max_tool_calls: int = 20
    tool_timeout_seconds: float = 30.0
    max_concurrent_agents: int = 5
    max_output_bytes: int = 1_048_576       # 1 MB
    max_memory_bytes: int = 536_870_912     # 512 MB
    tokens_used: int = 0
    tools_called: int = 0

    def can_proceed(self) -> bool:
        """Return ``True`` if both token and tool-call budgets have headroom."""
        return self.tokens_used < self.max_tokens and self.tools_called < self.max_tool_calls

    def deduct(self, tokens: int) -> None:
        """Deduct *tokens* and increment tool counter.

        Raises ``ResourceExhaustedError`` if the deduction would exceed
        either limit.  The budget is **not** mutated on failure.
        """
        new_tokens = self.tokens_used + tokens
        new_tools = self.tools_called + 1

        if new_tools > self.max_tool_calls:
            raise ResourceExhaustedError(
                "tool_calls",
                f"{new_tools} exceeds max {self.max_tool_calls}",
            )
        if new_tokens > self.max_tokens:
            raise ResourceExhaustedError(
                "token_budget",
                f"{new_tokens} exceeds max {self.max_tokens}",
            )

        self.tokens_used = new_tokens
        self.tools_called = new_tools

    def allocate(self, depth: int) -> ResourceBudget:
        """Create a child budget halved per *depth* level.

        ``depth=0`` returns a full copy; ``depth=1`` halves; ``depth=2``
        quarters, etc.  Usage counters are reset.  A floor of 1 prevents
        zero-budget children.
        """
        divisor = 2 ** depth
        return ResourceBudget(
            max_tokens=max(1, self.max_tokens // divisor),
            max_tool_calls=max(1, self.max_tool_calls // divisor),
            tool_timeout_seconds=self.tool_timeout_seconds,
            max_concurrent_agents=self.max_concurrent_agents,
            max_output_bytes=self.max_output_bytes,
            max_memory_bytes=self.max_memory_bytes,
            tokens_used=0,
            tools_called=0,
        )

    def check_memory(self, current_bytes: int) -> None:
        """Raise ``ResourceExhaustedError`` if *current_bytes* ≥ limit."""
        if current_bytes >= self.max_memory_bytes:
            raise ResourceExhaustedError(
                "memory",
                f"{current_bytes} bytes >= limit {self.max_memory_bytes}",
            )
