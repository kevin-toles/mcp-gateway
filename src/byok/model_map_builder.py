"""BEH-04.01 Step 2 — BYOK model_map builder.

Single exported function: build_model_map()

Reads BYOK env vars set by the MCP client (ANTHROPIC_API_KEY, OPENAI_API_KEY,
PLATFORM_PREFERRED_MODEL) and builds the model_map dict injected into the
ai-agents request payload.

When model_map is non-empty, ai-agents ModelResolver tier 3 (runtime override)
wins for every pipeline role, routing all LLM calls to the user-configured
provider without bypassing the governance layer (CAP-04 / AC-ASCP.18).

The API key itself is read from env by the llm-gateway service using its own
ANTHROPIC_API_KEY / OPENAI_API_KEY config; mcp-gateway only constructs the
model routing map (key forwarding is tracked as open question §9 #1).
"""

from __future__ import annotations

import os

# Pipeline roles that ModelResolver resolves (PIPELINE_PROTOCOL stages)
_PIPELINE_ROLES: tuple[str, ...] = (
    "ARCHITECT",
    "ENGINEER",
    "REVIEWER",
    "FINALIZER",
    "VALIDATOR",
)

# Default model per provider when PLATFORM_PREFERRED_MODEL is not set
_ANTHROPIC_DEFAULT_MODEL: str = "claude-sonnet-4-20250514"
_OPENAI_DEFAULT_MODEL: str = "gpt-4o"


def build_model_map() -> dict[str, str]:
    """Build a ModelResolver tier-3 override map from BYOK environment variables.

    Priority (highest to lowest):
      1. PLATFORM_PREFERRED_MODEL — explicit model name wins over provider default
      2. ANTHROPIC_API_KEY present — route all roles to claude-sonnet-4-20250514
      3. OPENAI_API_KEY present    — route all roles to gpt-4o
      4. None set                  — return empty dict (ai-agents uses own defaults)

    Returns:
        Dict mapping each PIPELINE_PROTOCOL role to a model ID, or {} when no
        BYOK env vars are configured (ModelResolver tier 3 is not activated).
    """
    anthropic_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    openai_key: str = os.getenv("OPENAI_API_KEY", "")
    preferred_model: str = os.getenv("PLATFORM_PREFERRED_MODEL", "")

    if not anthropic_key and not openai_key and not preferred_model:
        return {}

    if preferred_model:
        model = preferred_model
    elif anthropic_key:
        model = _ANTHROPIC_DEFAULT_MODEL
    else:
        model = _OPENAI_DEFAULT_MODEL

    return {role: model for role in _PIPELINE_ROLES}


__all__ = ["build_model_map"]
