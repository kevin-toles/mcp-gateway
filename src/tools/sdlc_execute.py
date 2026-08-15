"""ASCP.INT.3 / CAP-01 / BEH-04.01: `sdlc_execute` tool handler.

Submits a WBS document through the full COMP-07 → COMP-08 → COMP-09 → COMP-10
orchestration pipeline in ai-agents and returns a ValidationReport.

BEH-04.01 Step 2: ANTHROPIC_API_KEY / OPENAI_API_KEY / PLATFORM_PREFERRED_MODEL
env vars are intercepted here and injected as model_map into the ai-agents
payload so ModelResolver tier 3 (BYOK override) wins for all pipeline roles.
"""

from __future__ import annotations

from src.byok.model_map_builder import build_model_map
from src.models.schemas import SDLCExecuteInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "sdlc_execute"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def sdlc_execute(wbs_content: str, config: dict | None = None) -> dict:
        """Execute the full SDLC pipeline for a WBS document.

        Submits wbs_content through WorkflowOrchestrator → PlanningAgent →
        ImplementationAgent → ValidationAgent and returns a ValidationReport
        with verdict (PASS/FAIL), failures, and execution metadata.

        BYOK (BEH-04.01): If ANTHROPIC_API_KEY, OPENAI_API_KEY, or
        PLATFORM_PREFERRED_MODEL is set in the MCP client environment,
        build_model_map() constructs the model_map override so that all
        pipeline LLM calls route to the user's provider (CAP-04).
        """
        validated = SDLCExecuteInput(
            wbs_content=wbs_content,
            config=config or {},
            model_map=build_model_map(),
        )
        payload = validated.model_dump()
        result = await dispatcher.dispatch(TOOL_NAME, payload)
        return sanitizer.sanitize(result.body)

    return sdlc_execute
