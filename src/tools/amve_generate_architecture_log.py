"""amve_generate_architecture_log tool handler — AEI-7.

Dispatches to AMVE :8088 POST /v1/architecture/batch-scan.
Runs a batch scan with optional baseline comparison and event publishing.
"""

from typing import Any

from src.models.schemas import AMVEGenerateArchitectureLogInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "amve_generate_architecture_log"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def amve_generate_architecture_log(
        source_paths: list[str],
        violations: list[dict[str, Any]] | None = None,
        patterns: list[dict[str, Any]] | None = None,
        baseline_json: dict[str, Any] | None = None,
    ) -> dict:
        """Generate architecture log via batch scan with optional baseline."""
        validated = AMVEGenerateArchitectureLogInput(
            source_paths=source_paths,
            violations=violations or [],
            patterns=patterns or [],
            baseline_json=baseline_json,
        )
        payload = validated.model_dump()
        result = await dispatcher.dispatch(TOOL_NAME, payload)
        return sanitizer.sanitize(result.body)

    return amve_generate_architecture_log
