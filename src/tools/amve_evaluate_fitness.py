"""amve_evaluate_fitness tool handler — AEI-7.

Dispatches to AMVE :8088 POST /v1/fitness/evaluate.
Evaluates architecture fitness functions against a snapshot.

AC-AEI7.4: Accepts dimensions array (fitness_function_ids) and
baseline_snapshot (snapshot_id) params.
"""

from src.models.schemas import AMVEEvaluateFitnessInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "amve_evaluate_fitness"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def amve_evaluate_fitness(
        snapshot_id: str,
        fitness_function_ids: list[str] | None = None,
    ) -> dict:
        """Evaluate fitness functions against an architecture snapshot."""
        validated = AMVEEvaluateFitnessInput(
            snapshot_id=snapshot_id,
            fitness_function_ids=fitness_function_ids,
        )
        payload = validated.model_dump()
        result = await dispatcher.dispatch(TOOL_NAME, payload)
        return sanitizer.sanitize(result.body)

    return amve_evaluate_fitness
