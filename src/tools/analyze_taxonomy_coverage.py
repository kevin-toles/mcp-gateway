"""analyze_taxonomy_coverage tool handler — WBS-TAP9."""

from src.models.schemas import AnalyzeTaxonomyCoverageInput
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "analyze_taxonomy_coverage"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""

    async def analyze_taxonomy_coverage(
        taxonomy_path: str,
        output_path: str | None = None,
        collection: str = "all",
        top_k: int = 10,
        threshold: float = 0.3,
        max_leaf_nodes: int = 500,
        subtree_root: str | None = None,
        concurrency: int = 10,
        include_evidence: bool = True,
        scoring_weights: dict[str, float] | None = None,
    ) -> dict:
        """Analyze taxonomy coverage against the knowledge base."""
        validated = AnalyzeTaxonomyCoverageInput(
            taxonomy_path=taxonomy_path,
            output_path=output_path,
            collection=collection,
            top_k=top_k,
            threshold=threshold,
            max_leaf_nodes=max_leaf_nodes,
            subtree_root=subtree_root,
            concurrency=concurrency,
            include_evidence=include_evidence,
            scoring_weights=scoring_weights,
        )
        result = await dispatcher.dispatch(TOOL_NAME, validated.model_dump())
        return sanitizer.sanitize(result.body)

    return analyze_taxonomy_coverage
