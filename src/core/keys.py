"""Canonical service key normalization utilities.

All service-key normalization in the mcp-gateway codebase should use
this module to ensure consistent hyphen-form keys throughout the system.
"""


def normalize_service_key(key: str) -> str:
    """Normalize a service key: replace underscores with hyphens and lowercase.

    The canonical internal representation uses hyphen-form keys (e.g.
    ``"semantic-search"``, ``"code-orchestrator"``) as defined in
    :data:`src.core.health_config.SERVICE_TIERS`.

    Also lowercases the result so that mixed-case inputs like ``"Semantic_Search"``
    produce ``"semantic-search"``.

    Args:
        key: A service key that may contain underscores (e.g. ``"semantic_search"``).

    Returns:
        The normalized key with hyphens and lowercased (e.g. ``"semantic-search"``).
    """
    return key.replace("_", "-").lower()
