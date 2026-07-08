"""CMS tool base — handler factories for cms_* tools.

Each factory returns a typed async handler bound to a tool name and
dispatcher route. FastMCP derives the JSON schema from the function signature.
"""

from __future__ import annotations

from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher


def create_generic_post_handler(
    tool_name: str,
    dispatcher: ToolDispatcher,
    sanitizer: OutputSanitizer,
    param_names: list[str],
):
    """Generic factory for CMS tools — builds payload from named params."""

    async def handler(**kwargs) -> dict:  # type: ignore[misc]
        result = await dispatcher.dispatch(tool_name, kwargs)
        return sanitizer.sanitize(result.body)

    handler.__name__ = tool_name
    return handler
