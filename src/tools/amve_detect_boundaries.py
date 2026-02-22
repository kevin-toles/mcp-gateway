"""amve_detect_boundaries tool handler — AEI-7.

Dispatches to AMVE :8088 POST /v1/analysis/boundaries.
Detects service boundaries in source code.
"""

from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher
from src.tools.amve_base import create_analysis_handler

TOOL_NAME = "amve_detect_boundaries"


def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    """Return an async handler with a typed signature for FastMCP schema generation."""
    return create_analysis_handler(TOOL_NAME, dispatcher, sanitizer)
