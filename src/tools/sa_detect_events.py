"""sa_detect_events — struct-analyzer :8088 POST /v1/analysis/events."""
from src.tools.sa_base import create_source_analysis_handler
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "sa_detect_events"

def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    return create_source_analysis_handler(TOOL_NAME, dispatcher, sanitizer)
