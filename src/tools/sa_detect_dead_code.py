"""sa_detect_dead_code — struct-analyzer :8088 POST /v1/analysis/dead-code."""
from src.tools.sa_base import create_dead_code_handler
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "sa_detect_dead_code"

def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    return create_dead_code_handler(TOOL_NAME, dispatcher, sanitizer)
