"""sa_platform_scan — struct-analyzer :8088 POST /v1/architecture/platform-scan."""
from src.tools.sa_base import create_platform_scan_handler
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "sa_platform_scan"

def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    return create_platform_scan_handler(TOOL_NAME, dispatcher, sanitizer)
