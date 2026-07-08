"""sa_detect_drift — struct-analyzer :8088 POST /v1/architecture/drift."""
from src.tools.sa_base import create_drift_handler
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "sa_detect_drift"

def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    return create_drift_handler(TOOL_NAME, dispatcher, sanitizer)
