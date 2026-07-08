"""sa_extract_architecture — struct-analyzer :8088 POST /v1/architecture/extract."""
from src.tools.sa_base import create_extract_handler
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "sa_extract_architecture"

def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    return create_extract_handler(TOOL_NAME, dispatcher, sanitizer)
