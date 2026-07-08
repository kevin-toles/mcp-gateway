"""sa_architecture_mapping_log — struct-analyzer :8088 POST /v1/architecture/mapping-log."""
from src.tools.sa_base import create_mapping_log_handler
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "sa_architecture_mapping_log"

def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    return create_mapping_log_handler(TOOL_NAME, dispatcher, sanitizer)
