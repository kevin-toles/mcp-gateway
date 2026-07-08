"""sa_build_call_graph — struct-analyzer :8088 POST /v1/analysis/call-graph."""
from src.tools.sa_base import create_call_graph_handler
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "sa_build_call_graph"

def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    return create_call_graph_handler(TOOL_NAME, dispatcher, sanitizer)
