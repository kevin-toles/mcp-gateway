"""sa_get_fitness_functions — struct-analyzer :8088 GET /v1/fitness/functions."""
from src.tools.sa_base import create_fitness_list_handler
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "sa_get_fitness_functions"

def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    return create_fitness_list_handler(TOOL_NAME, dispatcher, sanitizer)
