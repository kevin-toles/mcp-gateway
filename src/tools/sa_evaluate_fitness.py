"""sa_evaluate_fitness — struct-analyzer :8088 POST /v1/fitness/evaluate."""
from src.tools.sa_base import create_fitness_eval_handler
from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher

TOOL_NAME = "sa_evaluate_fitness"

def create_handler(dispatcher: ToolDispatcher, sanitizer: OutputSanitizer):
    return create_fitness_eval_handler(TOOL_NAME, dispatcher, sanitizer)
