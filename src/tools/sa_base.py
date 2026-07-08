"""Struct-Analyzer tool base — handler factories for sa_* tools.

Each factory returns a typed async handler bound to a tool name and
dispatcher route. FastMCP derives the JSON schema from the function signature.
"""

from __future__ import annotations

from src.security.output_sanitizer import OutputSanitizer
from src.tool_dispatcher import ToolDispatcher


def create_source_analysis_handler(
    tool_name: str,
    dispatcher: ToolDispatcher,
    sanitizer: OutputSanitizer,
):
    """Factory for SA analysis tools that take source_path + language + exclude."""

    async def handler(
        source_path: str,
        language: str = "",
        exclude: list[str] | None = None,
    ) -> dict:
        payload: dict = {"source_path": source_path}
        if language:
            payload["language"] = language
        if exclude:
            payload["exclude"] = exclude
        result = await dispatcher.dispatch(tool_name, payload)
        return sanitizer.sanitize(result.body)

    handler.__name__ = tool_name
    return handler


def create_dead_code_handler(
    tool_name: str,
    dispatcher: ToolDispatcher,
    sanitizer: OutputSanitizer,
):
    """Factory for sa_detect_dead_code — source_path + optional call_graph_paths."""

    async def handler(
        source_path: str,
        call_graph_paths: list[str] | None = None,
    ) -> dict:
        payload: dict = {"source_path": source_path}
        if call_graph_paths:
            payload["call_graph_paths"] = call_graph_paths
        result = await dispatcher.dispatch(tool_name, payload)
        return sanitizer.sanitize(result.body)

    handler.__name__ = tool_name
    return handler


def create_call_graph_handler(
    tool_name: str,
    dispatcher: ToolDispatcher,
    sanitizer: OutputSanitizer,
):
    """Factory for sa_build_call_graph — source_paths[] + include_external + repo_id."""

    async def handler(
        source_paths: list[str],
        include_external: bool = False,
        repo_id: str = "",
    ) -> dict:
        payload: dict = {
            "source_paths": source_paths,
            "include_external": include_external,
        }
        if repo_id:
            payload["repo_id"] = repo_id
        result = await dispatcher.dispatch(tool_name, payload)
        return sanitizer.sanitize(result.body)

    handler.__name__ = tool_name
    return handler


def create_extract_handler(
    tool_name: str,
    dispatcher: ToolDispatcher,
    sanitizer: OutputSanitizer,
):
    """Factory for sa_extract_architecture."""

    async def handler(
        source_path: str,
        language: str = "",
        workers: int = 0,
        snapshot_id: str = "",
        repo_id: str = "",
    ) -> dict:
        payload: dict = {"source_path": source_path}
        if language:
            payload["language"] = language
        if workers:
            payload["workers"] = workers
        if snapshot_id:
            payload["snapshot_id"] = snapshot_id
        if repo_id:
            payload["repo_id"] = repo_id
        result = await dispatcher.dispatch(tool_name, payload)
        return sanitizer.sanitize(result.body)

    handler.__name__ = tool_name
    return handler


def create_drift_handler(
    tool_name: str,
    dispatcher: ToolDispatcher,
    sanitizer: OutputSanitizer,
):
    """Factory for sa_detect_drift — accepts snapshot dicts or SHA strings."""

    async def handler(
        snapshot_a: dict | None = None,
        snapshot_b: dict | None = None,
        snapshot_a_sha: str = "",
        snapshot_b_sha: str = "",
    ) -> dict:
        if snapshot_a_sha and snapshot_b_sha:
            payload: dict = {
                "snapshot_a_sha": snapshot_a_sha,
                "snapshot_b_sha": snapshot_b_sha,
            }
        else:
            payload = {"before": snapshot_a, "after": snapshot_b}
        result = await dispatcher.dispatch(tool_name, payload)
        return sanitizer.sanitize(result.body)

    handler.__name__ = tool_name
    return handler


def create_mapping_log_handler(
    tool_name: str,
    dispatcher: ToolDispatcher,
    sanitizer: OutputSanitizer,
):
    """Factory for sa_architecture_mapping_log — accepts single or multiple paths."""

    async def handler(
        source_path: str = "",
        source_paths: list[str] | None = None,
        language: str = "",
    ) -> dict:
        payload: dict = {}
        if source_paths:
            payload["source_paths"] = source_paths
        elif source_path:
            payload["source_path"] = source_path
        if language:
            payload["language"] = language
        result = await dispatcher.dispatch(tool_name, payload)
        return sanitizer.sanitize(result.body)

    handler.__name__ = tool_name
    return handler


def create_platform_scan_handler(
    tool_name: str,
    dispatcher: ToolDispatcher,
    sanitizer: OutputSanitizer,
):
    """Factory for sa_platform_scan — repo_paths[] + workers."""

    async def handler(
        repo_paths: list[str],
        workers: int = 0,
    ) -> dict:
        payload: dict = {"repo_paths": repo_paths}
        if workers:
            payload["workers"] = workers
        result = await dispatcher.dispatch(tool_name, payload)
        return sanitizer.sanitize(result.body)

    handler.__name__ = tool_name
    return handler


def create_batch_scan_handler(
    tool_name: str,
    dispatcher: ToolDispatcher,
    sanitizer: OutputSanitizer,
):
    """Factory for sa_batch_scan."""

    async def handler(
        source_paths: list[str],
        violations: list[dict] | None = None,
        patterns: list[dict] | None = None,
        baseline_json: dict | None = None,
    ) -> dict:
        payload: dict = {"source_paths": source_paths}
        if violations:
            payload["violations"] = violations
        if patterns:
            payload["patterns"] = patterns
        if baseline_json:
            payload["baseline_json"] = baseline_json
        result = await dispatcher.dispatch(tool_name, payload)
        return sanitizer.sanitize(result.body)

    handler.__name__ = tool_name
    return handler


def create_fitness_eval_handler(
    tool_name: str,
    dispatcher: ToolDispatcher,
    sanitizer: OutputSanitizer,
):
    """Factory for sa_evaluate_fitness."""

    async def handler(
        function_id: str = "",
        function: dict | None = None,
        snapshot: dict | None = None,
    ) -> dict:
        payload: dict = {}
        if function_id:
            payload["function_id"] = function_id
        if function:
            payload["function"] = function
        if snapshot:
            payload["snapshot"] = snapshot
        result = await dispatcher.dispatch(tool_name, payload)
        return sanitizer.sanitize(result.body)

    handler.__name__ = tool_name
    return handler


def create_fitness_list_handler(
    tool_name: str,
    dispatcher: ToolDispatcher,
    sanitizer: OutputSanitizer,
):
    """Factory for sa_get_fitness_functions — GET, no body."""

    async def handler() -> dict:
        result = await dispatcher.dispatch(tool_name, {}, method="GET")
        return sanitizer.sanitize(result.body)

    handler.__name__ = tool_name
    return handler
