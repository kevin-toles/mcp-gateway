"""Path traversal prevention — WBS-MCP5 (GREEN).

Validates file paths against traversal attacks (CWE-22), null-byte injection,
URL-encoded sequences, and symlink escapes.

Reference: Strategy §4.4 (Path Traversal — P0), §7.1 Control #2
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import unquote

_security_logger = logging.getLogger("mcp_gateway.security")


class PathValidationError(Exception):
    """Raised when a file path fails validation."""


def _double_decode(path: str) -> str:
    """Apply URL decoding twice to catch double-encoded sequences."""
    return unquote(unquote(path))


def _has_traversal_sequences(path: str) -> bool:
    """Check for dot-dot traversal in the decoded path.

    Normalizes backslashes to forward slashes before checking.
    """
    normalized = path.replace("\\", "/")
    parts = normalized.split("/")
    return ".." in parts


def validate_file_path(path: str, allowed_roots: list[Path]) -> Path:
    """Validate a file path against traversal attacks and allowed roots.

    1. Reject empty paths.
    2. Double URL-decode and check for null bytes.
    3. Check for traversal sequences (../) in decoded path.
    4. Resolve symlinks and verify the real path is under an allowed root.
    5. Verify the file exists.

    All blocked attempts are logged as SECURITY events.

    Returns the resolved ``Path`` on success.
    Raises ``PathValidationError`` on any failure.
    """
    if not path:
        _security_logger.warning("SECURITY event=path_traversal detail='empty path'")
        raise PathValidationError("empty path rejected")

    # Double-decode to catch %252e%252e%252f
    decoded = _double_decode(path)

    # Null byte check (on both original and decoded)
    if "\x00" in path or "\x00" in decoded:
        _security_logger.warning(
            "SECURITY event=null_byte detail='null byte in path' path=%r", path
        )
        raise PathValidationError("null byte in path rejected")

    # Traversal sequence check on decoded path
    if _has_traversal_sequences(decoded) or _has_traversal_sequences(path):
        _security_logger.warning(
            "SECURITY event=path_traversal detail='traversal sequence detected' path=%r", path
        )
        raise PathValidationError("path traversal detected")

    # Resolve to real path (follows symlinks)
    resolved = Path(decoded).resolve()

    # Check that resolved path is under at least one allowed root
    under_root = any(
        _is_under_root(resolved, root.resolve()) for root in allowed_roots
    )
    if not under_root:
        _security_logger.warning(
            "SECURITY event=path_outside_root detail='resolved path outside allowed roots' "
            "path=%r resolved=%s",
            path,
            resolved,
        )
        raise PathValidationError("path outside allowed roots")

    # Verify file exists
    if not resolved.exists():
        _security_logger.warning(
            "SECURITY event=path_not_found detail='file does not exist' path=%r", path
        )
        raise PathValidationError("file does not exist")

    return resolved


def _is_under_root(path: Path, root: Path) -> bool:
    """Check if *path* is equal to or a child of *root*."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
