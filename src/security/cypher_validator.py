"""Cypher injection prevention — WBS-MCP5 (GREEN).

Validates Cypher queries are read-only by rejecting write operations
(CREATE, DELETE, DETACH, DROP, MERGE, SET, REMOVE) and administrative
commands (CALL dbms.*).

Reference: Strategy §4.3 (Cypher Injection — P0), §7.1 Control #12
"""

from __future__ import annotations

import logging
import re

_security_logger = logging.getLogger("mcp_gateway.security")

# Forbidden write keywords — matched as whole words with word boundaries.
# We use \b to avoid matching substrings in property names like "set_count".
_FORBIDDEN_KEYWORDS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(rf"\b{kw}\b", re.IGNORECASE)
    for kw in (
        "CREATE",
        "DELETE",
        "DETACH",
        "DROP",
        "MERGE",
        "REMOVE",
    )
)

# SET requires special handling: must be preceded by whitespace or newline
# to avoid matching property names like "n.set_count" or "dataset".
_SET_PATTERN = re.compile(r"(?<=\s)SET\b", re.IGNORECASE)

# Administrative command pattern
_ADMIN_PATTERN = re.compile(r"\bCALL\s+dbms\.", re.IGNORECASE)


class CypherValidationError(Exception):
    """Raised when a Cypher query fails validation."""


def _strip_string_literals(query: str) -> str:
    """Remove string literal contents so keywords inside quotes don't trigger.

    Handles both single and double-quoted strings.
    """
    return re.sub(r"'[^']*'", "''", re.sub(r'"[^"]*"', '""', query))


def validate_cypher(query: str) -> str:
    """Validate a Cypher query is read-only.

    1. Reject empty/whitespace queries.
    2. Strip string literals so keywords inside quotes don't trigger.
    3. Check for forbidden write keywords.
    4. Check for administrative CALL dbms.* commands.

    All blocked attempts are logged as SECURITY events.

    Returns the original *query* on success.
    Raises ``CypherValidationError`` on any failure.
    """
    if not query or not query.strip():
        _security_logger.warning("SECURITY event=cypher_injection detail='empty query'")
        raise CypherValidationError("empty query rejected")

    # Strip string literal contents for keyword scanning
    stripped = _strip_string_literals(query)

    # Check forbidden write keywords
    for pattern in _FORBIDDEN_KEYWORDS:
        if pattern.search(stripped):
            keyword = pattern.pattern.replace(r"\b", "").upper()
            _security_logger.warning(
                "SECURITY event=cypher_injection detail='forbidden keyword: %s' query=%r",
                keyword,
                query,
            )
            raise CypherValidationError(
                f"forbidden write operation: {keyword}"
            )

    # Check SET separately (needs preceding whitespace)
    if _SET_PATTERN.search(stripped):
        _security_logger.warning(
            "SECURITY event=cypher_injection detail='forbidden keyword: SET' query=%r",
            query,
        )
        raise CypherValidationError("forbidden write operation: SET")

    # Check administrative commands
    if _ADMIN_PATTERN.search(stripped):
        _security_logger.warning(
            "SECURITY event=cypher_injection detail='admin command blocked' query=%r",
            query,
        )
        raise CypherValidationError("admin command blocked: CALL dbms.*")

    return query
