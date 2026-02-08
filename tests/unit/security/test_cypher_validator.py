"""Cypher injection prevention tests — WBS-MCP5 (RED).

Covers AC-5.4 (block write operations), AC-5.5 (block admin commands),
AC-5.6 (parameters, not interpolation), AC-5.7 (security event logging).
"""

import logging

import pytest

from src.security.cypher_validator import CypherValidationError, validate_cypher


# ── AC-5.4: Block write operations ──────────────────────────────────────


class TestCypherWriteBlocking:
    """All write operations are rejected."""

    @pytest.mark.parametrize(
        "cypher",
        [
            "MATCH (n) DELETE n",
            "MATCH (n)-[r]-() DELETE r",
            "match (n) delete n",  # case-insensitive
        ],
        ids=["delete-node", "delete-rel", "lowercase-delete"],
    )
    def test_delete_rejected(self, cypher: str) -> None:
        with pytest.raises(CypherValidationError, match="forbidden"):
            validate_cypher(cypher)

    @pytest.mark.parametrize(
        "cypher",
        [
            "MATCH (n) DETACH DELETE n",
            "MATCH (n:User) DETACH DELETE n",
            "match (n) detach delete n",
        ],
        ids=["detach-delete", "detach-delete-label", "lowercase"],
    )
    def test_detach_delete_rejected(self, cypher: str) -> None:
        with pytest.raises(CypherValidationError, match="forbidden"):
            validate_cypher(cypher)

    @pytest.mark.parametrize(
        "cypher",
        [
            "CREATE (n:Node {name: 'test'})",
            "CREATE (n:Node)-[:REL]->(m:Node)",
            "create (n:Node)",
        ],
        ids=["create-node", "create-rel", "lowercase"],
    )
    def test_create_rejected(self, cypher: str) -> None:
        with pytest.raises(CypherValidationError, match="forbidden"):
            validate_cypher(cypher)

    @pytest.mark.parametrize(
        "cypher",
        [
            "DROP INDEX ON :Person(name)",
            "DROP CONSTRAINT ON (p:Person)",
            "drop index on :Person(name)",
        ],
        ids=["drop-index", "drop-constraint", "lowercase"],
    )
    def test_drop_rejected(self, cypher: str) -> None:
        with pytest.raises(CypherValidationError, match="forbidden"):
            validate_cypher(cypher)

    @pytest.mark.parametrize(
        "cypher",
        [
            "MERGE (n:Person {name: 'Alice'})",
            "MATCH (a) MERGE (a)-[:KNOWS]->(b:Person {name: 'Bob'})",
            "merge (n:Person {name: 'Alice'})",
        ],
        ids=["merge-node", "merge-rel", "lowercase"],
    )
    def test_merge_rejected(self, cypher: str) -> None:
        with pytest.raises(CypherValidationError, match="forbidden"):
            validate_cypher(cypher)

    @pytest.mark.parametrize(
        "cypher",
        [
            "MATCH (n:Person) SET n.age = 30",
            "MATCH (n) SET n.prop = 'val'",
            "MATCH (n) SET n += {prop: 'val'}",
            "match (n) set n.age = 30",
        ],
        ids=["set-property", "set-string", "set-map", "lowercase"],
    )
    def test_set_rejected(self, cypher: str) -> None:
        with pytest.raises(CypherValidationError, match="forbidden"):
            validate_cypher(cypher)

    @pytest.mark.parametrize(
        "cypher",
        [
            "MATCH (n:Person) REMOVE n.age",
            "MATCH (n) REMOVE n:Label",
            "match (n) remove n.age",
        ],
        ids=["remove-prop", "remove-label", "lowercase"],
    )
    def test_remove_rejected(self, cypher: str) -> None:
        with pytest.raises(CypherValidationError, match="forbidden"):
            validate_cypher(cypher)


# ── AC-5.5: Block administrative commands ───────────────────────────────


class TestCypherAdminBlocking:
    """CALL dbms.* administrative commands are rejected."""

    @pytest.mark.parametrize(
        "cypher",
        [
            "CALL dbms.security.listUsers()",
            "CALL dbms.security.createUser('evil', 'pass', false)",
            "CALL dbms.cluster.overview()",
            "call dbms.security.listUsers()",  # case-insensitive
            "CALL dbms.listConfig()",
        ],
        ids=["list-users", "create-user", "cluster", "lowercase", "list-config"],
    )
    def test_dbms_admin_rejected(self, cypher: str) -> None:
        with pytest.raises(CypherValidationError, match="admin"):
            validate_cypher(cypher)

    def test_call_apoc_allowed(self) -> None:
        """Non-admin CALL (like APOC) should still pass."""
        result = validate_cypher("CALL apoc.meta.data()")
        assert result == "CALL apoc.meta.data()"


# ── AC-5.4/5.6: Read-only queries pass ──────────────────────────────────


class TestCypherReadOnly:
    """Valid read-only queries are accepted."""

    @pytest.mark.parametrize(
        "cypher",
        [
            "MATCH (n:Function) RETURN n LIMIT 10",
            "MATCH (n)-[r:CALLS]->(m) RETURN n.name, m.name",
            "MATCH (n:Class) WHERE n.name = $name RETURN n",
            "MATCH (n) RETURN count(n)",
            "MATCH p=(a)-[:DEPENDS_ON*1..3]->(b) RETURN p",
            "RETURN 1",
        ],
        ids=["basic-match", "relationship", "parameterized", "count", "variable-length", "literal-return"],
    )
    def test_read_only_accepted(self, cypher: str) -> None:
        result = validate_cypher(cypher)
        assert result == cypher

    def test_keyword_in_string_literal_accepted(self) -> None:
        """Keywords inside string literals should NOT trigger blocking."""
        q = "MATCH (n) WHERE n.name = 'DELETE_ME' RETURN n"
        result = validate_cypher(q)
        assert result == q

    def test_keyword_in_property_name_accepted(self) -> None:
        """Property names like .set_count should not trigger."""
        q = "MATCH (n) WHERE n.set_count > 0 RETURN n"
        result = validate_cypher(q)
        assert result == q


# ── AC-5.7: Security event logging for Cypher ──────────────────────────


class TestCypherSecurityLogging:
    """Blocked Cypher attempts emit security events."""

    def test_write_logs_security_event(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="mcp_gateway.security"):
            with pytest.raises(CypherValidationError):
                validate_cypher("CREATE (n:Evil)")
        assert any("SECURITY" in r.message and "cypher_injection" in r.message for r in caplog.records)

    def test_admin_logs_security_event(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="mcp_gateway.security"):
            with pytest.raises(CypherValidationError):
                validate_cypher("CALL dbms.security.listUsers()")
        assert any("SECURITY" in r.message and "cypher_injection" in r.message for r in caplog.records)

    def test_valid_query_no_security_log(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="mcp_gateway.security"):
            validate_cypher("MATCH (n) RETURN n LIMIT 5")
        assert not any("SECURITY" in r.message for r in caplog.records)


# ── Edge cases ──────────────────────────────────────────────────────────


class TestCypherEdgeCases:
    """Tricky inputs."""

    def test_empty_query_rejected(self) -> None:
        with pytest.raises(CypherValidationError):
            validate_cypher("")

    def test_whitespace_only_rejected(self) -> None:
        with pytest.raises(CypherValidationError):
            validate_cypher("   ")

    def test_multiline_with_write_rejected(self) -> None:
        q = """MATCH (n:Person)
WHERE n.age > 30
DELETE n"""
        with pytest.raises(CypherValidationError, match="forbidden"):
            validate_cypher(q)

    def test_comment_with_write_keyword(self) -> None:
        """Cypher with // comment containing write keyword — the keyword is in the actual query."""
        q = "MATCH (n) DELETE n // cleaning up"
        with pytest.raises(CypherValidationError, match="forbidden"):
            validate_cypher(q)
