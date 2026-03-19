# MCP Tool Parameter Standardization - Issue & Solution

## Problem

External LLMs calling MCP tools have no way to know parameter names vary:

**Current inconsistency:**
- `semantic_search` → `top_k`
- `hybrid_search` → `top_k`
- `knowledge_search` → `limit`
- `knowledge_refine` → `limit`
- `pattern_search` → `limit`
- `diagram_search` → `limit`

This causes validation failures when an LLM guesses wrong:
```python
mcp_ai-kitchen-br_diagram_search(query="...", top_k=3)
# ❌ Pydantic:unexpected_keyword_argument: top_k

mcp_ai-kitchen-br_semantic_search(query="...", limit=3)
# ❌ Pydantic validation: unexpected_keyword_argument: limit
```

## Root Cause

File: `/Users/kevintoles/POC/mcp-gateway/src/models/schemas.py`

**Legacy tools** (lines 37-51): Use `top_k`
- SemanticSearchInput
- HybridSearchInput

**New KB tools** (lines 816-950): Use `limit`
- KnowledgeSearchInput
- KnowledgeRefineInput
- PatternSearchInput
- DiagramSearchInput

## Solution: Accept Both Parameters

Add field validators that normalize EITHER parameter name to a canonical form:

```python
class SemanticSearchInput(BaseModel):
    """Input for semantic_search tool."""

    query: str = Field(..., min_length=1, max_length=2000)
    collection: str = Field(default="all", pattern=r"^(code|docs|textbooks|all)$")

    # Accept BOTH parameter names (backwards compatible)
    top_k: int | None = Field(default=None, ge=1, le=100)
    limit: int | None = Field(default=None, ge=1, le=100)

    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def normalize_result_count(self) -> "SemanticSearchInput":
        """Accept top_k OR limit, normalize to top_k."""
        if self.top_k is None and self.limit is None:
            self.top_k = 10  # default
        elif self.top_k is None:
            self.top_k = self.limit  # normalize limit → top_k
        elif self.limit is not None and self.limit != self.top_k:
            raise ValueError("Cannot specify both top_k and limit with different values")
        return self


class DiagramSearchInput(BaseModel):
    """Input for diagram_search — semantic search over ASCII / sequence diagrams."""

    query: str = Field(..., min_length=1, max_length=2000)
    diagram_type: str | None = Field(default=None, ...)

    # Accept BOTH parameter names (backwards compatible)
    limit: int | None = Field(default=None, ge=1, le=30)
    top_k: int | None = Field(default=None, ge=1, le=30)

    @model_validator(mode="after")
    def normalize_result_count(self) -> "DiagramSearchInput":
        """Accept limit OR top_k, normalize to limit."""
        if self.limit is None and self.top_k is None:
            self.limit = 10  # default
        elif self.limit is None:
            self.limit = self.top_k  # normalize top_k → limit
        elif self.top_k is not None and self.top_k != self.limit:
            raise ValueError("Cannot specify both limit and top_k with different values")
        return self
```

## Benefits

✅ **Zero breaking changes** - existing clients continue to work
✅ **LLM-friendly** - works regardless of which parameter name is used
✅ **Validation** - prevents contradictory values (`top_k=5, limit=10`)
✅ **Consistent internals** - each tool normalizes to its preferred canonical form

## Implementation Plan

1. Update `SemanticSearchInput` + `HybridSearchInput` to accept both `top_k` and `limit`
2. Update `KnowledgeSearchInput`, `KnowledgeRefineInput`, `PatternSearchInput`, `DiagramSearchInput` to accept both
3. Add `@model_validator` to normalize to canonical form
4. Test with both parameter names
5. Document in tool descriptions which is preferred (but both work)

## Files to Modify

- `/Users/kevintoles/POC/mcp-gateway/src/models/schemas.py` (lines 37-950)
- `/Users/kevintoles/POC/mcp-gateway/tests/unit/models/test_schemas.py` (add dual-parameter tests)

## Migration Timeline

**Phase 1** (immediate): Accept both, log deprecation warning for non-canonical form
**Phase 2** (6 months): Remove deprecation warning, both are first-class
**Phase 3** (never): Keep both forever — UX is more important than purity
