"""Decision Payload Schema Definition.

Standard schema for System 1 Decision Engine queries.
"""

from typing import Any

from pydantic import BaseModel, Field


class QueryContext(BaseModel):
    semantic_embedding: list[float] = Field(
        default_factory=list, description="Vector embedding of context"
    )
    symbolic_state: dict[str, Any] = Field(
        default_factory=dict, description="Symbolic world state representation"
    )
    history: list[dict[str, Any]] = Field(
        default_factory=list, description="Interaction or decision history"
    )


class DecisionPayload(BaseModel):
    query: str = Field(..., description="Target query or prompt")
    context: QueryContext = Field(
        default_factory=QueryContext, description="Semantic and symbolic context"
    )
    constraints: list[str] = Field(
        default_factory=list, description="Rule-based or symbolic constraints"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Query metadata")
