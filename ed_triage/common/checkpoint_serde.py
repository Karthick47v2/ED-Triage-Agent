"""Configure LangGraph checkpoint serialization for application models."""

from __future__ import annotations

from langgraph.checkpoint.serde import _msgpack as langgraph_msgpack
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


APP_MSGPACK_TYPES: frozenset[tuple[str, str]] = frozenset(
    {
        ("ed_triage.iia.schema", "IntakeSummary"),
        ("ed_triage.iia.schema", "Symptom"),
        ("ed_triage.cra.schema", "CRAResult"),
        ("ed_triage.cra.schema", "DifferentialDiagnosis"),
        ("ed_triage.paa.schema", "PriorityAssessment"),
    }
)


def checkpoint_jsonplus_serde() -> JsonPlusSerializer:
    """Create a serializer allowlisting LangGraph and application types."""
    allowed_types = langgraph_msgpack.SAFE_MSGPACK_TYPES | APP_MSGPACK_TYPES

    return JsonPlusSerializer(
        allowed_msgpack_modules=tuple(sorted(allowed_types)),
    )
