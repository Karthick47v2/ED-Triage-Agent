"""LangGraph checkpoint serde: allowlist app Pydantic types for msgpack round-trip."""

from langgraph.checkpoint.serde import _msgpack as _lg_msgpack
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# State channels persist these as EXT_PYDANTIC_V2.
_APP_MSGPACK_TYPES = frozenset(
    {
        ("ed_triage.iia.schema", "IntakeSummary"),
        ("ed_triage.iia.schema", "Symptom"),
        ("ed_triage.cra.schema", "CRAResult"),
        ("ed_triage.cra.schema", "DifferentialDiagnosis"),
        ("ed_triage.paa.schema", "PriorityAssessment"),
    }
)


def checkpoint_jsonplus_serde() -> JsonPlusSerializer:
    # Default JsonPlusSerializer uses allowed_msgpack_modules=True when strict mode is
    # off. In that case ``with_msgpack_allowlist()`` is a no-op, so app types never get
    # registered and every pydantic ext logs "Deserializing unregistered type".
    # Start from LangGraph's safe builtins + messages, then add our models
    # explicitly.
    allowed = set(_lg_msgpack.SAFE_MSGPACK_TYPES) | set(_APP_MSGPACK_TYPES)
    return JsonPlusSerializer(allowed_msgpack_modules=tuple(allowed))
