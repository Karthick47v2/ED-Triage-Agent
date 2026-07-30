"""Phase 1 orchestration: IIA -> CRA -> PAA."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ed_triage.common.checkpoint_serde import checkpoint_jsonplus_serde
from ed_triage.cra.agent import run_cra
from ed_triage.iia.agent import (
    check_conversation_end,
    extraction_node,
    interview_node,
)
from ed_triage.iia.state import AgentState
from ed_triage.paa.agent import run_paa

logger = logging.getLogger(__name__)


def require_state_value(state: AgentState, key: str) -> Any:
    """Return a required state value or raise a descriptive error."""
    value = state.get(key)

    if value is None:
        raise ValueError(f"Required state field is missing: {key!r}")

    return value


def cra_node(state: AgentState) -> dict[str, Any]:
    """Run CRA on extracted intake and store the result."""
    logger.info("Entering CRA node")

    intake_data = require_state_value(state, "intake_data")
    return {"cra_result": run_cra(intake_summary=intake_data)}


def paa_node(state: AgentState) -> dict[str, Any]:
    """Run PAA on intake + CRA and store the priority assessment."""
    logger.info("Entering PAA node")

    intake_data = require_state_value(state, "intake_data")
    cra_result = require_state_value(state, "cra_result")

    return {
        "paa_result": run_paa(
            intake_summary=intake_data,
            cra_result=cra_result,
        )
    }


def build_phase1_graph() -> CompiledStateGraph:
    """Build and compile the Phase 1 IIA -> CRA -> PAA graph."""
    workflow = StateGraph(AgentState)

    workflow.add_node("interviewer", interview_node)
    workflow.add_node("extractor", extraction_node)
    workflow.add_node("cra", cra_node)
    workflow.add_node("paa", paa_node)

    workflow.set_entry_point("interviewer")

    workflow.add_conditional_edges(
        "interviewer",
        check_conversation_end,
        {
            "extractor": "extractor",
            "__end__": END,
        },
    )

    workflow.add_edge("extractor", "cra")
    workflow.add_edge("cra", "paa")
    workflow.add_edge("paa", END)

    checkpointer = MemorySaver(serde=checkpoint_jsonplus_serde())
    return workflow.compile(checkpointer=checkpointer)
