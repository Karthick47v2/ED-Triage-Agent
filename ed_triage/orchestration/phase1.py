"""Phase 1 orchestration: IIA -> CRA -> PAA."""

import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from ed_triage.common.checkpoint_serde import checkpoint_jsonplus_serde
from ed_triage.cra.agent import run_cra
from ed_triage.iia.agent import check_conversation_end, extraction_node, interview_node
from ed_triage.iia.state import AgentState
from ed_triage.paa.agent import run_paa

logger = logging.getLogger("Orchestrator")


def cra_node(state: AgentState):
    logger.info("Entering CRA Node...")
    intake_data = state.get("intake_data")
    if not intake_data:
        raise ValueError("CRA Node called without 'intake_data'.")
    return {"cra_result": run_cra(intake_data)}


def paa_node(state: AgentState):
    logger.info("Entering PAA Node...")
    intake_data = state.get("intake_data")
    cra_result = state.get("cra_result")
    if not intake_data or not cra_result:
        raise ValueError(
            "PAA Node called without 'intake_data' or 'cra_result'.")
    return {"paa_result": run_paa(intake_data, cra_result)}


def build_phase1_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("interviewer", interview_node)
    workflow.add_node("extractor", extraction_node)
    workflow.add_node("cra", cra_node)
    workflow.add_node("paa", paa_node)
    workflow.set_entry_point("interviewer")
    workflow.add_conditional_edges(
        "interviewer",
        check_conversation_end,
        {"extractor": "extractor", "__end__": END},
    )
    workflow.add_edge("extractor", "cra")
    workflow.add_edge("cra", "paa")
    workflow.add_edge("paa", END)
    memory = MemorySaver(serde=checkpoint_jsonplus_serde())
    return workflow.compile(checkpointer=memory)
