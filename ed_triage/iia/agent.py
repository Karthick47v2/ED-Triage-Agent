import logging
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END

from ed_triage.common.llm import get_llm_iia
from ed_triage.common.message_sanitize import (
    message_content_to_plain_text,
    sanitize_ai_message_for_checkpoint,
)
from ed_triage.iia.schema import IntakeSummary
from ed_triage.iia.extractor_prompts import EXTRACTOR_SYSTEM_PROMPT
from ed_triage.iia.prompts import IIA_SYSTEM_PROMPT
from ed_triage.iia.state import AgentState

logger = logging.getLogger("IIA-Agent")


def interview_node(state: AgentState):
    """Conversational node; asks questions."""
    logger.debug("Entering 'interviewer' node")
    model = get_llm_iia()
    prompt = ChatPromptTemplate.from_messages([
        ("system", IIA_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ])

    chain = prompt | model
    response = chain.invoke(state)
    plain = message_content_to_plain_text(response.content)
    logger.debug(f"Interviewer response: {plain[:50]}...")
    return {"messages": [sanitize_ai_message_for_checkpoint(response)]}


def extraction_node(state: AgentState):
    """Extracts structured intake data from conversation history when done."""
    logger.debug("Entering 'extractor' node")
    model = get_llm_iia()
    structured_llm = model.with_structured_output(
        IntakeSummary, method="function_calling"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", EXTRACTOR_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ])

    chain = prompt | structured_llm
    intake_data = chain.invoke(state)
    logger.info(
        f"Extraction complete. Emergency: {
            intake_data.emergency_detected}")
    if not intake_data.interview_complete:
        intake_data.interview_complete = True
    return {"intake_data": intake_data}


def check_conversation_end(
        state: AgentState) -> Literal["extractor", "__end__"]:
    """Returns extractor if last message has [CONVERSATION_END], else __end__."""
    messages = state["messages"]
    if not messages:
        return "__end__"

    last_message = messages[-1]
    raw = last_message.content if hasattr(last_message, "content") else ""
    content = message_content_to_plain_text(raw)

    if "[CONVERSATION_END]" in content:
        logger.debug("Termination token detected. Routing to 'extractor'.")
        return "extractor"

    logger.debug("No termination token. Routing to END (awaiting user input).")
    return "__end__"


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("interviewer", interview_node)
    workflow.add_node("extractor", extraction_node)

    workflow.set_entry_point("interviewer")
    workflow.add_conditional_edges(
        "interviewer",
        check_conversation_end,
        {"extractor": "extractor", "__end__": END}
    )
    workflow.add_edge("extractor", END)

    return workflow.compile()
