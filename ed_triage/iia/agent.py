import logging
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ed_triage.common.llm import get_llm_iia
from ed_triage.common.message_sanitize import (
    message_content_to_plain_text,
    sanitize_ai_message_for_checkpoint,
)
from ed_triage.common.retry import invoke_with_retry
from ed_triage.iia.extractor_prompts import EXTRACTOR_SYSTEM_PROMPT
from ed_triage.iia.prompts import IIA_SYSTEM_PROMPT
from ed_triage.iia.schema import IntakeSummary
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
    logger.debug("Interviewer response: %s...", plain[:50])
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
    intake_data = invoke_with_retry(chain, state, label="IIA Extractor")
    logger.info(
        "Extraction complete. Emergency: %s",
        intake_data.emergency_detected,
    )
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
    content = message_content_to_plain_text(
        getattr(last_message, "content", None)
    )

    if "[CONVERSATION_END]" in content:
        logger.debug("Termination token detected. Routing to 'extractor'.")
        return "extractor"

    logger.debug("No termination token. Routing to END (awaiting user input).")
    return "__end__"
