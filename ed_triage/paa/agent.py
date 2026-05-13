import logging
from langchain_core.prompts import ChatPromptTemplate

from ed_triage.common.llm import get_llm
from ed_triage.common.retry import invoke_with_retry
from ed_triage.paa.schema import PriorityAssessment
from ed_triage.paa.prompts import PAA_SYSTEM_PROMPT
from ed_triage.iia.schema import IntakeSummary
from ed_triage.cra.schema import CRAResult

logger = logging.getLogger("PAA-Agent")


def run_paa(intake_summary: IntakeSummary, cra_result: CRAResult) -> PriorityAssessment:
    """Run PAA — queue priority from intake + CRA result."""
    logger.info("Running PAA to determine queue priority...")

    structured_llm = get_llm().with_structured_output(
        PriorityAssessment, method="function_calling"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", PAA_SYSTEM_PROMPT),
        ("human", """PATIENT INTAKE SUMMARY:
{intake_data}

CLINICAL REASONING ANALYSIS:
{cra_result}""")
    ])

    result = invoke_with_retry(prompt | structured_llm, {
        "intake_data": intake_summary.model_dump_json(),
        "cra_result": cra_result.model_dump_json(),
    }, label="PAA")

    logger.info("PAA complete. Priority: %s (ESI ~%s)", result.priority_score, result.tentative_esi)
    return result
