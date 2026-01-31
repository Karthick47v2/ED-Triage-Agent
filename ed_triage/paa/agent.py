import logging
from langchain_core.prompts import ChatPromptTemplate

from ed_triage.common.llm import get_llm
from ed_triage.paa.schema import PriorityAssessment
from ed_triage.paa.prompts import PAA_SYSTEM_PROMPT
from ed_triage.iia.schema import IntakeSummary
from ed_triage.cra.schema import CRAResult

logger = logging.getLogger("PAA-Agent")

def run_paa(intake_summary: IntakeSummary, cra_result: CRAResult) -> PriorityAssessment:
    """Run PAA for queue priority from intake + CRA result."""
    logger.info("Running PAA to determine queue priority...")
    
    llm = get_llm()
    structured_llm = llm.with_structured_output(PriorityAssessment)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", PAA_SYSTEM_PROMPT),
        ("human", """
        PATIENT INTAKE SUMMARY:
        {intake_data}
        
        CLINICAL REASONING ANALYSIS:
        {cra_result}
        """)
    ])
    
    chain = prompt | structured_llm
    
    result = chain.invoke({
        "intake_data": intake_summary.model_dump_json(),
        "cra_result": cra_result.model_dump_json()
    })
    
    logger.info(f"PAA Complete. Priority: {result.priority_score} (ESI ~{result.tentative_esi})")
    return result
