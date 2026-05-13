import os
import logging
from functools import lru_cache

from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from ed_triage.common.llm import get_llm
from ed_triage.common.retry import invoke_with_retry
from ed_triage.cra.schema import CRAResult
from ed_triage.cra.prompts import CRA_PHASE2_APPENDIX, CRA_SYSTEM_PROMPT
from ed_triage.iia.schema import IntakeSummary
from ed_triage.common.schemas import PhysicalExam, VitalSigns

logger = logging.getLogger("CRA-Agent")


@lru_cache(maxsize=1)
def get_retriever():
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    )
    vector_store = Chroma(
        persist_directory="ed_triage/chroma_db",
        embedding_function=embeddings,
        collection_name="esi_handbook"
    )
    return vector_store.as_retriever(search_kwargs={"k": 5})


def _build_rag_context(query: str) -> str:
    """Retrieve ESI handbook passages for the given query."""
    docs = get_retriever().invoke(query)
    return "\n\n".join(
        [f"[Page {d.metadata.get('page')}]: {d.page_content}" for d in docs])


def run_cra(intake_summary: IntakeSummary) -> CRAResult:
    """Run CRA (Phase 1) on intake summary — symptoms only."""
    logger.info("Running CRA (Phase 1) for complaint: %s", intake_summary.chief_complaint)
    query = f"{intake_summary.chief_complaint} {' '.join([s.name for s in intake_summary.hpi])}"
    logger.debug("RAG Query: %s", query)
    context = _build_rag_context(query)

    structured_llm = get_llm().with_structured_output(CRAResult, method="function_calling")
    prompt = ChatPromptTemplate.from_messages([
        ("system", CRA_SYSTEM_PROMPT),
        ("human", "Patient Intake Data:\n{intake_data}\n\nRelevant ESI Guidelines Context:\n{context}")
    ])
    result = invoke_with_retry(prompt | structured_llm, {
        "intake_data": intake_summary.model_dump_json(indent=2),
        "context": context,
    }, label="CRA Phase 1")
    logger.info("CRA (Phase 1) complete.")
    return result


def run_cra_phase2(
    intake_summary: IntakeSummary,
    vital_signs: VitalSigns,
    physical_exam: PhysicalExam
) -> CRAResult:
    """Run CRA (Phase 2) integrating vitals and physical exam."""
    logger.info("Running CRA (Phase 2) for complaint: %s", intake_summary.chief_complaint)
    symptoms = ' '.join([s.name for s in intake_summary.hpi])
    query = f"{intake_summary.chief_complaint} {symptoms}"
    if vital_signs.heart_rate_bpm and (
            vital_signs.heart_rate_bpm > 100 or vital_signs.heart_rate_bpm < 60):
        query += " tachycardia bradycardia"
    if vital_signs.oxygen_saturation_percent and vital_signs.oxygen_saturation_percent < 94:
        query += " hypoxia respiratory distress"
    if vital_signs.blood_pressure_systolic_mmHg and vital_signs.blood_pressure_systolic_mmHg < 90:
        query += " hypotension shock"

    logger.debug("RAG Query (Phase 2): %s", query)
    context = _build_rag_context(query)

    structured_llm = get_llm().with_structured_output(CRAResult, method="function_calling")
    prompt = ChatPromptTemplate.from_messages([
        ("system", CRA_SYSTEM_PROMPT + CRA_PHASE2_APPENDIX),
        ("human", """Patient Intake Data:
{intake_data}

VITAL SIGNS:
{vital_signs}

PHYSICAL EXAMINATION:
{physical_exam}

Relevant ESI Guidelines Context:
{context}""")
    ])
    result = invoke_with_retry(prompt | structured_llm, {
        "intake_data": intake_summary.model_dump_json(indent=2),
        "vital_signs": vital_signs.model_dump_json(indent=2),
        "physical_exam": physical_exam.model_dump_json(indent=2),
        "context": context,
    }, label="CRA Phase 2")
    logger.info("CRA (Phase 2) complete.")
    return result
