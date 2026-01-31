import os
import logging
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from ed_triage.common.llm import get_llm
from ed_triage.cra.schema import CRAResult
from ed_triage.cra.prompts import CRA_SYSTEM_PROMPT, CRA_PHASE2_APPENDIX
from ed_triage.iia.schema import IntakeSummary
from ed_triage.common.schemas import VitalSigns, PhysicalExam

logger = logging.getLogger("CRA-Agent")

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

def run_cra(intake_summary: IntakeSummary) -> CRAResult:
    """Run CRA (Phase 1) on intake summary (symptoms only)."""
    logger.info(f"Running CRA (Phase 1) for complaint: {intake_summary.chief_complaint}")
    retriever = get_retriever()
    query = f"{intake_summary.chief_complaint} {' '.join([s.name for s in intake_summary.hpi])}"
    logger.debug(f"RAG Query: {query}")
    docs = retriever.invoke(query)
    context = "\n\n".join([f"[Page {d.metadata.get('page')}]: {d.page_content}" for d in docs])
    llm = get_llm()
    structured_llm = llm.with_structured_output(CRAResult)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", CRA_SYSTEM_PROMPT),
        ("human", "Patient Intake Data:\n{intake_data}\n\nRelevant ESI Guidelines Context:\n{context}")
    ])
    
    chain = prompt | structured_llm
    result = chain.invoke({
        "intake_data": intake_summary.model_dump_json(indent=2),
        "context": context
    })
    logger.info("CRA (Phase 1) Analysis complete.")
    return result


def run_cra_phase2(
    intake_summary: IntakeSummary,
    vital_signs: VitalSigns,
    physical_exam: PhysicalExam
) -> CRAResult:
    """Run CRA (Phase 2) with vitals and physical exam."""
    logger.info(f"Running CRA (Phase 2) for complaint: {intake_summary.chief_complaint}")
    retriever = get_retriever()
    symptoms = ' '.join([s.name for s in intake_summary.hpi])
    query = f"{intake_summary.chief_complaint} {symptoms}"
    if vital_signs.heart_rate_bpm and (vital_signs.heart_rate_bpm > 100 or vital_signs.heart_rate_bpm < 60):
        query += " tachycardia bradycardia"
    if vital_signs.oxygen_saturation_percent and vital_signs.oxygen_saturation_percent < 94:
        query += " hypoxia respiratory distress"
    if vital_signs.blood_pressure_systolic_mmHg and vital_signs.blood_pressure_systolic_mmHg < 90:
        query += " hypotension shock"
        
    logger.debug(f"RAG Query (Phase 2): {query}")
    docs = retriever.invoke(query)
    context = "\n\n".join([f"[Page {d.metadata.get('page')}]: {d.page_content}" for d in docs])
    llm = get_llm()
    structured_llm = llm.with_structured_output(CRAResult)
    
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
    
    chain = prompt | structured_llm
    result = chain.invoke({
        "intake_data": intake_summary.model_dump_json(indent=2),
        "vital_signs": vital_signs.model_dump_json(indent=2),
        "physical_exam": physical_exam.model_dump_json(indent=2),
        "context": context
    })
    
    logger.info("CRA (Phase 2) Analysis complete.")
    return result

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    dummy_intake = IntakeSummary(
        chief_complaint="Chest pain",
        hpi=[{
            "name": "Chest Pain",
            "onset": "2 hours ago",
            "location": "Substernal",
            "character": "Pressure",
            "severity": "8/10",
            "radiation": "Left arm"
        }],
        medical_history=["Hypertension", "Diabetes"],
        emergency_detected=True,
        emergency_reason="Crushing chest pain and radiation to arm"
    )
    
    res = run_cra(dummy_intake)
    print(res.model_dump_json(indent=2))
