from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings

from ed_triage.common.age import age_known, resolve_age_for_vitals
from ed_triage.common.llm import get_llm, required_env
from ed_triage.common.retry import invoke_with_retry
from ed_triage.common.schemas import PhysicalExam, VitalSigns
from ed_triage.common.vital_assessment import assess_from_vital_signs_schema
from ed_triage.cra.prompts import CRA_PHASE2_APPENDIX, CRA_SYSTEM_PROMPT
from ed_triage.cra.schema import CRAResult
from ed_triage.cra.vital_rag import vital_rag_query_boost
from ed_triage.iia.schema import IntakeSummary


logger = logging.getLogger(__name__)

DEFAULT_CHROMA_DIRECTORY = Path(__file__).resolve().parents[1] / "chroma_db"
CHROMA_COLLECTION_NAME = "esi_handbook"
RETRIEVAL_RESULT_COUNT = 5


PHASE1_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CRA_SYSTEM_PROMPT),
        (
            "human",
            """
Patient Intake Data:
{intake_data}

Relevant ESI Guidelines Context:
{context}
""".strip(),
        ),
    ]
)

PHASE2_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", f"{CRA_SYSTEM_PROMPT}\n\n{CRA_PHASE2_APPENDIX}"),
        (
            "human",
            """
Patient Intake Data:
{intake_data}

Vital Signs:
{vital_signs}

Physical Examination:
{physical_exam}

Relevant ESI Guidelines Context:
{context}
""".strip(),
        ),
    ]
)


@lru_cache(maxsize=1)
def get_retriever() -> Any:
    """Create and cache the ESI handbook retriever."""
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=required_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        openai_api_version=required_env("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=required_env("AZURE_OPENAI_ENDPOINT"),
        api_key=required_env("AZURE_OPENAI_API_KEY"),
    )

    chroma_directory = Path(
        os.getenv("CHROMA_DB_DIRECTORY", str(DEFAULT_CHROMA_DIRECTORY))
    ).resolve()

    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=str(chroma_directory),
        embedding_function=embeddings,
    )
    return vector_store.as_retriever(
        search_kwargs={"k": RETRIEVAL_RESULT_COUNT}
    )


@lru_cache(maxsize=1)
def _get_phase1_chain() -> Any:
    """Create and cache the Phase 1 runnable chain."""
    structured_llm = get_llm().with_structured_output(
        CRAResult,
        method="function_calling",
    )
    return PHASE1_PROMPT | structured_llm


@lru_cache(maxsize=1)
def _get_phase2_chain() -> Any:
    """Create and cache the Phase 2 runnable chain."""
    structured_llm = get_llm().with_structured_output(
        CRAResult,
        method="function_calling",
    )
    return PHASE2_PROMPT | structured_llm


def _build_rag_query(intake_summary: IntakeSummary) -> str:
    """Build a retrieval query from the complaint and HPI."""
    parts = [
        intake_summary.chief_complaint,
        *(symptom.name for symptom in intake_summary.hpi),
    ]
    return " ".join(
        str(part).strip()
        for part in parts
        if part and str(part).strip()
    )


def _build_rag_context(query: str) -> str:
    """Retrieve and format relevant ESI handbook passages."""
    documents = get_retriever().invoke(query)
    passages: list[str] = []

    for document in documents:
        content = document.page_content.strip()
        if not content:
            continue

        page = document.metadata.get("page")
        source = document.metadata.get("source", "ESI handbook")
        citation = (
            f"{source}, page {page}" if page is not None else str(source)
        )
        passages.append(f"[{citation}]\n{content}")

    return "\n\n".join(passages)


def _invoke_cra(
    *,
    chain: Any,
    payload: dict[str, str],
    label: str,
) -> CRAResult:
    """Invoke a CRA chain and verify its structured result."""
    result = invoke_with_retry(chain, payload, label=label)
    if not isinstance(result, CRAResult):
        raise TypeError(
            f"{label} returned {type(result).__name__}; expected CRAResult."
        )
    return result


def run_cra(intake_summary: IntakeSummary) -> CRAResult:
    """Run CRA Phase 1 using intake symptoms."""
    logger.info(
        "Running CRA Phase 1 for complaint: %s",
        intake_summary.chief_complaint,
    )

    query = _build_rag_query(intake_summary)
    logger.debug("CRA Phase 1 RAG query: %s", query)

    result = _invoke_cra(
        chain=_get_phase1_chain(),
        payload={
            "intake_data": intake_summary.model_dump_json(),
            "context": _build_rag_context(query),
        },
        label="CRA Phase 1",
    )
    logger.info(
        "CRA Phase 1 complete. Suggested ESI: %s; differentials: %d; "
        "critical findings: %d",
        result.suggested_esi_level,
        len(result.differential_diagnoses),
        len(result.critical_findings),
    )
    return result


def run_cra_phase2(
    intake_summary: IntakeSummary,
    vital_signs: VitalSigns,
    physical_exam: PhysicalExam,
    *,
    age_years: float | None = None,
    age_months: float | None = None,
    age_days: int | None = None,
) -> CRAResult:
    """Run CRA Phase 2 using intake, vitals, and physical examination."""
    logger.info(
        "Running CRA Phase 2 for complaint: %s",
        intake_summary.chief_complaint,
    )

    age = resolve_age_for_vitals(
        intake_summary,
        age_years=age_years,
        age_months=age_months,
        age_days=age_days,
    )
    if not age_known(age):
        logger.warning(
            "Patient age is unavailable; adult vital thresholds will be used."
        )

    resolved_years, resolved_months, resolved_days = age
    vital_assessment = assess_from_vital_signs_schema(
        vital_signs,
        age_years=resolved_years,
        age_months=resolved_months,
        age_days=resolved_days,
    )

    query = _build_rag_query(intake_summary) + vital_rag_query_boost(
        vital_assessment
    )
    logger.debug("CRA Phase 2 RAG query: %s", query)

    result = _invoke_cra(
        chain=_get_phase2_chain(),
        payload={
            "intake_data": intake_summary.model_dump_json(),
            "vital_signs": vital_signs.model_dump_json(),
            "physical_exam": physical_exam.model_dump_json(),
            "context": _build_rag_context(query),
        },
        label="CRA Phase 2",
    )
    logger.info("CRA Phase 2 complete")
    return result
