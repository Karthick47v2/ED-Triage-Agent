"""TCA: final ESI classification with vitals + physical exam (Phase 2)."""
from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate

from ed_triage.common.llm import get_llm
from ed_triage.common.retry import invoke_with_retry
from ed_triage.common.schemas import PhysicalExam, VitalSigns
from ed_triage.common.vital_assessment import (
    assess_from_vital_signs_schema,
    format_assessment_for_llm,
)
from ed_triage.cra.schema import CRAResult
from ed_triage.iia.schema import IntakeSummary
from ed_triage.common.age import age_known, resolve_age_for_vitals
from ed_triage.tca.missing_vitals_prompt import format_missing_vitals_footer
from ed_triage.tca.prompts import TCA_SYSTEM_PROMPT
from ed_triage.tca.schema import TCAResult

logger = logging.getLogger("TCA-Agent")


def run_tca(
    intake_summary: IntakeSummary,
    cra_result: CRAResult,
    vital_signs: VitalSigns,
    physical_exam: PhysicalExam,
    *,
    age_years: float | None = None,
    age_months: float | None = None,
    age_days: int | None = None,
) -> TCAResult:
    """Run TCA: deterministic vital assessment + LLM for final ESI classification."""
    logger.info("Running TCA for final ESI classification...")

    age = resolve_age_for_vitals(
        intake_summary, age_years=age_years, age_months=age_months, age_days=age_days
    )
    if not age_known(age):
        logger.info("Patient age not provided; using adult default thresholds.")

    ry, rm, rd = age
    vital_assessment = assess_from_vital_signs_schema(
        vital_signs, age_years=ry, age_months=rm, age_days=rd
    )
    vital_assessment_text = format_assessment_for_llm(vital_assessment)
    vital_assessment_text += format_missing_vitals_footer(vital_signs)

    logger.info(
        "Vital Assessment: %s — %d critical, %d high-risk findings",
        vital_assessment.overall_status.value,
        len(vital_assessment.critical_findings),
        len(vital_assessment.high_risk_findings),
    )

    structured_llm = get_llm().with_structured_output(TCAResult, method="function_calling")
    prompt = ChatPromptTemplate.from_messages([
        ("system", TCA_SYSTEM_PROMPT),
        ("human", """PATIENT INTAKE SUMMARY:
{intake_data}

CLINICAL REASONING ANALYSIS (CRA):
{cra_result}

{vital_assessment}

PHYSICAL EXAMINATION:
{physical_exam}

The vital signs have been pre-assessed using ESI handbook thresholds (Table 6-1, Figure 6-1, Table 6-2).
Statuses include NORMAL, ABNORMAL, HIGH_RISK, CRITICAL, UNKNOWN (not recorded), and NOT_ASSESSED (recorded but not threshold-classified).
Use the deterministic assessment and its recommendation text as inputs to the ESI steps above — not as automatic ESI assignments.

Apply the ESI decision algorithm and provide your final classification.""")
    ])

    result = invoke_with_retry(prompt | structured_llm, {
        "intake_data": intake_summary.model_dump_json(indent=2),
        "cra_result": cra_result.model_dump_json(indent=2),
        "vital_assessment": vital_assessment_text,
        "physical_exam": physical_exam.model_dump_json(indent=2),
    }, label="TCA")

    logger.info("TCA complete. Final ESI: %s (Confidence: %.2f)", result.final_esi, result.confidence)
    return result
