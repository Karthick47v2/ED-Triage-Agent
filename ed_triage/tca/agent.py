"""TCA: final ESI classification with vitals + physical exam (Phase 2)."""
import logging
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate

from ed_triage.common.llm import get_llm
from ed_triage.tca.schema import TCAResult
from ed_triage.tca.prompts import TCA_SYSTEM_PROMPT
from ed_triage.iia.schema import IntakeSummary
from ed_triage.cra.schema import CRAResult
from ed_triage.common.schemas import VitalSigns, PhysicalExam
from ed_triage.common.vital_assessment import (
    assess_from_vital_signs_schema,
    format_assessment_for_llm,
)

logger = logging.getLogger("TCA-Agent")


def run_tca(
    intake_summary: IntakeSummary,
    cra_result: CRAResult,
    vital_signs: VitalSigns,
    physical_exam: PhysicalExam,
    age_years: Optional[float] = None
) -> TCAResult:
    """Run TCA: deterministic vital assessment + LLM for final ESI classification."""
    logger.info("Running TCA for final ESI classification...")
    vital_assessment = assess_from_vital_signs_schema(vital_signs, age_years)
    vital_assessment_text = format_assessment_for_llm(vital_assessment)
    
    logger.info(f"Vital Assessment: {vital_assessment.overall_status.value} - {len(vital_assessment.high_risk_findings)} high-risk findings")
    llm = get_llm()
    structured_llm = llm.with_structured_output(TCAResult)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", TCA_SYSTEM_PROMPT),
        ("human", """
PATIENT INTAKE SUMMARY:
{intake_data}

CLINICAL REASONING ANALYSIS (CRA):
{cra_result}

{vital_assessment}

PHYSICAL EXAMINATION:
{physical_exam}

The vital signs have been pre-assessed using ESI handbook thresholds (Table 6-1, Figure 6-1, Table 6-2).
Use this deterministic assessment to inform your ESI classification.

Apply the ESI decision algorithm and provide your final classification.
""")
    ])
    
    chain = prompt | structured_llm
    
    result = chain.invoke({
        "intake_data": intake_summary.model_dump_json(indent=2),
        "cra_result": cra_result.model_dump_json(indent=2),
        "vital_assessment": vital_assessment_text,
        "physical_exam": physical_exam.model_dump_json(indent=2)
    })
    
    logger.info(f"TCA Complete. Final ESI: {result.final_esi} (Confidence: {result.confidence:.2f})")
    return result


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    from ed_triage.iia.schema import Symptom
    
    dummy_intake = IntakeSummary(
        chief_complaint="Chest pain",
        hpi=[Symptom(
            name="Chest Pain",
            onset="2 hours ago",
            location="Substernal",
            character="Pressure",
            severity="8/10",
            radiation="Left arm"
        )],
        medical_history=["Hypertension", "Diabetes"],
        emergency_detected=True,
        emergency_reason="Crushing chest pain and radiation to arm"
    )
    
    from ed_triage.cra.schema import DifferentialDiagnosis
    dummy_cra = CRAResult(
        differential_diagnoses=[
            DifferentialDiagnosis(
                condition="Acute Coronary Syndrome",
                likelihood="High",
                reasoning="Classic presentation with risk factors"
            )
        ],
        risk_factors=["Hypertension", "Diabetes", "Classic cardiac symptoms"],
        critical_findings=["Chest pain radiating to arm", "Diaphoresis"],
        esi_handbook_references=["ESI-2: High-risk cardiac presentation"],
        suggested_esi_level=2,
        clinical_explanation="High suspicion for ACS given classic presentation and risk factors"
    )
    
    dummy_vitals = VitalSigns(
        heart_rate_bpm=110,
        respiratory_rate_bpm=22,
        blood_pressure_systolic_mmHg=160,
        blood_pressure_diastolic_mmHg=95,
        oxygen_saturation_percent=96
    )
    
    dummy_exam = PhysicalExam(
        physical_exam="Patient appears diaphoretic and anxious. Skin is pale and moist."
    )
    
    result = run_tca(dummy_intake, dummy_cra, dummy_vitals, dummy_exam)
    print(result.model_dump_json(indent=2))
