"""Phase 2 graph: CRA (Phase 2) → TCA → END (post-vital classification)."""
import logging
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, END

from ed_triage.iia.schema import IntakeSummary
from ed_triage.cra.schema import CRAResult
from ed_triage.cra.agent import run_cra_phase2
from ed_triage.tca.schema import TCAResult
from ed_triage.tca.agent import run_tca
from ed_triage.common.schemas import VitalSigns, PhysicalExam

logger = logging.getLogger("Phase2-Orchestrator")


class Phase2State(TypedDict):
    intake_data: IntakeSummary
    vital_signs: VitalSigns
    physical_exam: PhysicalExam
    phase1_cra_result: Optional[CRAResult]
    cra_result: Optional[CRAResult]
    tca_result: Optional[TCAResult]


def cra_phase2_node(state: Phase2State) -> dict:
    logger.info("Running CRA Phase 2 node...")
    
    intake_data = state["intake_data"]
    vital_signs = state["vital_signs"]
    physical_exam = state["physical_exam"]
    
    result = run_cra_phase2(intake_data, vital_signs, physical_exam)
    return {"cra_result": result}


def tca_node(state: Phase2State) -> dict:
    logger.info("Running TCA node...")
    
    intake_data = state["intake_data"]
    cra_result = state["cra_result"]
    vital_signs = state["vital_signs"]
    physical_exam = state["physical_exam"]
    
    result = run_tca(intake_data, cra_result, vital_signs, physical_exam)
    return {"tca_result": result}


def build_phase2_graph():
    workflow = StateGraph(Phase2State)
    workflow.add_node("cra_phase2", cra_phase2_node)
    workflow.add_node("tca", tca_node)
    workflow.set_entry_point("cra_phase2")
    workflow.add_edge("cra_phase2", "tca")
    workflow.add_edge("tca", END)
    
    return workflow.compile()


def run_phase2(
    intake_data: IntakeSummary,
    vital_signs: VitalSigns,
    physical_exam: PhysicalExam,
    phase1_cra_result: Optional[CRAResult] = None
) -> TCAResult:
    """Run Phase 2 pipeline (CRA Phase 2 → TCA)."""
    graph = build_phase2_graph()
    
    initial_state = {
        "intake_data": intake_data,
        "vital_signs": vital_signs,
        "physical_exam": physical_exam,
        "phase1_cra_result": phase1_cra_result,
        "cra_result": None,
        "tca_result": None
    }
    
    final_state = graph.invoke(initial_state)
    return final_state["tca_result"]


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    from ed_triage.iia.schema import Symptom
    from ed_triage.cra.schema import DifferentialDiagnosis
    dummy_intake = IntakeSummary(
        chief_complaint="Shortness of breath",
        hpi=[Symptom(
            name="Dyspnea",
            onset="3 hours ago",
            character="Progressive",
            severity="7/10"
        )],
        medical_history=["COPD", "CHF"],
        emergency_detected=False
    )
    
    dummy_vitals = VitalSigns(
        heart_rate_bpm=112,
        respiratory_rate_bpm=28,
        blood_pressure_systolic_mmHg=145,
        blood_pressure_diastolic_mmHg=88,
        oxygen_saturation_percent=89
    )
    
    dummy_exam = PhysicalExam(
        physical_exam="Patient sitting upright, using accessory muscles. Skin is pale and diaphoretic. Speaking in 2-3 word sentences."
    )
    
    result = run_phase2(dummy_intake, dummy_vitals, dummy_exam)
    print(f"\nFinal ESI: {result.final_esi}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Rationale: {result.rationale}")
