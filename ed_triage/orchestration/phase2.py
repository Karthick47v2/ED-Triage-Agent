"""Phase 2 orchestration: CRA phase2 -> TCA."""

import logging
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from ed_triage.cra.agent import run_cra_phase2
from ed_triage.cra.schema import CRAResult
from ed_triage.common.schemas import PhysicalExam, VitalSigns
from ed_triage.iia.schema import IntakeSummary
from ed_triage.tca.agent import run_tca
from ed_triage.tca.schema import TCAResult

logger = logging.getLogger("Phase2-Orchestrator")


class Phase2State(TypedDict):
    intake_data: IntakeSummary
    vital_signs: VitalSigns
    physical_exam: PhysicalExam
    phase1_cra_result: Optional[CRAResult]
    cra_result: Optional[CRAResult]
    tca_result: Optional[TCAResult]
    patient_age_years: Optional[float]
    patient_age_months: Optional[float]
    patient_age_days: Optional[int]


def cra_phase2_node(state: Phase2State) -> dict:
    logger.info("Running CRA Phase 2 node...")
    return {
        "cra_result": run_cra_phase2(
            state["intake_data"], state["vital_signs"], state["physical_exam"]
        )
    }


def tca_node(state: Phase2State) -> dict:
    logger.info("Running TCA node...")
    return {
        "tca_result": run_tca(
            state["intake_data"],
            state["cra_result"],
            state["vital_signs"],
            state["physical_exam"],
            age_years=state["patient_age_years"],
            age_months=state["patient_age_months"],
            age_days=state["patient_age_days"],
        )
    }


def build_phase2_graph():
    workflow = StateGraph(Phase2State)
    workflow.add_node("cra_phase2", cra_phase2_node)
    workflow.add_node("tca", tca_node)
    workflow.set_entry_point("cra_phase2")
    workflow.add_edge("cra_phase2", "tca")
    workflow.add_edge("tca", END)
    return workflow.compile()


def run_phase2_pipeline(
    intake_data: IntakeSummary,
    vital_signs: VitalSigns,
    physical_exam: PhysicalExam,
    phase1_cra_result: Optional[CRAResult] = None,
    age_years: Optional[float] = None,
    age_months: Optional[float] = None,
    age_days: Optional[int] = None,
) -> TCAResult:
    graph = build_phase2_graph()
    final_state = graph.invoke(
        {
            "intake_data": intake_data,
            "vital_signs": vital_signs,
            "physical_exam": physical_exam,
            "phase1_cra_result": phase1_cra_result,
            "cra_result": None,
            "tca_result": None,
            "patient_age_years": age_years,
            "patient_age_months": age_months,
            "patient_age_days": age_days,
        }
    )
    return final_state["tca_result"]
