"""Phase 2 orchestration: CRA Phase 2 -> TCA."""

from __future__ import annotations

from ed_triage.common.schemas import PhysicalExam, VitalSigns
from ed_triage.cra.agent import run_cra_phase2
from ed_triage.iia.schema import IntakeSummary
from ed_triage.tca.agent import run_tca
from ed_triage.tca.schema import TCAResult


def run_phase2_pipeline(
    intake_summary: IntakeSummary,
    vital_signs: VitalSigns,
    physical_exam: PhysicalExam,
    *,
    age_years: float | None = None,
    age_months: float | None = None,
    age_days: int | None = None,
) -> TCAResult:
    """Run CRA Phase 2 followed by TCA."""

    cra_result = run_cra_phase2(
        intake_summary=intake_summary,
        vital_signs=vital_signs,
        physical_exam=physical_exam,
        age_years=age_years,
        age_months=age_months,
        age_days=age_days,
    )

    return run_tca(
        intake_summary=intake_summary,
        cra_result=cra_result,
        vital_signs=vital_signs,
        physical_exam=physical_exam,
        age_years=age_years,
        age_months=age_months,
        age_days=age_days,
    )
