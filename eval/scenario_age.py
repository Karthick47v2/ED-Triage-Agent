"""Merge intake-extracted age with optional ETEK scenario structured age (Phase 2)."""
from __future__ import annotations

from ed_triage.common.age import AgeTriple, coalesce_age_fields
from eval.schemas import EvaluationScenario


def merge_intake_and_scenario_age(
    intake_years: float | None,
    intake_months: float | None,
    intake_days: int | None,
    scenario: EvaluationScenario | None = None,
) -> AgeTriple:
    """Per-field: scenario structured age wins when set; else intake."""
    if scenario is None:
        return intake_years, intake_months, intake_days
    return coalesce_age_fields(
        (scenario.age_years, scenario.age_months, scenario.age_days),
        (intake_years, intake_months, intake_days),
    )
