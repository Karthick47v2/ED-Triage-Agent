"""Merge intake-extracted age with optional ETEK scenario structured age (Phase 2)."""
from __future__ import annotations

from typing import Optional, Tuple

from eval.schemas import EvaluationScenario


def merge_intake_and_scenario_age(
    intake_years: Optional[float],
    intake_months: Optional[float],
    intake_days: Optional[int],
    scenario: Optional[EvaluationScenario] = None,
) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    """Return (age_years, age_months, age_days) for TCA vital thresholds.

    Per-field: when ``scenario`` has a non-null structured age field, it wins;
    otherwise the intake value is used. If ``scenario`` is ``None``, intake
    values are returned unchanged.
    """
    if scenario is None:
        return intake_years, intake_months, intake_days

    y = scenario.age_years if scenario.age_years is not None else intake_years
    m = scenario.age_months if scenario.age_months is not None else intake_months
    d = scenario.age_days if scenario.age_days is not None else intake_days
    return y, m, d
