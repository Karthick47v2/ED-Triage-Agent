"""Resolve patient age for vital sign thresholds (intake + optional overrides)."""
from typing import Optional, Tuple

from ed_triage.iia.schema import IntakeSummary


def resolve_age_for_vitals(
    intake_summary: IntakeSummary,
    age_years: Optional[float] = None,
    age_months: Optional[float] = None,
    age_days: Optional[int] = None,
) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    """
    Per-field: explicit run_tca / Phase2 kwargs override intake when not None.
    """
    y = age_years if age_years is not None else intake_summary.age_years
    m = age_months if age_months is not None else intake_summary.age_months
    d = age_days if age_days is not None else intake_summary.age_days
    return (y, m, d)


def age_known(
        y: Optional[float],
        m: Optional[float],
        d: Optional[int]) -> bool:
    return y is not None or m is not None or d is not None
