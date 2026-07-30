"""Age-group resolution for ESI vital-sign thresholds."""

from __future__ import annotations

from math import isfinite
from numbers import Real

from ed_triage.common.vital_assessment.tables import AgeGroup


_AVERAGE_DAYS_PER_MONTH = 30.437
_AVERAGE_MONTHS_PER_YEAR = 12

# Keep conversions consistent so 1 year equals 12 months exactly.
_AVERAGE_DAYS_PER_YEAR = (
    _AVERAGE_DAYS_PER_MONTH * _AVERAGE_MONTHS_PER_YEAR
)


def _validate_age_components(
    age_years: float | None,
    age_months: float | None,
    age_days: int | None,
) -> None:
    for name, value in (
        ("age_years", age_years),
        ("age_months", age_months),
        ("age_days", age_days),
    ):
        if value is None:
            continue
        # bool is a subclass of int; reject it as an age component.
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number or None")
        if not isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
        if value < 0:
            raise ValueError(f"{name} cannot be negative")


def resolve_age_in_days(
    *,
    age_years: float | None = None,
    age_months: float | None = None,
    age_days: int | None = None,
) -> float | None:
    """Combine additive age components into an approximate age in days.

    For example, one year and three months should be supplied as
    ``age_years=1, age_months=3``.

    Returns:
        Approximate age in days, or ``None`` if no components are supplied.
    """
    _validate_age_components(age_years, age_months, age_days)

    if age_years is None and age_months is None and age_days is None:
        return None

    return (
        (age_years or 0) * _AVERAGE_DAYS_PER_YEAR
        + (age_months or 0) * _AVERAGE_DAYS_PER_MONTH
        + (age_days or 0)
    )


def determine_age_group_from_days(
    total_days: float | None,
) -> AgeGroup:
    """Map resolved age in days to an ESI age group.

    Unknown age defaults to ``ADULT``.
    """
    if total_days is None:
        return AgeGroup.ADULT
    if not isfinite(float(total_days)):
        raise ValueError("total_days must be finite")
    if total_days < 0:
        raise ValueError("total_days cannot be negative")

    total_months = total_days / _AVERAGE_DAYS_PER_MONTH

    if total_months < 1:
        return AgeGroup.NEONATE
    if total_months < 12:
        return AgeGroup.INFANT
    if total_months < 36:
        return AgeGroup.TODDLER
    if total_months < 60:
        return AgeGroup.PRESCHOOL
    if total_months < 144:
        return AgeGroup.SCHOOL_AGE
    if total_months < 216:
        return AgeGroup.ADOLESCENT
    return AgeGroup.ADULT


def determine_age_group(
    *,
    age_years: float | None = None,
    age_months: float | None = None,
    age_days: int | None = None,
) -> AgeGroup:
    """Map additive age components to an ESI age group.

    Uses the same day conversion as ``resolve_age_in_days``. Unknown age
    defaults to ``ADULT``.
    """
    return determine_age_group_from_days(
        resolve_age_in_days(
            age_years=age_years,
            age_months=age_months,
            age_days=age_days,
        )
    )
