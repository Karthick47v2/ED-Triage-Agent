"""Shared age-field coalescing and intake-override helpers."""

from __future__ import annotations

from typing import Protocol, TypeAlias


AgeTriple: TypeAlias = tuple[float | None, float | None, int | None]


class HasAgeFields(Protocol):
    """Minimal age-field surface for intake / scenario objects."""

    age_years: float | None
    age_months: float | None
    age_days: int | None


def coalesce_age_fields(
    preferred: AgeTriple,
    fallback: AgeTriple,
) -> AgeTriple:
    """Use preferred age fields when present, otherwise use fallbacks."""
    py, pm, pd = preferred
    fy, fm, fd = fallback

    return (
        py if py is not None else fy,
        pm if pm is not None else fm,
        pd if pd is not None else fd,
    )


def resolve_age_for_vitals(
    intake_summary: HasAgeFields,
    *,
    age_years: float | None = None,
    age_months: float | None = None,
    age_days: int | None = None,
) -> AgeTriple:
    """Explicit run kwargs override intake when not None."""
    return coalesce_age_fields(
        (age_years, age_months, age_days),
        (intake_summary.age_years, intake_summary.age_months, intake_summary.age_days),
    )


def age_known(triple: AgeTriple) -> bool:
    """Return whether any additive age component is present."""
    y, m, d = triple
    return y is not None or m is not None or d is not None
