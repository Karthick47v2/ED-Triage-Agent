"""Types and ESI handbook vital-sign thresholds."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Final


class VitalStatus(str, Enum):
    """Classification assigned to an individual or aggregate vital assessment."""

    NORMAL = "normal"
    ABNORMAL = "abnormal"
    HIGH_RISK = "high_risk"
    CRITICAL = "critical"
    UNKNOWN = "unknown"  # Not recorded or missing.
    NOT_ASSESSED = "not_assessed"  # Recorded without a configured range.


# Statuses that mean a vital was missing or could not be fully assessed.
GAP_STATUSES: Final[frozenset[VitalStatus]] = frozenset({
    VitalStatus.UNKNOWN,
    VitalStatus.NOT_ASSESSED,
})


@dataclass(slots=True)
class VitalAssessment:
    """Assessment result for one vital sign."""

    name: str
    value: int | float | None
    unit: str
    status: VitalStatus
    normal_range: str
    high_risk_threshold: str
    reasoning: str


@dataclass(slots=True)
class VitalSignsAssessment:
    """Individual and aggregate vital-sign assessment results.

    ``has_high_risk_vitals`` is true only when the HIGH_RISK findings bucket is
    non-empty. A CRITICAL-only result sets ``has_critical_vitals`` and leaves
    ``has_high_risk_vitals`` false (critical findings are not dual-counted).
    """

    heart_rate: VitalAssessment
    respiratory_rate: VitalAssessment
    blood_pressure_systolic: VitalAssessment
    blood_pressure_diastolic: VitalAssessment
    oxygen_saturation: VitalAssessment
    temperature: VitalAssessment

    has_high_risk_vitals: bool
    has_critical_vitals: bool
    high_risk_findings: list[str]
    critical_findings: list[str]
    abnormal_findings: list[str]
    overall_status: VitalStatus
    esi_recommendation: str

    @property
    def vitals(self) -> tuple[VitalAssessment, ...]:
        """Return individual vital assessments in display order."""
        return (
            self.heart_rate,
            self.respiratory_rate,
            self.blood_pressure_systolic,
            self.blood_pressure_diastolic,
            self.oxygen_saturation,
            self.temperature,
        )

    @property
    def has_incomplete_vitals(self) -> bool:
        """Return whether any vital is missing or not fully assessed."""
        return any(vital.status in GAP_STATUSES for vital in self.vitals)


class AgeGroup(str, Enum):
    """Age groups used by ESI vital-sign thresholds."""

    NEONATE = "neonate"          # <1 month
    INFANT = "infant"            # 1–12 months
    TODDLER = "toddler"          # 1–3 years
    PRESCHOOL = "preschool"      # 3–5 years
    SCHOOL_AGE = "school_age"    # 5–12 years
    ADOLESCENT = "adolescent"    # 12–18 years
    ADULT = "adult"              # Adult age range


@dataclass(frozen=True, slots=True)
class AgeVitalRanges:
    """ESI Handbook Table 6-1 normal vital ranges for an age group."""

    heart_rate_min: int
    heart_rate_max: int
    respiratory_rate_min: int
    respiratory_rate_max: int
    systolic_bp_min: int
    systolic_bp_max: int

    def __post_init__(self) -> None:
        _validate_range(
            "heart rate",
            self.heart_rate_min,
            self.heart_rate_max,
        )
        _validate_range(
            "respiratory rate",
            self.respiratory_rate_min,
            self.respiratory_rate_max,
        )
        _validate_range(
            "systolic blood pressure",
            self.systolic_bp_min,
            self.systolic_bp_max,
        )


@dataclass(frozen=True, slots=True)
class AgeHighRiskThresholds:
    """ESI Figure 6-1 high-risk HR and RR upper thresholds."""

    heart_rate_above: int
    respiratory_rate_above: int

    def __post_init__(self) -> None:
        if self.heart_rate_above < 0:
            raise ValueError("heart_rate_above cannot be negative")
        if self.respiratory_rate_above < 0:
            raise ValueError("respiratory_rate_above cannot be negative")


def _validate_range(name: str, minimum: int, maximum: int) -> None:
    """Validate an inclusive, non-negative reference range."""
    if minimum < 0:
        raise ValueError(f"{name} minimum cannot be negative")
    if maximum < minimum:
        raise ValueError(
            f"{name} maximum ({maximum}) cannot be below minimum ({minimum})"
        )


# ESI Handbook Table 6-1.
# High-side HIGH_RISK_VITALS_BY_AGE HR/RR cutoffs equal these normal maxima, so
# there is no separate mild ABNORMAL band for tachycardia/tachypnea (only the
# low side can be ABNORMAL without reaching HIGH_RISK).
NORMAL_VITALS_BY_AGE: Final[Mapping[AgeGroup, AgeVitalRanges]] = (
    MappingProxyType({
        AgeGroup.NEONATE: AgeVitalRanges(
            heart_rate_min=90,
            heart_rate_max=190,
            respiratory_rate_min=35,
            respiratory_rate_max=60,
            systolic_bp_min=67,
            systolic_bp_max=84,
        ),
        AgeGroup.INFANT: AgeVitalRanges(
            heart_rate_min=90,
            heart_rate_max=180,
            respiratory_rate_min=30,
            respiratory_rate_max=55,
            systolic_bp_min=72,
            systolic_bp_max=104,
        ),
        AgeGroup.TODDLER: AgeVitalRanges(
            heart_rate_min=80,
            heart_rate_max=140,
            respiratory_rate_min=22,
            respiratory_rate_max=40,
            systolic_bp_min=86,
            systolic_bp_max=104,
        ),
        AgeGroup.PRESCHOOL: AgeVitalRanges(
            heart_rate_min=65,
            heart_rate_max=120,
            respiratory_rate_min=18,
            respiratory_rate_max=35,
            systolic_bp_min=89,
            systolic_bp_max=112,
        ),
        AgeGroup.SCHOOL_AGE: AgeVitalRanges(
            heart_rate_min=70,
            heart_rate_max=120,
            respiratory_rate_min=16,
            respiratory_rate_max=30,
            systolic_bp_min=90,
            systolic_bp_max=115,
        ),
        AgeGroup.ADOLESCENT: AgeVitalRanges(
            heart_rate_min=60,
            heart_rate_max=100,
            respiratory_rate_min=12,
            respiratory_rate_max=20,
            systolic_bp_min=100,
            systolic_bp_max=130,
        ),
        AgeGroup.ADULT: AgeVitalRanges(
            heart_rate_min=60,
            heart_rate_max=100,
            respiratory_rate_min=12,
            respiratory_rate_max=20,
            systolic_bp_min=90,
            systolic_bp_max=140,
        ),
    })
)


# ESI Handbook Figure 6-1.
HIGH_RISK_VITALS_BY_AGE: Final[
    Mapping[AgeGroup, AgeHighRiskThresholds]
] = MappingProxyType({
    AgeGroup.NEONATE: AgeHighRiskThresholds(
        heart_rate_above=190,
        respiratory_rate_above=60,
    ),
    AgeGroup.INFANT: AgeHighRiskThresholds(
        heart_rate_above=180,
        respiratory_rate_above=55,
    ),
    AgeGroup.TODDLER: AgeHighRiskThresholds(
        heart_rate_above=140,
        respiratory_rate_above=40,
    ),
    AgeGroup.PRESCHOOL: AgeHighRiskThresholds(
        heart_rate_above=120,
        respiratory_rate_above=35,
    ),
    AgeGroup.SCHOOL_AGE: AgeHighRiskThresholds(
        heart_rate_above=120,
        respiratory_rate_above=30,
    ),
    AgeGroup.ADOLESCENT: AgeHighRiskThresholds(
        heart_rate_above=100,
        respiratory_rate_above=20,
    ),
    AgeGroup.ADULT: AgeHighRiskThresholds(
        heart_rate_above=100,
        respiratory_rate_above=20,
    ),
})


# Heart rate / respiratory rate critical floors (age-agnostic).
HR_CRITICAL_BELOW: Final = 40
RR_CRITICAL_BELOW: Final = 8

# Oxygen saturation.
SPO2_CRITICAL_BELOW: Final = 85
SPO2_HIGH_RISK_THRESHOLD: Final = 92
SPO2_MILD_ABNORMAL_THRESHOLD: Final = 95

# Blood pressure.
SBP_CRITICAL_BELOW: Final = 60
SBP_HYPOTENSION_THRESHOLD: Final = 90
SBP_HYPERTENSIVE_URGENCY: Final = 180
DBP_HYPERTENSIVE_URGENCY: Final = 120

# Adult diastolic reference when age-specific DBP ranges are unavailable.
ADULT_DBP_NORMAL_MIN: Final = 50
ADULT_DBP_NORMAL_MAX: Final = 90

# Temperature clinical thresholds.
TEMP_CRITICAL_LOW: Final = 32.0
TEMP_CRITICAL_HIGH: Final = 41.0
TEMP_PEDIATRIC_FEVER_90_DAYS: Final = 38.0
TEMP_PEDIATRIC_FEVER_OLDER: Final = 38.5
TEMP_HYPOTHERMIA: Final = 36.0
TEMP_ADULT_HYPOTHERMIA: Final = 35.0
TEMP_ADULT_HIGH_FEVER: Final = 39.0

# Core normal range used to distinguish NORMAL from mild ABNORMAL.
TEMP_CORE_NORMAL_MIN: Final = 36.5
TEMP_CORE_NORMAL_MAX: Final = 37.5

# Plausible measurement limits, distinct from clinical thresholds.
TEMP_INPUT_MIN: Final = 20.0
TEMP_INPUT_MAX: Final = 50.0
