"""Per-vital and aggregate vital signs assessment."""
from __future__ import annotations

from enum import Enum, auto
from math import isfinite
from numbers import Real

from ed_triage.common.schemas import VitalSigns
from ed_triage.common.vital_assessment.age import (
    determine_age_group_from_days,
    resolve_age_in_days,
)
from ed_triage.common.vital_assessment.tables import (
    ADULT_DBP_NORMAL_MAX,
    ADULT_DBP_NORMAL_MIN,
    DBP_HYPERTENSIVE_URGENCY,
    HIGH_RISK_VITALS_BY_AGE,
    HR_CRITICAL_BELOW,
    NORMAL_VITALS_BY_AGE,
    RR_CRITICAL_BELOW,
    SBP_CRITICAL_BELOW,
    SBP_HYPERTENSIVE_URGENCY,
    SBP_HYPOTENSION_THRESHOLD,
    SPO2_CRITICAL_BELOW,
    SPO2_HIGH_RISK_THRESHOLD,
    SPO2_MILD_ABNORMAL_THRESHOLD,
    TEMP_ADULT_HIGH_FEVER,
    TEMP_ADULT_HYPOTHERMIA,
    TEMP_CORE_NORMAL_MAX,
    TEMP_CORE_NORMAL_MIN,
    TEMP_CRITICAL_HIGH,
    TEMP_CRITICAL_LOW,
    TEMP_HYPOTHERMIA,
    TEMP_INPUT_MAX,
    TEMP_INPUT_MIN,
    TEMP_PEDIATRIC_FEVER_90_DAYS,
    TEMP_PEDIATRIC_FEVER_OLDER,
    AgeGroup,
    VitalAssessment,
    VitalSignsAssessment,
    VitalStatus,
)


_STATUS_PRIORITY = (
    VitalStatus.CRITICAL,
    VitalStatus.HIGH_RISK,
    VitalStatus.ABNORMAL,
    VitalStatus.NORMAL,
)

_ESI_RECOMMENDATIONS = {
    VitalStatus.CRITICAL: (
        "Critical vital signs detected; immediate clinical assessment required."
    ),
    VitalStatus.HIGH_RISK: (
        "High-risk vital signs detected; consider higher acuity in clinical context."
    ),
    VitalStatus.ABNORMAL: (
        "Abnormal vital signs detected; determine ESI level using the full assessment."
    ),
    VitalStatus.UNKNOWN: (
        "Vital signs unavailable or not assessed against configured thresholds; "
        "ESI acuity cannot be determined from vitals alone."
    ),
    VitalStatus.NORMAL: (
        "Measured vital signs do not meet configured abnormal thresholds; "
        "determine ESI level using presentation and expected resources."
    ),
}


class TemperaturePolicy(Enum):
    INFANT_UNDER_90_DAYS = auto()
    PEDIATRIC = auto()
    ADULT = auto()


def _validate_number(
    name: str,
    value: int | float | None,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    if value is None:
        return
    # bool is a subclass of int; reject it as a vital measurement.
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number or None")
    if not isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} cannot be below {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} cannot exceed {maximum}")


def _unknown(name: str, unit: str, reason: str) -> VitalAssessment:
    return VitalAssessment(
        name=name,
        value=None,
        unit=unit,
        status=VitalStatus.UNKNOWN,
        normal_range="N/A",
        high_risk_threshold="N/A",
        reasoning=reason,
    )


def _assess_range_vital(
    *,
    name: str,
    value: int | float | None,
    unit: str,
    input_name: str,
    maximum_value: float,
    missing_reason: str,
    normal_min: float,
    normal_max: float,
    high_risk_above: float,
    critical_below: float,
    short_label: str,
    critical_description: str,
    source: str = "ESI Figure 6-1",
) -> VitalAssessment:
    """Shared ladder: UNKNOWN → CRITICAL (low) → HIGH_RISK (high) → ABNORMAL → NORMAL."""
    _validate_number(
        input_name, value, minimum=0, maximum=maximum_value
    )
    if value is None:
        return _unknown(name, unit, missing_reason)

    normal_range = f"{normal_min:g}-{normal_max:g} {unit}"
    high_risk_threshold = f">{high_risk_above:g} {unit}"

    if value < critical_below:
        status = VitalStatus.CRITICAL
        reasoning = (
            f"{short_label} {value:g} is critically low ({critical_description})"
        )
    elif value > high_risk_above:
        status = VitalStatus.HIGH_RISK
        reasoning = (
            f"{short_label} {value:g} exceeds high-risk threshold "
            f"(>{high_risk_above:g}) per {source}"
        )
    elif not normal_min <= value <= normal_max:
        status = VitalStatus.ABNORMAL
        reasoning = (
            f"{short_label} {value:g} is outside normal range ({normal_range})"
        )
    else:
        status = VitalStatus.NORMAL
        reasoning = (
            f"{short_label} {value:g} is within normal range ({normal_range})"
        )

    return VitalAssessment(
        name=name,
        value=value,
        unit=unit,
        status=status,
        normal_range=normal_range,
        high_risk_threshold=high_risk_threshold,
        reasoning=reasoning,
    )


def assess_heart_rate(hr: int | None, age_group: AgeGroup) -> VitalAssessment:
    normal = NORMAL_VITALS_BY_AGE[age_group]
    high_risk = HIGH_RISK_VITALS_BY_AGE[age_group]
    return _assess_range_vital(
        name="Heart Rate",
        input_name="heart_rate",
        value=hr,
        unit="bpm",
        maximum_value=350,
        missing_reason="Heart rate not provided",
        normal_min=normal.heart_rate_min,
        normal_max=normal.heart_rate_max,
        high_risk_above=high_risk.heart_rate_above,
        critical_below=HR_CRITICAL_BELOW,
        short_label="HR",
        critical_description="severe bradycardia",
    )


def assess_respiratory_rate(
    rr: int | None, age_group: AgeGroup
) -> VitalAssessment:
    normal = NORMAL_VITALS_BY_AGE[age_group]
    high_risk = HIGH_RISK_VITALS_BY_AGE[age_group]
    return _assess_range_vital(
        name="Respiratory Rate",
        input_name="respiratory_rate",
        value=rr,
        unit="breaths/min",
        maximum_value=100,
        missing_reason="Respiratory rate not provided",
        normal_min=normal.respiratory_rate_min,
        normal_max=normal.respiratory_rate_max,
        high_risk_above=high_risk.respiratory_rate_above,
        critical_below=RR_CRITICAL_BELOW,
        short_label="RR",
        critical_description="respiratory failure risk",
    )


def assess_blood_pressure_systolic(
    sbp: int | None, age_group: AgeGroup
) -> VitalAssessment:
    _validate_number(
        "blood_pressure_systolic", sbp, minimum=0, maximum=350
    )
    if sbp is None:
        return _unknown(
            "Systolic Blood Pressure", "mmHg", "Systolic BP not provided"
        )

    normal = NORMAL_VITALS_BY_AGE[age_group]
    normal_range = (
        f"{normal.systolic_bp_min}-{normal.systolic_bp_max} mmHg"
    )
    high_risk_threshold = (
        f"<{SBP_HYPOTENSION_THRESHOLD} or >{SBP_HYPERTENSIVE_URGENCY} mmHg"
    )

    if sbp < SBP_CRITICAL_BELOW:
        status = VitalStatus.CRITICAL
        reasoning = f"SBP {sbp} is critically low (severe hypotension/shock)"
    elif sbp < SBP_HYPOTENSION_THRESHOLD:
        status = VitalStatus.HIGH_RISK
        reasoning = (
            f"SBP {sbp} indicates hypotension (<{SBP_HYPOTENSION_THRESHOLD} mmHg)"
        )
    elif sbp > SBP_HYPERTENSIVE_URGENCY:
        status = VitalStatus.HIGH_RISK
        reasoning = (
            f"SBP {sbp} indicates hypertensive urgency "
            f"(>{SBP_HYPERTENSIVE_URGENCY} mmHg)"
        )
    elif sbp < normal.systolic_bp_min or sbp > normal.systolic_bp_max:
        status = VitalStatus.ABNORMAL
        reasoning = f"SBP {sbp} is outside normal range ({normal_range})"
    else:
        status = VitalStatus.NORMAL
        reasoning = f"SBP {sbp} is within normal range ({normal_range})"

    return VitalAssessment(
        name="Systolic Blood Pressure",
        value=sbp,
        unit="mmHg",
        status=status,
        normal_range=normal_range,
        high_risk_threshold=high_risk_threshold,
        reasoning=reasoning,
    )


def assess_blood_pressure_diastolic(
    dbp: int | None, age_group: AgeGroup
) -> VitalAssessment:
    _validate_number(
        "blood_pressure_diastolic", dbp, minimum=0, maximum=250
    )
    if dbp is None:
        return _unknown(
            "Diastolic Blood Pressure", "mmHg", "Diastolic BP not provided"
        )

    # ESI handbook SBP bands are age-adjusted; DBP lacks a parallel table here.
    # For non-adults: only apply hypertensive-urgency HIGH_RISK.
    high_risk_threshold = f">{DBP_HYPERTENSIVE_URGENCY} mmHg"
    adult_range = f"{ADULT_DBP_NORMAL_MIN}-{ADULT_DBP_NORMAL_MAX} mmHg"

    if dbp > DBP_HYPERTENSIVE_URGENCY:
        return VitalAssessment(
            name="Diastolic Blood Pressure",
            value=dbp,
            unit="mmHg",
            status=VitalStatus.HIGH_RISK,
            normal_range=(
                adult_range
                if age_group == AgeGroup.ADULT
                else "No age-specific configured range"
            ),
            high_risk_threshold=high_risk_threshold,
            reasoning=(
                f"DBP {dbp} indicates hypertensive urgency "
                f"(>{DBP_HYPERTENSIVE_URGENCY} mmHg)"
            ),
        )

    if age_group != AgeGroup.ADULT:
        return VitalAssessment(
            name="Diastolic Blood Pressure",
            value=dbp,
            unit="mmHg",
            status=VitalStatus.NOT_ASSESSED,
            normal_range="No age-specific configured range",
            high_risk_threshold=high_risk_threshold,
            reasoning=(
                f"DBP {dbp} recorded, but no age-specific pediatric "
                "diastolic reference range is configured"
            ),
        )

    if dbp < ADULT_DBP_NORMAL_MIN or dbp > ADULT_DBP_NORMAL_MAX:
        status = VitalStatus.ABNORMAL
        reasoning = (
            f"DBP {dbp} is outside configured adult range ({adult_range})"
        )
    else:
        status = VitalStatus.NORMAL
        reasoning = (
            f"DBP {dbp} is within configured adult range ({adult_range})"
        )

    return VitalAssessment(
        name="Diastolic Blood Pressure",
        value=dbp,
        unit="mmHg",
        status=status,
        normal_range=adult_range,
        high_risk_threshold=high_risk_threshold,
        reasoning=reasoning,
    )


def assess_oxygen_saturation(spo2: float | None) -> VitalAssessment:
    _validate_number("oxygen_saturation", spo2, minimum=0, maximum=100)
    if spo2 is None:
        return _unknown("Oxygen Saturation", "%", "SpO2 not provided")

    normal_range = f"≥{SPO2_MILD_ABNORMAL_THRESHOLD}%"
    high_risk_threshold = f"<{SPO2_HIGH_RISK_THRESHOLD}%"

    if spo2 < SPO2_CRITICAL_BELOW:
        status = VitalStatus.CRITICAL
        reasoning = f"SpO2 {spo2}% is critically low (severe hypoxia)"
    elif spo2 < SPO2_HIGH_RISK_THRESHOLD:
        status = VitalStatus.HIGH_RISK
        reasoning = (
            f"SpO2 {spo2}% is below high-risk threshold "
            f"(<{SPO2_HIGH_RISK_THRESHOLD}%) per ESI Figure 6-1"
        )
    elif spo2 < SPO2_MILD_ABNORMAL_THRESHOLD:
        status = VitalStatus.ABNORMAL
        reasoning = (
            f"SpO2 {spo2}% is mildly low "
            f"(normal ≥{SPO2_MILD_ABNORMAL_THRESHOLD}%)"
        )
    else:
        status = VitalStatus.NORMAL
        reasoning = f"SpO2 {spo2}% is within normal range"

    return VitalAssessment(
        name="Oxygen Saturation",
        value=spo2,
        unit="%",
        status=status,
        normal_range=normal_range,
        high_risk_threshold=high_risk_threshold,
        reasoning=reasoning,
    )


def _temperature_policy(
    age_group: AgeGroup,
    resolved_age_days: float | None,
) -> TemperaturePolicy:
    if age_group == AgeGroup.NEONATE:
        return TemperaturePolicy.INFANT_UNDER_90_DAYS
    if resolved_age_days is not None and resolved_age_days < 90:
        return TemperaturePolicy.INFANT_UNDER_90_DAYS
    if age_group in {
        AgeGroup.INFANT,
        AgeGroup.TODDLER,
        AgeGroup.PRESCHOOL,
        AgeGroup.SCHOOL_AGE,
    }:
        return TemperaturePolicy.PEDIATRIC
    return TemperaturePolicy.ADULT


def assess_temperature(
    temp_celsius: float | None,
    age_group: AgeGroup,
    resolved_age_days: float | None = None,
) -> VitalAssessment:
    """Assess temperature against age-appropriate ESI Table 6-2 thresholds."""
    _validate_number(
        "temperature_celsius",
        temp_celsius,
        minimum=TEMP_INPUT_MIN,
        maximum=TEMP_INPUT_MAX,
    )
    if temp_celsius is None:
        return _unknown("Temperature", "°C", "Temperature not provided")

    policy = _temperature_policy(age_group, resolved_age_days)
    if policy is TemperaturePolicy.INFANT_UNDER_90_DAYS:
        fever_threshold = TEMP_PEDIATRIC_FEVER_90_DAYS
        hypothermia_threshold = TEMP_HYPOTHERMIA
        high_risk_threshold = (
            f"≥{fever_threshold}°C or ≤{hypothermia_threshold}°C"
        )
    elif policy is TemperaturePolicy.PEDIATRIC:
        fever_threshold = TEMP_PEDIATRIC_FEVER_OLDER
        hypothermia_threshold = TEMP_HYPOTHERMIA
        high_risk_threshold = (
            f"≥{fever_threshold}°C or ≤{hypothermia_threshold}°C"
        )
    else:
        fever_threshold = TEMP_ADULT_HIGH_FEVER
        hypothermia_threshold = TEMP_ADULT_HYPOTHERMIA
        # Adult high fever is ABNORMAL only; hypothermia remains HIGH_RISK.
        high_risk_threshold = f"≤{hypothermia_threshold}°C"

    normal_range = f"{TEMP_CORE_NORMAL_MIN}-{TEMP_CORE_NORMAL_MAX} °C"

    if temp_celsius < TEMP_CRITICAL_LOW:
        status = VitalStatus.CRITICAL
        reasoning = f"Temp {temp_celsius}°C is critically low (severe hypothermia)"
    elif temp_celsius > TEMP_CRITICAL_HIGH:
        status = VitalStatus.CRITICAL
        reasoning = f"Temp {temp_celsius}°C is critically high (hyperpyrexia)"
    elif temp_celsius <= hypothermia_threshold:
        status = VitalStatus.HIGH_RISK
        reasoning = (
            f"Temp {temp_celsius}°C indicates hypothermia "
            f"(≤{hypothermia_threshold}°C) per ESI Table 6-2"
        )
    elif temp_celsius >= fever_threshold:
        if policy is TemperaturePolicy.INFANT_UNDER_90_DAYS:
            status = VitalStatus.HIGH_RISK
            reasoning = (
                f"Temp {temp_celsius}°C is fever in infant <90 days "
                f"(≥{fever_threshold}°C) per ESI Table 6-2 - HIGH RISK"
            )
        elif policy is TemperaturePolicy.PEDIATRIC:
            status = VitalStatus.HIGH_RISK
            reasoning = (
                f"Temp {temp_celsius}°C exceeds pediatric fever threshold "
                f"(≥{fever_threshold}°C) per ESI Table 6-2"
            )
        else:
            status = VitalStatus.ABNORMAL
            reasoning = (
                f"Temp {temp_celsius}°C is elevated (≥{fever_threshold}°C)"
            )
    elif not (TEMP_CORE_NORMAL_MIN <= temp_celsius <= TEMP_CORE_NORMAL_MAX):
        status = VitalStatus.ABNORMAL
        reasoning = (
            f"Temp {temp_celsius}°C is outside the configured normal range "
            f"({normal_range})"
        )
    else:
        status = VitalStatus.NORMAL
        reasoning = (
            f"Temp {temp_celsius}°C is within the configured normal range "
            f"({normal_range})"
        )

    return VitalAssessment(
        name="Temperature",
        value=temp_celsius,
        unit="°C",
        status=status,
        normal_range=normal_range,
        high_risk_threshold=high_risk_threshold,
        reasoning=reasoning,
    )


def _summarize_assessments(
    assessments: tuple[VitalAssessment, ...],
) -> tuple[VitalStatus, list[str], list[str], list[str]]:
    findings = {
        VitalStatus.CRITICAL: [],
        VitalStatus.HIGH_RISK: [],
        VitalStatus.ABNORMAL: [],
    }
    for assessment in assessments:
        if assessment.status in findings:
            findings[assessment.status].append(assessment.reasoning)

    overall_status = next(
        (
            status
            for status in _STATUS_PRIORITY
            if any(a.status == status for a in assessments)
        ),
        VitalStatus.UNKNOWN,
    )
    return (
        overall_status,
        findings[VitalStatus.CRITICAL],
        findings[VitalStatus.HIGH_RISK],
        findings[VitalStatus.ABNORMAL],
    )


def assess_vital_signs(
    heart_rate: int | None = None,
    respiratory_rate: int | None = None,
    blood_pressure_systolic: int | None = None,
    blood_pressure_diastolic: int | None = None,
    oxygen_saturation: float | None = None,
    temperature_celsius: float | None = None,
    *,
    age_years: float | None = None,
    age_months: float | None = None,
    age_days: int | None = None,
) -> VitalSignsAssessment:
    """Assess all vital signs against ESI handbook thresholds."""
    resolved_age_days = resolve_age_in_days(
        age_years=age_years,
        age_months=age_months,
        age_days=age_days,
    )
    age_group = determine_age_group_from_days(resolved_age_days)

    assessments = (
        assess_heart_rate(heart_rate, age_group),
        assess_respiratory_rate(respiratory_rate, age_group),
        assess_blood_pressure_systolic(blood_pressure_systolic, age_group),
        assess_blood_pressure_diastolic(blood_pressure_diastolic, age_group),
        assess_oxygen_saturation(oxygen_saturation),
        assess_temperature(
            temperature_celsius, age_group, resolved_age_days
        ),
    )
    (
        overall_status,
        critical_findings,
        high_risk_findings,
        abnormal_findings,
    ) = _summarize_assessments(assessments)
    (
        hr_assessment,
        rr_assessment,
        sbp_assessment,
        dbp_assessment,
        spo2_assessment,
        temp_assessment,
    ) = assessments

    return VitalSignsAssessment(
        heart_rate=hr_assessment,
        respiratory_rate=rr_assessment,
        blood_pressure_systolic=sbp_assessment,
        blood_pressure_diastolic=dbp_assessment,
        oxygen_saturation=spo2_assessment,
        temperature=temp_assessment,
        has_high_risk_vitals=bool(high_risk_findings),
        has_critical_vitals=bool(critical_findings),
        high_risk_findings=high_risk_findings,
        critical_findings=critical_findings,
        abnormal_findings=abnormal_findings,
        overall_status=overall_status,
        esi_recommendation=_ESI_RECOMMENDATIONS[overall_status],
    )


def assess_from_vital_signs_schema(
    vital_signs: VitalSigns,
    *,
    age_years: float | None = None,
    age_months: float | None = None,
    age_days: int | None = None,
) -> VitalSignsAssessment:
    """Assess a VitalSigns model using age-appropriate thresholds."""
    return assess_vital_signs(
        heart_rate=vital_signs.heart_rate_bpm,
        respiratory_rate=vital_signs.respiratory_rate_bpm,
        blood_pressure_systolic=vital_signs.blood_pressure_systolic_mmhg,
        blood_pressure_diastolic=vital_signs.blood_pressure_diastolic_mmhg,
        oxygen_saturation=vital_signs.oxygen_saturation_percent,
        temperature_celsius=vital_signs.temperature_celsius(),
        age_years=age_years,
        age_months=age_months,
        age_days=age_days,
    )
