"""Vital signs assessment against ESI handbook thresholds (Table 6-1, Figure 6-1, Table 6-2)."""
from typing import Optional
from dataclasses import dataclass
from enum import Enum


class VitalStatus(str, Enum):
    """ESI-based vital sign status categories."""
    NORMAL = "normal"
    ABNORMAL = "abnormal"  # Outside normal but not high-risk
    HIGH_RISK = "high_risk"  # ESI-2 criteria met (Figure 6-1)
    CRITICAL = "critical"  # ESI-1 criteria (immediately life-threatening)
    UNKNOWN = "unknown"  # Cannot assess (missing data)


@dataclass
class VitalAssessment:
    """Assessment result for a single vital sign."""
    name: str
    value: Optional[float]
    unit: str
    status: VitalStatus
    normal_range: str
    high_risk_threshold: str
    reasoning: str


@dataclass
class VitalSignsAssessment:
    """Complete vital signs assessment result."""
    heart_rate: VitalAssessment
    respiratory_rate: VitalAssessment
    blood_pressure_systolic: VitalAssessment
    blood_pressure_diastolic: VitalAssessment
    oxygen_saturation: VitalAssessment
    temperature: VitalAssessment
    
    # Summary
    has_high_risk_vitals: bool
    has_critical_vitals: bool
    high_risk_findings: list[str]
    abnormal_findings: list[str]
    overall_status: VitalStatus
    esi_recommendation: str


class AgeGroup(str, Enum):
    """Age groups for vital sign thresholds."""
    NEONATE = "neonate"  # < 1 month
    INFANT = "infant"  # 1-12 months
    TODDLER = "toddler"  # 1-3 years
    PRESCHOOL = "preschool"  # 3-5 years
    SCHOOL_AGE = "school_age"  # 5-12 years
    ADOLESCENT = "adolescent"  # 12-18 years
    ADULT = "adult"  # > 18 years


# ESI Handbook Table 6-1: Normal Vital Signs by Age
# Format: (hr_min, hr_max, rr_min, rr_max, sbp_min, sbp_max)
NORMAL_VITALS_BY_AGE = {
    AgeGroup.NEONATE:     (90, 190, 35, 60, 67, 84),
    AgeGroup.INFANT:      (90, 180, 30, 55, 72, 104),
    AgeGroup.TODDLER:     (80, 140, 22, 40, 86, 104),
    AgeGroup.PRESCHOOL:   (65, 120, 18, 35, 89, 112),
    AgeGroup.SCHOOL_AGE:  (70, 120, 16, 30, 90, 115),
    AgeGroup.ADOLESCENT:  (60, 100, 12, 20, 100, 130),
    AgeGroup.ADULT:       (60, 100, 12, 20, 90, 140),
}

# ESI Handbook Figure 6-1: High-Risk Vital Signs (ESI-2 criteria)
# Format: (hr_high_risk, rr_high_risk)
HIGH_RISK_VITALS_BY_AGE = {
    AgeGroup.NEONATE:     (190, 60),
    AgeGroup.INFANT:      (180, 55),
    AgeGroup.TODDLER:     (140, 40),
    AgeGroup.PRESCHOOL:   (120, 35),
    AgeGroup.SCHOOL_AGE:  (120, 30),
    AgeGroup.ADOLESCENT:  (100, 20),
    AgeGroup.ADULT:       (100, 20),
}

# SpO2 threshold (same for all ages per Figure 6-1)
SPO2_HIGH_RISK_THRESHOLD = 92  # < 92% is high-risk

# Blood pressure high-risk thresholds (adult)
SBP_HYPOTENSION_THRESHOLD = 90  # < 90 mmHg
SBP_HYPERTENSIVE_URGENCY = 180  # > 180 mmHg
DBP_HYPERTENSIVE_URGENCY = 120  # > 120 mmHg

# ESI Handbook Table 6-2: Pediatric Temperature Red Flags
# Temp > threshold OR < 36°C in children is high-risk
TEMP_PEDIATRIC_FEVER_90_DAYS = 38.0  # °C - fever in infant < 90 days
TEMP_PEDIATRIC_FEVER_OLDER = 38.5  # °C - fever in child > 3 months
TEMP_HYPOTHERMIA = 36.0  # °C - hypothermia (all pediatric ages)
TEMP_ADULT_HYPOTHERMIA = 35.0  # °C
TEMP_ADULT_HIGH_FEVER = 39.0  # °C


def determine_age_group(age_years: Optional[float] = None, age_months: Optional[float] = None) -> AgeGroup:
    """
    Determine age group from age in years or months.
    Defaults to ADULT if age not provided.
    """
    if age_years is None and age_months is None:
        return AgeGroup.ADULT
    
    # Convert to months for precision
    total_months = (age_years or 0) * 12 + (age_months or 0)
    
    if total_months < 1:
        return AgeGroup.NEONATE
    elif total_months < 12:
        return AgeGroup.INFANT
    elif total_months < 36:  # 1-3 years
        return AgeGroup.TODDLER
    elif total_months < 60:  # 3-5 years
        return AgeGroup.PRESCHOOL
    elif total_months < 144:  # 5-12 years
        return AgeGroup.SCHOOL_AGE
    elif total_months < 216:  # 12-18 years
        return AgeGroup.ADOLESCENT
    else:
        return AgeGroup.ADULT


def parse_temperature(temp_str: Optional[str]) -> Optional[float]:
    """
    Parse temperature string to Celsius.
    Handles formats like "38.2 °C", "100.4 °F", "38.2C", etc.
    """
    if not temp_str:
        return None
    
    temp_str = temp_str.strip().upper()
    
    # Extract numeric value
    import re
    match = re.search(r'([\d.]+)', temp_str)
    if not match:
        return None
    
    value = float(match.group(1))
    
    if '°F' in temp_str or 'F' in temp_str:
        # Convert Fahrenheit to Celsius
        value = (value - 32) * 5 / 9
    
    if value > 50:
        value = (value - 32) * 5 / 9
    
    return round(value, 1)


def assess_heart_rate(
    hr: Optional[int],
    age_group: AgeGroup
) -> VitalAssessment:
    """Assess heart rate against age-appropriate thresholds."""
    if hr is None:
        return VitalAssessment(
            name="Heart Rate",
            value=None,
            unit="bpm",
            status=VitalStatus.UNKNOWN,
            normal_range="N/A",
            high_risk_threshold="N/A",
            reasoning="Heart rate not provided"
        )
    
    hr_min, hr_max, _, _, _, _ = NORMAL_VITALS_BY_AGE[age_group]
    hr_high_risk, _ = HIGH_RISK_VITALS_BY_AGE[age_group]
    
    normal_range = f"{hr_min}-{hr_max} bpm"
    high_risk_threshold = f">{hr_high_risk} bpm"
    
    # Critical: very low HR
    if hr < 40:
        status = VitalStatus.CRITICAL
        reasoning = f"HR {hr} is critically low (severe bradycardia)"
    # High-risk: exceeds Figure 6-1 threshold
    elif hr > hr_high_risk:
        status = VitalStatus.HIGH_RISK
        reasoning = f"HR {hr} exceeds high-risk threshold (>{hr_high_risk}) per ESI Figure 6-1"
    # Abnormal: outside normal range
    elif hr < hr_min or hr > hr_max:
        status = VitalStatus.ABNORMAL
        reasoning = f"HR {hr} is outside normal range ({normal_range})"
    else:
        status = VitalStatus.NORMAL
        reasoning = f"HR {hr} is within normal range ({normal_range})"
    
    return VitalAssessment(
        name="Heart Rate",
        value=hr,
        unit="bpm",
        status=status,
        normal_range=normal_range,
        high_risk_threshold=high_risk_threshold,
        reasoning=reasoning
    )


def assess_respiratory_rate(
    rr: Optional[int],
    age_group: AgeGroup
) -> VitalAssessment:
    """Assess respiratory rate against age-appropriate thresholds."""
    if rr is None:
        return VitalAssessment(
            name="Respiratory Rate",
            value=None,
            unit="breaths/min",
            status=VitalStatus.UNKNOWN,
            normal_range="N/A",
            high_risk_threshold="N/A",
            reasoning="Respiratory rate not provided"
        )
    
    _, _, rr_min, rr_max, _, _ = NORMAL_VITALS_BY_AGE[age_group]
    _, rr_high_risk = HIGH_RISK_VITALS_BY_AGE[age_group]
    
    normal_range = f"{rr_min}-{rr_max} breaths/min"
    high_risk_threshold = f">{rr_high_risk} breaths/min"
    
    # Critical: very low RR (apnea/respiratory failure)
    if rr < 8:
        status = VitalStatus.CRITICAL
        reasoning = f"RR {rr} is critically low (respiratory failure risk)"
    # High-risk: exceeds Figure 6-1 threshold
    elif rr > rr_high_risk:
        status = VitalStatus.HIGH_RISK
        reasoning = f"RR {rr} exceeds high-risk threshold (>{rr_high_risk}) per ESI Figure 6-1"
    # Abnormal: outside normal range
    elif rr < rr_min or rr > rr_max:
        status = VitalStatus.ABNORMAL
        reasoning = f"RR {rr} is outside normal range ({normal_range})"
    else:
        status = VitalStatus.NORMAL
        reasoning = f"RR {rr} is within normal range ({normal_range})"
    
    return VitalAssessment(
        name="Respiratory Rate",
        value=rr,
        unit="breaths/min",
        status=status,
        normal_range=normal_range,
        high_risk_threshold=high_risk_threshold,
        reasoning=reasoning
    )


def assess_blood_pressure_systolic(
    sbp: Optional[int],
    age_group: AgeGroup
) -> VitalAssessment:
    """Assess systolic blood pressure against thresholds."""
    if sbp is None:
        return VitalAssessment(
            name="Systolic Blood Pressure",
            value=None,
            unit="mmHg",
            status=VitalStatus.UNKNOWN,
            normal_range="N/A",
            high_risk_threshold="N/A",
            reasoning="Systolic BP not provided"
        )
    
    _, _, _, _, sbp_min, sbp_max = NORMAL_VITALS_BY_AGE[age_group]
    
    normal_range = f"{sbp_min}-{sbp_max} mmHg"
    high_risk_threshold = f"<{SBP_HYPOTENSION_THRESHOLD} or >{SBP_HYPERTENSIVE_URGENCY} mmHg"
    
    # Critical: severe hypotension
    if sbp < 60:
        status = VitalStatus.CRITICAL
        reasoning = f"SBP {sbp} is critically low (severe hypotension/shock)"
    # High-risk: hypotension
    elif sbp < SBP_HYPOTENSION_THRESHOLD:
        status = VitalStatus.HIGH_RISK
        reasoning = f"SBP {sbp} indicates hypotension (<{SBP_HYPOTENSION_THRESHOLD} mmHg)"
    # High-risk: hypertensive urgency
    elif sbp > SBP_HYPERTENSIVE_URGENCY:
        status = VitalStatus.HIGH_RISK
        reasoning = f"SBP {sbp} indicates hypertensive urgency (>{SBP_HYPERTENSIVE_URGENCY} mmHg)"
    # Abnormal: outside normal range
    elif sbp < sbp_min or sbp > sbp_max:
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
        reasoning=reasoning
    )


def assess_blood_pressure_diastolic(
    dbp: Optional[int],
    age_group: AgeGroup
) -> VitalAssessment:
    """Assess diastolic blood pressure against thresholds."""
    if dbp is None:
        return VitalAssessment(
            name="Diastolic Blood Pressure",
            value=None,
            unit="mmHg",
            status=VitalStatus.UNKNOWN,
            normal_range="N/A",
            high_risk_threshold="N/A",
            reasoning="Diastolic BP not provided"
        )
    
    # Diastolic ranges are less standardized; using general thresholds
    normal_range = "60-80 mmHg (adult)"
    high_risk_threshold = f">{DBP_HYPERTENSIVE_URGENCY} mmHg"
    
    # High-risk: hypertensive urgency
    if dbp > DBP_HYPERTENSIVE_URGENCY:
        status = VitalStatus.HIGH_RISK
        reasoning = f"DBP {dbp} indicates hypertensive urgency (>{DBP_HYPERTENSIVE_URGENCY} mmHg)"
    # Abnormal
    elif dbp < 50 or dbp > 90:
        status = VitalStatus.ABNORMAL
        reasoning = f"DBP {dbp} is outside typical range"
    else:
        status = VitalStatus.NORMAL
        reasoning = f"DBP {dbp} is within normal range"
    
    return VitalAssessment(
        name="Diastolic Blood Pressure",
        value=dbp,
        unit="mmHg",
        status=status,
        normal_range=normal_range,
        high_risk_threshold=high_risk_threshold,
        reasoning=reasoning
    )


def assess_oxygen_saturation(
    spo2: Optional[float]
) -> VitalAssessment:
    """Assess oxygen saturation against thresholds."""
    if spo2 is None:
        return VitalAssessment(
            name="Oxygen Saturation",
            value=None,
            unit="%",
            status=VitalStatus.UNKNOWN,
            normal_range="N/A",
            high_risk_threshold="N/A",
            reasoning="SpO2 not provided"
        )
    
    normal_range = "≥95%"
    high_risk_threshold = f"<{SPO2_HIGH_RISK_THRESHOLD}%"
    
    # Critical: severe hypoxia
    if spo2 < 85:
        status = VitalStatus.CRITICAL
        reasoning = f"SpO2 {spo2}% is critically low (severe hypoxia)"
    # High-risk: per ESI Figure 6-1
    elif spo2 < SPO2_HIGH_RISK_THRESHOLD:
        status = VitalStatus.HIGH_RISK
        reasoning = f"SpO2 {spo2}% is below high-risk threshold (<{SPO2_HIGH_RISK_THRESHOLD}%) per ESI Figure 6-1"
    # Abnormal: mild hypoxia
    elif spo2 < 95:
        status = VitalStatus.ABNORMAL
        reasoning = f"SpO2 {spo2}% is mildly low (normal ≥95%)"
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
        reasoning=reasoning
    )


def assess_temperature(
    temp_celsius: Optional[float],
    age_group: AgeGroup,
    age_days: Optional[int] = None
) -> VitalAssessment:
    """
    Assess temperature against age-appropriate thresholds.
    Per ESI Table 6-2 for pediatric temperature red flags.
    """
    if temp_celsius is None:
        return VitalAssessment(
            name="Temperature",
            value=None,
            unit="°C",
            status=VitalStatus.UNKNOWN,
            normal_range="N/A",
            high_risk_threshold="N/A",
            reasoning="Temperature not provided"
        )
    
    is_pediatric = age_group in [AgeGroup.NEONATE, AgeGroup.INFANT, AgeGroup.TODDLER, 
                                  AgeGroup.PRESCHOOL, AgeGroup.SCHOOL_AGE]
    is_infant_under_90_days = (age_days is not None and age_days < 90) or age_group == AgeGroup.NEONATE
    
    if is_infant_under_90_days:
        fever_threshold = TEMP_PEDIATRIC_FEVER_90_DAYS
        hypothermia_threshold = TEMP_HYPOTHERMIA
        normal_range = "36.0-38.0 °C"
        high_risk_threshold = f">{fever_threshold}°C or <{hypothermia_threshold}°C"
    elif is_pediatric:
        fever_threshold = TEMP_PEDIATRIC_FEVER_OLDER
        hypothermia_threshold = TEMP_HYPOTHERMIA
        normal_range = "36.0-38.5 °C"
        high_risk_threshold = f">{fever_threshold}°C or <{hypothermia_threshold}°C"
    else:  # Adult
        fever_threshold = TEMP_ADULT_HIGH_FEVER
        hypothermia_threshold = TEMP_ADULT_HYPOTHERMIA
        normal_range = "36.0-38.0 °C"
        high_risk_threshold = f">{fever_threshold}°C or <{hypothermia_threshold}°C"
    if temp_celsius < 32:
        status = VitalStatus.CRITICAL
        reasoning = f"Temp {temp_celsius}°C is critically low (severe hypothermia)"
    elif temp_celsius > 41:
        status = VitalStatus.CRITICAL
        reasoning = f"Temp {temp_celsius}°C is critically high (hyperpyrexia)"
    elif temp_celsius < hypothermia_threshold:
        status = VitalStatus.HIGH_RISK
        reasoning = f"Temp {temp_celsius}°C indicates hypothermia (<{hypothermia_threshold}°C) per ESI Table 6-2"
    elif temp_celsius > fever_threshold:
        if is_infant_under_90_days:
            status = VitalStatus.HIGH_RISK
            reasoning = f"Temp {temp_celsius}°C is fever in infant <90 days (>{fever_threshold}°C) per ESI Table 6-2 - HIGH RISK"
        elif is_pediatric:
            status = VitalStatus.HIGH_RISK
            reasoning = f"Temp {temp_celsius}°C exceeds pediatric fever threshold (>{fever_threshold}°C) per ESI Table 6-2"
        else:
            status = VitalStatus.ABNORMAL
            reasoning = f"Temp {temp_celsius}°C is elevated (>{fever_threshold}°C)"
    elif temp_celsius < 36.5 or temp_celsius > 37.5:
        status = VitalStatus.ABNORMAL
        reasoning = f"Temp {temp_celsius}°C is slightly outside normal range"
    else:
        status = VitalStatus.NORMAL
        reasoning = f"Temp {temp_celsius}°C is within normal range"
    
    return VitalAssessment(
        name="Temperature",
        value=temp_celsius,
        unit="°C",
        status=status,
        normal_range=normal_range,
        high_risk_threshold=high_risk_threshold,
        reasoning=reasoning
    )


def assess_vital_signs(
    heart_rate: Optional[int] = None,
    respiratory_rate: Optional[int] = None,
    blood_pressure_systolic: Optional[int] = None,
    blood_pressure_diastolic: Optional[int] = None,
    oxygen_saturation: Optional[float] = None,
    temperature: Optional[str] = None,
    age_years: Optional[float] = None,
    age_months: Optional[float] = None,
    age_days: Optional[int] = None
) -> VitalSignsAssessment:
    """
    Assess all vital signs against ESI handbook thresholds.
    
    Args:
        heart_rate: Heart rate in bpm
        respiratory_rate: Respiratory rate in breaths/min
        blood_pressure_systolic: Systolic BP in mmHg
        blood_pressure_diastolic: Diastolic BP in mmHg
        oxygen_saturation: SpO2 percentage
        temperature: Temperature string (e.g., "38.2 °C" or "100.4 °F")
        age_years: Patient age in years
        age_months: Patient age in months (for infants)
        age_days: Patient age in days (for neonates)
        
    Returns:
        VitalSignsAssessment with complete evaluation
    """
    age_group = determine_age_group(age_years, age_months)
    temp_celsius = parse_temperature(temperature)
    
    hr_assessment = assess_heart_rate(heart_rate, age_group)
    rr_assessment = assess_respiratory_rate(respiratory_rate, age_group)
    sbp_assessment = assess_blood_pressure_systolic(blood_pressure_systolic, age_group)
    dbp_assessment = assess_blood_pressure_diastolic(blood_pressure_diastolic, age_group)
    spo2_assessment = assess_oxygen_saturation(oxygen_saturation)
    temp_assessment = assess_temperature(temp_celsius, age_group, age_days)
    
    all_assessments = [hr_assessment, rr_assessment, sbp_assessment,
                       dbp_assessment, spo2_assessment, temp_assessment]
    high_risk_findings = [a.reasoning for a in all_assessments if a.status == VitalStatus.HIGH_RISK]
    critical_findings = [a.reasoning for a in all_assessments if a.status == VitalStatus.CRITICAL]
    abnormal_findings = [a.reasoning for a in all_assessments if a.status == VitalStatus.ABNORMAL]
    
    has_critical = len(critical_findings) > 0
    has_high_risk = len(high_risk_findings) > 0
    if has_critical:
        overall_status = VitalStatus.CRITICAL
        esi_recommendation = "ESI-1: Critical vital signs detected - immediate intervention required"
    elif has_high_risk:
        overall_status = VitalStatus.HIGH_RISK
        esi_recommendation = "ESI-2: High-risk vital signs detected per ESI Figure 6-1/Table 6-2"
    elif abnormal_findings:
        overall_status = VitalStatus.ABNORMAL
        esi_recommendation = "ESI 3-5: Abnormal vitals present but not meeting high-risk criteria"
    else:
        overall_status = VitalStatus.NORMAL
        esi_recommendation = "ESI 3-5: Vital signs within normal limits, classify based on resources"
    
    return VitalSignsAssessment(
        heart_rate=hr_assessment,
        respiratory_rate=rr_assessment,
        blood_pressure_systolic=sbp_assessment,
        blood_pressure_diastolic=dbp_assessment,
        oxygen_saturation=spo2_assessment,
        temperature=temp_assessment,
        has_high_risk_vitals=has_high_risk,
        has_critical_vitals=has_critical,
        high_risk_findings=high_risk_findings + critical_findings,
        abnormal_findings=abnormal_findings,
        overall_status=overall_status,
        esi_recommendation=esi_recommendation
    )


def format_assessment_for_llm(assessment: VitalSignsAssessment) -> str:
    """
    Format vital signs assessment as a structured string for LLM consumption.
    """
    lines = [
        "## VITAL SIGNS ASSESSMENT (Deterministic Tool Output)",
        "",
        f"**Overall Status: {assessment.overall_status.value.upper()}**",
        f"**ESI Recommendation: {assessment.esi_recommendation}**",
        "",
    ]
    
    if assessment.high_risk_findings:
        lines.append("### HIGH-RISK FINDINGS (ESI-2 criteria met):")
        for finding in assessment.high_risk_findings:
            lines.append(f"- ⚠️ {finding}")
        lines.append("")
    
    if assessment.abnormal_findings:
        lines.append("### ABNORMAL FINDINGS:")
        for finding in assessment.abnormal_findings:
            lines.append(f"- {finding}")
        lines.append("")
    
    lines.append("### INDIVIDUAL VITAL ASSESSMENTS:")
    for vital in [assessment.heart_rate, assessment.respiratory_rate,
                  assessment.blood_pressure_systolic, assessment.oxygen_saturation,
                  assessment.temperature]:
        if vital.status != VitalStatus.UNKNOWN:
            status_icon = {
                VitalStatus.NORMAL: "✓",
                VitalStatus.ABNORMAL: "△",
                VitalStatus.HIGH_RISK: "⚠️",
                VitalStatus.CRITICAL: "🚨"
            }.get(vital.status, "?")
            lines.append(f"- {status_icon} {vital.name}: {vital.value} {vital.unit} → {vital.status.value}")
    
    return "\n".join(lines)


def assess_from_vital_signs_schema(vital_signs, age_years: Optional[float] = None):
    """Assess from VitalSigns Pydantic model."""
    return assess_vital_signs(
        heart_rate=vital_signs.heart_rate_bpm,
        respiratory_rate=vital_signs.respiratory_rate_bpm,
        blood_pressure_systolic=vital_signs.blood_pressure_systolic_mmHg,
        blood_pressure_diastolic=vital_signs.blood_pressure_diastolic_mmHg,
        oxygen_saturation=vital_signs.oxygen_saturation_percent,
        temperature=vital_signs.temperature,
        age_years=age_years
    )


if __name__ == "__main__":
    # Test cases
    print("=" * 60)
    print("TEST 1: Normal adult vitals")
    result = assess_vital_signs(
        heart_rate=75,
        respiratory_rate=16,
        blood_pressure_systolic=120,
        blood_pressure_diastolic=80,
        oxygen_saturation=98,
        temperature="37.0 °C",
        age_years=35
    )
    print(format_assessment_for_llm(result))
    
    print("\n" + "=" * 60)
    print("TEST 2: High-risk adult (tachycardia + hypoxia)")
    result = assess_vital_signs(
        heart_rate=115,
        respiratory_rate=24,
        blood_pressure_systolic=95,
        oxygen_saturation=89,
        temperature="38.5 °C",
        age_years=65
    )
    print(format_assessment_for_llm(result))
    
    print("\n" + "=" * 60)
    print("TEST 3: Febrile infant <90 days (HIGH RISK)")
    result = assess_vital_signs(
        heart_rate=160,
        respiratory_rate=45,
        temperature="38.3 °C",
        age_days=60
    )
    print(format_assessment_for_llm(result))
    
    print("\n" + "=" * 60)
    print("TEST 4: Toddler with high HR")
    result = assess_vital_signs(
        heart_rate=155,
        respiratory_rate=35,
        oxygen_saturation=94,
        age_years=2
    )
    print(format_assessment_for_llm(result))
