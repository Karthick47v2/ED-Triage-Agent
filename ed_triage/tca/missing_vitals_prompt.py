"""Append explicit missing-vital fields to TCA prompt (Problem 4)."""
from __future__ import annotations

from ed_triage.common.schemas import VitalSigns

# Keep in sync with VitalSigns schema field names (non-temperature).
VITAL_SIGN_TRIAGE_FIELDS: tuple[str, ...] = (
    "heart_rate_bpm",
    "respiratory_rate_bpm",
    "blood_pressure_systolic_mmhg",
    "blood_pressure_diastolic_mmhg",
    "oxygen_saturation_percent",
)

VITAL_FIELD_LABELS = {
    "heart_rate_bpm": "Heart rate (bpm)",
    "respiratory_rate_bpm": "Respiratory rate (breaths/min)",
    "blood_pressure_systolic_mmhg": "Systolic blood pressure (mmHg)",
    "blood_pressure_diastolic_mmhg": "Diastolic blood pressure (mmHg)",
    "temperature": "Temperature",
    "oxygen_saturation_percent": "Oxygen saturation (SpO2 %)",
}


def missing_vital_field_labels(vital_signs: VitalSigns) -> list[str]:
    """Human-readable labels for VitalSigns fields that are missing or incomplete.

    Temperature is missing when it cannot be converted to Celsius (value and unit
    both required).
    """
    missing = [
        VITAL_FIELD_LABELS[f]
        for f in VITAL_SIGN_TRIAGE_FIELDS
        if getattr(vital_signs, f, None) is None
    ]
    if vital_signs.temperature_celsius() is None:
        missing.append(VITAL_FIELD_LABELS["temperature"])
    return missing


def format_missing_vitals_footer(vital_signs: VitalSigns) -> str:
    """
    Short block for the TCA human prompt listing absent schema fields.
    Returns empty string if all fields are populated.
    """
    missing = missing_vital_field_labels(vital_signs)
    if not missing:
        return ""
    joined = ", ".join(missing)
    return (
        "\n\n**Missing vital fields (not recorded):** " +
        joined +
        "\n**Instruction:** Do not assume normal values for missing vitals. Incorporate uncertainty "
        "into your ESI classification and confidence. For high-acuity presentations, do not downgrade "
        "acuity solely because the deterministic summary is NORMAL or ABNORMAL on a **partial** set of "
        "vitals when key measurements (e.g. SpO2, heart rate, blood pressure) were not recorded.")
