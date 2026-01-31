"""Common schemas for Phase 2 (vitals, physical exam)."""
from typing import Optional
from pydantic import BaseModel, Field


class VitalSigns(BaseModel):
    """Vital signs collected by clinician."""
    heart_rate_bpm: Optional[int] = Field(None, description="Heart rate in beats per minute")
    respiratory_rate_bpm: Optional[int] = Field(None, description="Respiratory rate in breaths per minute")
    blood_pressure_systolic_mmHg: Optional[int] = Field(None, description="Systolic blood pressure in mmHg")
    blood_pressure_diastolic_mmHg: Optional[int] = Field(None, description="Diastolic blood pressure in mmHg")
    temperature: Optional[str] = Field(None, description="Temperature with unit, e.g. '38.2 °C'")
    oxygen_saturation_percent: Optional[float] = Field(None, description="SpO2 percentage")


class PhysicalExam(BaseModel):
    """Physical exam findings by clinician."""
    physical_exam: Optional[str] = Field(
        None, 
        description="Free-text physical exam observations (appearance, skin, respiratory effort, mental status)"
    )
