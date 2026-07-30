"""Common schemas for Phase 2: vital signs and physical examination."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class VitalSigns(BaseModel):
    """Vital signs collected by a clinician."""

    heart_rate_bpm: int | None = Field(
        default=None,
        ge=0,
        le=350,
        description="Heart rate in beats per minute.",
    )
    respiratory_rate_bpm: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Respiratory rate in breaths per minute.",
    )
    blood_pressure_systolic_mmhg: int | None = Field(
        default=None,
        ge=0,
        le=350,
        description="Systolic blood pressure in mmHg.",
    )
    blood_pressure_diastolic_mmhg: int | None = Field(
        default=None,
        ge=0,
        le=250,
        description="Diastolic blood pressure in mmHg.",
    )
    temperature_value: float | None = Field(
        default=None,
        description="Measured body temperature.",
    )
    temperature_unit: Literal["C", "F"] | None = Field(
        default=None,
        description="Temperature unit: C for Celsius or F for Fahrenheit.",
    )
    oxygen_saturation_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Peripheral oxygen saturation as a percentage.",
    )

    def temperature_celsius(self) -> float | None:
        """Normalize temperature to Celsius when value and unit are present."""
        if self.temperature_value is None or self.temperature_unit is None:
            return None
        if self.temperature_unit == "C":
            return round(float(self.temperature_value), 1)
        return round((float(self.temperature_value) - 32.0) * 5.0 / 9.0, 1)


class PhysicalExam(BaseModel):
    """Physical examination findings documented by a clinician."""

    physical_exam: str | None = Field(
        default=None,
        description=(
            "Free-text physical examination observations, including general "
            "appearance, skin, respiratory effort, and mental status."
        ),
    )
