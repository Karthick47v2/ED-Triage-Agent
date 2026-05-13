from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class DifferentialDiagnosis(BaseModel):
    """Potential clinical condition and its likelihood."""
    model_config = ConfigDict(extra="forbid")
    condition: str = Field(..., description="Name of the potential condition.")
    likelihood: str = Field(...,
                            description="Likelihood of the condition (e.g., High, Moderate, Low).")
    reasoning: str = Field(...,
                           description="Brief clinical justification for this consideration.")


class CRAResult(BaseModel):
    """Output of the Clinical Reasoning Agent (CRA)."""
    model_config = ConfigDict(extra="forbid")
    differential_diagnoses: List[DifferentialDiagnosis] = Field(
        ..., description="List of potential differential diagnoses.")
    risk_factors: List[str] = Field(...,
                                    description="List of identified clinical risk factors.")
    critical_findings: List[str] = Field(
        ..., description="Any life-threatening or time-sensitive findings.")
    esi_handbook_references: List[str] = Field(
        ..., description="Direct references or quotes from the ESI handbook retrieved via RAG.")
    suggested_esi_level: Optional[int] = Field(
        None, description="A tentative ESI level (1-5) based on the analysis.")
    clinical_explanation: str = Field(
        ..., description="Comprehensive clinical reasoning for the assessment.")
