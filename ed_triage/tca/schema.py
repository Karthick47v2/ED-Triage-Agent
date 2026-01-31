"""Output schema for TCA (Phase 2)."""
from typing import List, Literal
from pydantic import BaseModel, Field


class TCAResult(BaseModel):
    """Final triage classification from TCA (vitals + physical exam)."""
    final_esi: int = Field(
        ..., 
        description="Final ESI level (1-5) after applying ESI decision algorithm",
        ge=1, 
        le=5
    )
    confidence: float = Field(
        ..., 
        description="Calibrated confidence score (0.0-1.0) in the classification",
        ge=0.0, 
        le=1.0
    )
    rationale: str = Field(
        ..., 
        description="Structured reasoning with explicit ESI criteria references"
    )
    esi_criteria_references: List[str] = Field(
        ..., 
        description="Specific ESI handbook criteria supporting the classification"
    )
    uncertainty_flags: List[str] = Field(
        default_factory=list, 
        description="Flags for cases requiring additional clinical scrutiny"
    )
    requires_immediate_intervention: bool = Field(
        False, 
        description="True if ESI-1 criteria met (immediate life-saving intervention)"
    )
    high_risk_situation: bool = Field(
        False, 
        description="True if ESI-2 criteria met (high-risk, altered mental status, severe distress)"
    )
    predicted_resources: Literal["zero", "one", "two_or_more"] = Field(
        ..., 
        description="Predicted resource utilization for ESI 3-5 classification"
    )
