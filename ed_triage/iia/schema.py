from typing import List, Optional
from pydantic import BaseModel, Field

class Symptom(BaseModel):
    """Symptom with OLDCARTS fields."""
    name: str = Field(description="Name of the symptom (e.g., 'Chest Pain', 'Sore Throat').")
    onset: Optional[str] = Field(None, description="When did it start?")
    location: Optional[str] = Field(None, description="Where is it located?")
    duration: Optional[str] = Field(None, description="How long has it lasted?")
    character: Optional[str] = Field(None, description="What does it feel like? (e.g., sharp, dull)")
    aggravating_factors: Optional[List[str]] = Field(None, description="What makes it worse?")
    alleviating_factors: Optional[List[str]] = Field(None, description="What makes it better?")
    radiation: Optional[str] = Field(None, description="Does the pain move anywhere?")
    timing: Optional[str] = Field(None, description="Is it constant or intermittent?")
    severity: Optional[str] = Field(None, description="Severity (1-10 scale or qualitative).")

class IntakeSummary(BaseModel):
    """Patient intake summary (chief complaint, HPI, history, meds, allergies, emergency flag)."""
    chief_complaint: str = Field(..., description="The main reason for the visit.")
    hpi: List[Symptom] = Field(default_factory=list, description="List of symptoms detailed via OLDCARTS.")
    medical_history: Optional[List[str]] = Field(default_factory=list, description="Past medical history.")
    medications: Optional[List[str]] = Field(default_factory=list, description="Current medications.")
    allergies: Optional[List[str]] = Field(default_factory=list, description="Known allergies.")
    emergency_detected: bool = Field(False, description="True if an emergency condition is detected requiring immediate escalation.")
    emergency_reason: Optional[str] = Field(None, description="Reason for emergency escalation if detected.")
    interview_complete: bool = Field(False, description="True if the intake interview runs to completion (or shortened completion).")
