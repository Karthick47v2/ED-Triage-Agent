"""
Pydantic schemas for evaluation data models.
"""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from ed_triage.iia.schema import IntakeSummary
from ed_triage.cra.schema import CRAResult
from ed_triage.tca.schema import TCAResult
from ed_triage.common.schemas import VitalSigns, PhysicalExam


class EvaluationScenario(BaseModel):
    """Input scenario from ETEK dataset."""
    scenario_number: int
    description: str
    category: int = Field(..., description="Ground truth ESI level (1-5)")
    rationale: str


class EvaluationResult(BaseModel):
    """Result from a single evaluation run."""
    scenario_number: int
    ground_truth_esi: int
    ground_truth_priority: Literal["HIGH", "LOW"]
    
    # Phase 1: PAA Predictions
    predicted_esi: Optional[int] = None
    predicted_priority: Optional[Literal["HIGH", "LOW"]] = None
    confidence: Optional[float] = None
    
    # IIA Outputs
    emergency_detected: bool = False
    chief_complaint: Optional[str] = None
    intake_data: Optional[IntakeSummary] = None
    
    # CRA Outputs (Phase 1)
    cra_result: Optional[CRAResult] = None
    
    # Phase 2: TCA Predictions
    phase2_predicted_esi: Optional[int] = None
    phase2_confidence: Optional[float] = None
    phase2_rationale: Optional[str] = None
    tca_result: Optional[TCAResult] = None
    cra_phase2_result: Optional[CRAResult] = None
    
    # Phase 2: Input data
    vital_signs: Optional[VitalSigns] = None
    physical_exam: Optional[PhysicalExam] = None
    
    # Timing
    phase1_latency_ms: float = 0.0
    phase2_latency_ms: float = 0.0
    interview_turns: int = 0
    
    # Error tracking
    success: bool = True
    error_message: Optional[str] = None
    
    # Conversation transcript (for debugging)
    transcript: Optional[List[dict]] = None



class EvaluationSummary(BaseModel):
    """Aggregate metrics for evaluation run."""
    total_scenarios: int
    successful_runs: int
    failed_runs: int
    
    # ESI Accuracy
    esi_exact_accuracy: float = Field(..., description="% exact match")
    esi_within_one_accuracy: float = Field(..., description="% within ±1 level")
    
    # Priority Classification
    priority_accuracy: float
    high_priority_sensitivity: float = Field(..., description="TP / (TP + FN)")
    high_priority_specificity: float = Field(..., description="TN / (TN + FP)")
    
    # Triage Safety
    undertriage_rate: float = Field(..., description="% predicted ≥2 levels below")
    overtriage_rate: float = Field(..., description="% predicted ≥2 levels above")
    
    # Latency
    latency_mean_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    
    # Per-ESI breakdown
    per_esi_accuracy: Optional[dict] = None
