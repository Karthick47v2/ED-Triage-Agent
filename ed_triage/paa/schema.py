from typing import Literal
from pydantic import BaseModel, Field

class PriorityAssessment(BaseModel):
    """Output of the Priority Assessment Agent."""
    is_high_priority: bool = Field(..., description="True if the patient requires immediate assessment (jumping the queue).")
    priority_score: Literal["HIGH", "LOW"] = Field(..., description="Classification of queue priority.")
    tentative_esi: int = Field(..., description="Estimated ESI Level (1-5) based on current information.")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0) in the assessment.", ge=0.0, le=1.0)
    reasoning: str = Field(..., description="Explanation for the priority classification, citing risk factors.")
