"""
Clinical Reasoning Agent (CRA)
"""
from ed_triage.cra.agent import create_cra_chain
from ed_triage.cra.schema import ClinicalReasoningInput, ClinicalReasoningOutput

__all__ = [
    "create_cra_chain",
    "ClinicalReasoningInput",
    "ClinicalReasoningOutput"
]
