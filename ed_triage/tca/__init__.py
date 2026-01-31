"""
Triage Classification Agent (TCA)
"""
from ed_triage.tca.agent import create_tca_chain
from ed_triage.tca.schema import TriageClassificationInput, TriageClassificationOutput

__all__ = [
    "create_tca_chain",
    "TriageClassificationInput",
    "TriageClassificationOutput"
]
