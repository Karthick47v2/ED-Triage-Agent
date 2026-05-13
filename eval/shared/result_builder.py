"""Helpers for assembling EvaluationResult objects from pipeline state dicts."""
from __future__ import annotations

from typing import Any, Mapping, Optional

from eval.schemas import EvaluationResult


def apply_phase1_state(
    result: EvaluationResult, state: Optional[Mapping[str, Any]]
) -> None:
    """Populate Phase 1 fields on ``result`` from a Phase 1 pipeline state dict.

    Sets ``result.success`` based on the presence of a PAA result.
    """
    paa = state.get("paa_result") if state else None
    intake = state.get("intake_data") if state else None
    cra = state.get("cra_result") if state else None

    if paa:
        result.predicted_esi = paa.tentative_esi
        result.predicted_priority = paa.priority_score
        result.confidence = paa.confidence
    if intake:
        result.emergency_detected = intake.emergency_detected
        result.chief_complaint = intake.chief_complaint
        result.intake_data = intake
    if cra:
        result.cra_result = cra

    result.success = bool(paa)
