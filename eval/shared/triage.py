"""Shared triage mapping and result-extraction helpers."""
from __future__ import annotations

from typing import List, Sequence, Tuple

from eval.schemas import EvaluationResult


def derive_priority(esi_level: int) -> str:
    """ESI 1-3 -> HIGH, ESI 4-5 -> LOW (matches PAA priority semantics)."""
    return "HIGH" if esi_level <= 3 else "LOW"


def phase1_esi_pairs(
    results: Sequence[EvaluationResult],
) -> Tuple[List[int], List[int]]:
    """(ground_truth, predicted) ESI pairs for successful Phase 1 (PAA) rows."""
    ok = [r for r in results if r.success and r.predicted_esi is not None]
    return [r.ground_truth_esi for r in ok], [r.predicted_esi for r in ok]


def phase2_esi_pairs(
    results: Sequence[EvaluationResult],
) -> Tuple[List[int], List[int]]:
    """(ground_truth, predicted) ESI pairs for successful Phase 2 (TCA) rows."""
    ok = [r for r in results if r.success and r.phase2_predicted_esi is not None]
    return [r.ground_truth_esi for r in ok], [r.phase2_predicted_esi for r in ok]
