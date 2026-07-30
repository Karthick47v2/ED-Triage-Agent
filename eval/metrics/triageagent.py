"""TRIAGEAGENT (Wang et al., EMNLP 2024) Appendix D.2 rates."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from eval.schemas import EvaluationResult
from eval.shared.triage import phase1_esi_pairs, phase2_esi_pairs

_TA_EMPTY: Dict[str, Any] = {
    "n": 0,
    "total_discordance_count": 0,
    "total_discordance_rate": 0.0,
    "undertriage_count": 0,
    "undertriage_rate": 0.0,
    "overtriage_count": 0,
    "overtriage_rate": 0.0,
    "significant_undertriage_count": 0,
    "significant_undertriage_rate": 0.0,
    "significant_overtriage_count": 0,
    "significant_overtriage_rate": 0.0,
}


def compute_triageagent_metrics(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> Dict[str, Any]:
    """Counts and rates for each TRIAGEAGENT metric. Empty input returns zeros."""
    n = len(y_true)
    if n == 0 or len(y_pred) != n:
        return dict(_TA_EMPTY)

    total_mis = sum(1 for t, p in zip(y_true, y_pred) if t != p)
    under = sum(1 for t, p in zip(y_true, y_pred) if p > t)
    over = sum(1 for t, p in zip(y_true, y_pred) if p < t)
    sig_under = sum(
        1 for t, p in zip(y_true, y_pred) if t in (1, 2) and p in (3, 4, 5)
    )
    sig_over = sum(
        1 for t, p in zip(y_true, y_pred) if t in (2, 3, 4) and p == 1
    )

    return {
        "n": n,
        "total_discordance_count": total_mis,
        "total_discordance_rate": total_mis / n,
        "undertriage_count": under,
        "undertriage_rate": under / n,
        "overtriage_count": over,
        "overtriage_rate": over / n,
        "significant_undertriage_count": sig_under,
        "significant_undertriage_rate": sig_under / n,
        "significant_overtriage_count": sig_over,
        "significant_overtriage_rate": sig_over / n,
    }


def triageagent_metrics_from_results(
    results: List[EvaluationResult], *, phase: int
) -> Dict[str, Any]:
    if phase == 1:
        y_true, y_pred = phase1_esi_pairs(results)
    elif phase == 2:
        y_true, y_pred = phase2_esi_pairs(results)
    else:
        raise ValueError("phase must be 1 or 2")
    return compute_triageagent_metrics(y_true, y_pred)
