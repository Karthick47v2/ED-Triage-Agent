"""Aggregate Phase 1 / Phase 2 evaluation summaries."""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Dict, List

from eval.schemas import EvaluationResult, EvaluationSummary
from eval.shared.metric_utils import (
    exact_and_within_one,
    latency_stats,
    per_class_accuracy,
    two_level_rates,
)

ESI_LABELS: List[int] = [1, 2, 3, 4, 5]


def _split_success(
    results: List[EvaluationResult], esi_attr: str
) -> List[EvaluationResult]:
    return [r for r in results if r.success and getattr(r, esi_attr) is not None]


def calculate_metrics(results: List[EvaluationResult]) -> EvaluationSummary:
    """Phase 1 aggregate metrics (PAA tentative ESI + HIGH/LOW priority)."""
    successful = _split_success(results, "predicted_esi")
    if not successful:
        raise ValueError("No successful evaluations to compute metrics from")

    y_true = [r.ground_truth_esi for r in successful]
    y_pred = [r.predicted_esi for r in successful]
    exact, within_one = exact_and_within_one(y_true, y_pred)
    under, over = two_level_rates(y_true, y_pred)

    priority_matches = sum(
        1 for r in successful if r.predicted_priority == r.ground_truth_priority
    )
    tp = sum(
        1
        for r in successful
        if r.ground_truth_priority == "HIGH" and r.predicted_priority == "HIGH"
    )
    fn = sum(
        1
        for r in successful
        if r.ground_truth_priority == "HIGH" and r.predicted_priority == "LOW"
    )
    tn = sum(
        1
        for r in successful
        if r.ground_truth_priority == "LOW" and r.predicted_priority == "LOW"
    )
    fp = sum(
        1
        for r in successful
        if r.ground_truth_priority == "LOW" and r.predicted_priority == "HIGH"
    )

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    mean_ms, median_ms, p95_ms = latency_stats(
        [r.phase1_latency_ms for r in successful]
    )

    return EvaluationSummary(
        total_scenarios=len(results),
        successful_runs=len(successful),
        failed_runs=len(results) - len(successful),
        esi_exact_accuracy=exact,
        esi_within_one_accuracy=within_one,
        priority_accuracy=priority_matches / len(successful),
        high_priority_sensitivity=sensitivity,
        high_priority_specificity=specificity,
        undertriage_rate=under,
        overtriage_rate=over,
        latency_mean_ms=mean_ms,
        latency_median_ms=median_ms,
        latency_p95_ms=p95_ms,
        per_esi_accuracy=per_class_accuracy(y_true, y_pred),
    )


@dataclass
class Phase2Summary:
    total_scenarios: int
    successful_runs: int
    failed_runs: int
    esi_exact_accuracy: float
    esi_within_one_accuracy: float
    high_acuity_sensitivity: float
    high_acuity_specificity: float
    undertriage_rate: float
    overtriage_rate: float
    latency_mean_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    per_esi_accuracy: Dict[int, float]
    mean_confidence: float


def calculate_phase2_metrics(results: List[EvaluationResult]) -> Phase2Summary:
    """Phase 2 aggregate metrics (TCA final ESI + ESI 1-2 detection)."""
    successful = _split_success(results, "phase2_predicted_esi")
    if not successful:
        raise ValueError("No successful Phase 2 evaluations to compute metrics from")

    y_true = [r.ground_truth_esi for r in successful]
    y_pred = [r.phase2_predicted_esi for r in successful]
    exact, within_one = exact_and_within_one(y_true, y_pred)
    under, over = two_level_rates(y_true, y_pred)

    tp = sum(1 for gt, pred in zip(y_true, y_pred) if gt <= 2 and pred <= 2)
    fn = sum(1 for gt, pred in zip(y_true, y_pred) if gt <= 2 and pred >= 3)
    tn = sum(1 for gt, pred in zip(y_true, y_pred) if gt >= 3 and pred >= 3)
    fp = sum(1 for gt, pred in zip(y_true, y_pred) if gt >= 3 and pred <= 2)
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    mean_ms, median_ms, p95_ms = latency_stats(
        [r.phase2_latency_ms for r in successful]
    )
    confidences = [
        r.phase2_confidence for r in successful if r.phase2_confidence is not None
    ]
    mean_confidence = statistics.mean(confidences) if confidences else 0.0

    return Phase2Summary(
        total_scenarios=len(results),
        successful_runs=len(successful),
        failed_runs=len(results) - len(successful),
        esi_exact_accuracy=exact,
        esi_within_one_accuracy=within_one,
        high_acuity_sensitivity=sensitivity,
        high_acuity_specificity=specificity,
        undertriage_rate=under,
        overtriage_rate=over,
        latency_mean_ms=mean_ms,
        latency_median_ms=median_ms,
        latency_p95_ms=p95_ms,
        per_esi_accuracy=per_class_accuracy(y_true, y_pred),
        mean_confidence=mean_confidence,
    )
