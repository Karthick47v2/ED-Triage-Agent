"""
Metrics calculation for Phase 1 and Phase 2 evaluation.
"""
import json
import statistics
from typing import List, Dict, Any, Optional
from collections import defaultdict
from dataclasses import dataclass

from eval.schemas import EvaluationResult, EvaluationSummary


def calculate_metrics(results: List[EvaluationResult]) -> EvaluationSummary:
    """
    Calculate aggregate metrics from Phase 1 evaluation results.
    
    Args:
        results: List of individual evaluation results
        
    Returns:
        EvaluationSummary with all computed metrics
    """
    successful = [r for r in results if r.success and r.predicted_esi is not None]
    failed = [r for r in results if not r.success or r.predicted_esi is None]
    
    if not successful:
        raise ValueError("No successful evaluations to compute metrics from")
    
    # ESI Accuracy
    esi_exact_matches = sum(1 for r in successful if r.predicted_esi == r.ground_truth_esi)
    esi_within_one = sum(1 for r in successful if abs(r.predicted_esi - r.ground_truth_esi) <= 1)
    
    esi_exact_accuracy = esi_exact_matches / len(successful)
    esi_within_one_accuracy = esi_within_one / len(successful)
    
    # Priority Classification (HIGH = ESI 1-3, LOW = ESI 4-5)
    priority_matches = sum(1 for r in successful if r.predicted_priority == r.ground_truth_priority)
    priority_accuracy = priority_matches / len(successful)
    
    # Sensitivity & Specificity for HIGH priority
    tp = sum(1 for r in successful if r.ground_truth_priority == "HIGH" and r.predicted_priority == "HIGH")
    fn = sum(1 for r in successful if r.ground_truth_priority == "HIGH" and r.predicted_priority == "LOW")
    tn = sum(1 for r in successful if r.ground_truth_priority == "LOW" and r.predicted_priority == "LOW")
    fp = sum(1 for r in successful if r.ground_truth_priority == "LOW" and r.predicted_priority == "HIGH")
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Undertriage & Overtriage (≥2 levels difference)
    undertriage_count = sum(1 for r in successful if r.predicted_esi - r.ground_truth_esi >= 2)
    overtriage_count = sum(1 for r in successful if r.ground_truth_esi - r.predicted_esi >= 2)
    
    undertriage_rate = undertriage_count / len(successful)
    overtriage_rate = overtriage_count / len(successful)
    
    # Latency statistics
    latencies = [r.phase1_latency_ms for r in successful if r.phase1_latency_ms > 0]
    if latencies:
        latency_mean = statistics.mean(latencies)
        latency_median = statistics.median(latencies)
        latency_p95 = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0]
    else:
        latency_mean = latency_median = latency_p95 = 0.0
    
    # Per-ESI accuracy breakdown
    per_esi = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in successful:
        per_esi[r.ground_truth_esi]["total"] += 1
        if r.predicted_esi == r.ground_truth_esi:
            per_esi[r.ground_truth_esi]["correct"] += 1
    
    per_esi_accuracy = {
        esi: data["correct"] / data["total"] if data["total"] > 0 else 0.0
        for esi, data in sorted(per_esi.items())
    }
    
    return EvaluationSummary(
        total_scenarios=len(results),
        successful_runs=len(successful),
        failed_runs=len(failed),
        esi_exact_accuracy=esi_exact_accuracy,
        esi_within_one_accuracy=esi_within_one_accuracy,
        priority_accuracy=priority_accuracy,
        high_priority_sensitivity=sensitivity,
        high_priority_specificity=specificity,
        undertriage_rate=undertriage_rate,
        overtriage_rate=overtriage_rate,
        latency_mean_ms=latency_mean,
        latency_median_ms=latency_median,
        latency_p95_ms=latency_p95,
        per_esi_accuracy=per_esi_accuracy
    )


@dataclass
class Phase2Summary:
    """Aggregate metrics for Phase 2 evaluation."""
    total_scenarios: int
    successful_runs: int
    failed_runs: int
    
    # ESI Accuracy
    esi_exact_accuracy: float
    esi_within_one_accuracy: float
    
    # High-Acuity Detection (ESI 1-2)
    high_acuity_sensitivity: float  # True positive rate for ESI 1-2
    high_acuity_specificity: float  # True negative rate for ESI 3-5
    
    # Triage Safety
    undertriage_rate: float
    overtriage_rate: float
    
    # Latency
    latency_mean_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    
    # Per-ESI breakdown
    per_esi_accuracy: Dict[int, float]
    
    # Confidence calibration
    mean_confidence: float


def calculate_phase2_metrics(results: List[EvaluationResult]) -> Phase2Summary:
    """
    Calculate aggregate metrics from Phase 2 evaluation results.
    
    Args:
        results: List of evaluation results with phase2_predicted_esi
        
    Returns:
        Phase2Summary with all computed metrics
    """
    successful = [r for r in results if r.success and r.phase2_predicted_esi is not None]
    failed = [r for r in results if not r.success or r.phase2_predicted_esi is None]
    
    if not successful:
        raise ValueError("No successful Phase 2 evaluations to compute metrics from")
    
    # ESI Accuracy
    esi_exact_matches = sum(1 for r in successful if r.phase2_predicted_esi == r.ground_truth_esi)
    esi_within_one = sum(1 for r in successful if abs(r.phase2_predicted_esi - r.ground_truth_esi) <= 1)
    
    esi_exact_accuracy = esi_exact_matches / len(successful)
    esi_within_one_accuracy = esi_within_one / len(successful)
    
    # High-Acuity Detection (ESI 1-2)
    # Sensitivity: Of actual ESI 1-2, how many did we correctly identify as ESI 1-2?
    # Specificity: Of actual ESI 3-5, how many did we correctly identify as ESI 3-5?
    actual_high_acuity = [r for r in successful if r.ground_truth_esi <= 2]
    actual_low_acuity = [r for r in successful if r.ground_truth_esi >= 3]
    
    tp = sum(1 for r in actual_high_acuity if r.phase2_predicted_esi <= 2)
    fn = sum(1 for r in actual_high_acuity if r.phase2_predicted_esi >= 3)
    tn = sum(1 for r in actual_low_acuity if r.phase2_predicted_esi >= 3)
    fp = sum(1 for r in actual_low_acuity if r.phase2_predicted_esi <= 2)
    
    high_acuity_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    high_acuity_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Undertriage & Overtriage (≥2 levels difference)
    undertriage_count = sum(1 for r in successful if r.phase2_predicted_esi - r.ground_truth_esi >= 2)
    overtriage_count = sum(1 for r in successful if r.ground_truth_esi - r.phase2_predicted_esi >= 2)
    
    undertriage_rate = undertriage_count / len(successful)
    overtriage_rate = overtriage_count / len(successful)
    
    # Latency statistics
    latencies = [r.phase2_latency_ms for r in successful if r.phase2_latency_ms > 0]
    if latencies:
        latency_mean = statistics.mean(latencies)
        latency_median = statistics.median(latencies)
        latency_p95 = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0]
    else:
        latency_mean = latency_median = latency_p95 = 0.0
    
    # Per-ESI accuracy breakdown
    per_esi = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in successful:
        per_esi[r.ground_truth_esi]["total"] += 1
        if r.phase2_predicted_esi == r.ground_truth_esi:
            per_esi[r.ground_truth_esi]["correct"] += 1
    
    per_esi_accuracy = {
        esi: data["correct"] / data["total"] if data["total"] > 0 else 0.0
        for esi, data in sorted(per_esi.items())
    }
    
    # Mean confidence
    confidences = [r.phase2_confidence for r in successful if r.phase2_confidence is not None]
    mean_confidence = statistics.mean(confidences) if confidences else 0.0
    
    return Phase2Summary(
        total_scenarios=len(results),
        successful_runs=len(successful),
        failed_runs=len(failed),
        esi_exact_accuracy=esi_exact_accuracy,
        esi_within_one_accuracy=esi_within_one_accuracy,
        high_acuity_sensitivity=high_acuity_sensitivity,
        high_acuity_specificity=high_acuity_specificity,
        undertriage_rate=undertriage_rate,
        overtriage_rate=overtriage_rate,
        latency_mean_ms=latency_mean,
        latency_median_ms=latency_median,
        latency_p95_ms=latency_p95,
        per_esi_accuracy=per_esi_accuracy,
        mean_confidence=mean_confidence
    )


def print_summary(summary: EvaluationSummary):
    """Print formatted Phase 1 metrics summary."""
    print("\n" + "=" * 60)
    print("         PHASE 1 EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\nScenarios: {summary.successful_runs}/{summary.total_scenarios} successful")
    if summary.failed_runs > 0:
        print(f"  ⚠️  {summary.failed_runs} failed runs")
    
    print("\n--- ESI Classification ---")
    print(f"  Exact Match:    {summary.esi_exact_accuracy:.1%}")
    print(f"  Within ±1:      {summary.esi_within_one_accuracy:.1%}")
    
    print("\n--- Priority Classification (HIGH vs LOW) ---")
    print(f"  Accuracy:       {summary.priority_accuracy:.1%}")
    print(f"  Sensitivity:    {summary.high_priority_sensitivity:.1%}")
    print(f"  Specificity:    {summary.high_priority_specificity:.1%}")
    
    print("\n--- Triage Safety ---")
    print(f"  Undertriage:    {summary.undertriage_rate:.1%}")
    print(f"  Overtriage:     {summary.overtriage_rate:.1%}")
    
    print("\n--- Latency (ms) ---")
    print(f"  Mean:           {summary.latency_mean_ms:.0f}")
    print(f"  Median:         {summary.latency_median_ms:.0f}")
    print(f"  P95:            {summary.latency_p95_ms:.0f}")
    
    if summary.per_esi_accuracy:
        print("\n--- Per-ESI Accuracy ---")
        for esi, acc in summary.per_esi_accuracy.items():
            print(f"  ESI {esi}:         {acc:.1%}")
    
    print("\n" + "=" * 60)


def print_phase2_summary(summary: Phase2Summary):
    """Print formatted Phase 2 metrics summary."""
    print("\n" + "=" * 60)
    print("         PHASE 2 EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\nScenarios: {summary.successful_runs}/{summary.total_scenarios} successful")
    if summary.failed_runs > 0:
        print(f"  ⚠️  {summary.failed_runs} failed runs")
    
    print("\n--- ESI Classification ---")
    print(f"  Exact Match:    {summary.esi_exact_accuracy:.1%}")
    print(f"  Within ±1:      {summary.esi_within_one_accuracy:.1%}")
    
    print("\n--- High-Acuity Detection (ESI 1-2) ---")
    print(f"  Sensitivity:    {summary.high_acuity_sensitivity:.1%}")
    print(f"  Specificity:    {summary.high_acuity_specificity:.1%}")
    
    print("\n--- Triage Safety ---")
    print(f"  Undertriage:    {summary.undertriage_rate:.1%}")
    print(f"  Overtriage:     {summary.overtriage_rate:.1%}")
    
    print("\n--- Latency (ms) ---")
    print(f"  Mean:           {summary.latency_mean_ms:.0f}")
    print(f"  Median:         {summary.latency_median_ms:.0f}")
    print(f"  P95:            {summary.latency_p95_ms:.0f}")
    
    print(f"\n--- Confidence ---")
    print(f"  Mean:           {summary.mean_confidence:.2f}")
    
    if summary.per_esi_accuracy:
        print("\n--- Per-ESI Accuracy ---")
        for esi, acc in summary.per_esi_accuracy.items():
            print(f"  ESI {esi}:         {acc:.1%}")
    
    print("\n" + "=" * 60)


# CLI usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate evaluation metrics")
    parser.add_argument("results_file", help="Path to results JSON file")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=1, help="Phase (1 or 2)")
    args = parser.parse_args()
    
    with open(args.results_file, "r") as f:
        data = json.load(f)
    
    results = [EvaluationResult(**r) for r in data["results"]]
    
    if args.phase == 1:
        summary = calculate_metrics(results)
        print_summary(summary)
        print("\n--- JSON Summary ---")
        print(summary.model_dump_json(indent=2))
    else:
        summary = calculate_phase2_metrics(results)
        print_phase2_summary(summary)

