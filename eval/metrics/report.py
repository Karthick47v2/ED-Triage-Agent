"""Phase reports and human-readable printers."""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

from eval.metrics.aggregate import (
    ESI_LABELS,
    calculate_metrics,
    calculate_phase2_metrics,
)
from eval.metrics.advanced import advanced_block
from eval.metrics.ordinal import ordinal_report
from eval.metrics.triageagent import triageagent_metrics_from_results
from eval.schemas import EvaluationResult, EvaluationSummary
from eval.shared.triage import phase1_esi_pairs, phase2_esi_pairs


def _print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"         {title}")
    print("=" * 60)


def _print_latency(summary) -> None:
    print("\n--- Latency (ms) ---")
    print(f"  Mean:           {summary.latency_mean_ms:.0f}")
    print(f"  Median:         {summary.latency_median_ms:.0f}")
    print(f"  P95:            {summary.latency_p95_ms:.0f}")


def _print_per_esi(per_esi: Optional[Dict[int, float]]) -> None:
    if not per_esi:
        return
    print("\n--- Per-ESI Accuracy ---")
    for esi, acc in per_esi.items():
        print(f"  ESI {esi}:         {acc:.1%}")


def print_summary(summary: EvaluationSummary) -> None:
    _print_section("PHASE 1 EVALUATION RESULTS")
    print(f"\nScenarios: {summary.successful_runs}/{summary.total_scenarios} successful")
    if summary.failed_runs > 0:
        print(f"  {summary.failed_runs} failed runs")
    print("\n--- ESI Classification ---")
    print(f"  Exact Match:    {summary.esi_exact_accuracy:.1%}")
    print(f"  Within +/-1:    {summary.esi_within_one_accuracy:.1%}")
    print("\n--- Priority Classification (HIGH vs LOW) ---")
    print(f"  Accuracy:       {summary.priority_accuracy:.1%}")
    print(f"  Sensitivity:    {summary.high_priority_sensitivity:.1%}")
    print(f"  Specificity:    {summary.high_priority_specificity:.1%}")
    print("\n--- Triage Safety ---")
    print(f"  Undertriage:    {summary.undertriage_rate:.1%}")
    print(f"  Overtriage:     {summary.overtriage_rate:.1%}")
    _print_latency(summary)
    _print_per_esi(summary.per_esi_accuracy)
    print("\n" + "=" * 60)


def report_phase1(results: list[EvaluationResult]) -> Dict[str, Any]:
    """All Phase 1 metrics: aggregate + ordinal + TRIAGEAGENT + sklearn."""
    base = calculate_metrics(results).model_dump()
    y_true, y_pred = phase1_esi_pairs(results)
    return {
        "phase": 1,
        **base,
        **ordinal_report(y_true, y_pred),
        "triageagent": triageagent_metrics_from_results(results, phase=1),
        "advanced": advanced_block(results, y_true, y_pred, phase=1),
    }


def report_phase2(results: list[EvaluationResult]) -> Dict[str, Any]:
    """All Phase 2 metrics: aggregate + ordinal + TRIAGEAGENT + sklearn."""
    s2 = calculate_phase2_metrics(results)
    base = {
        "total_scenarios": s2.total_scenarios,
        "successful_runs": s2.successful_runs,
        "failed_runs": s2.failed_runs,
        "esi_exact_accuracy": s2.esi_exact_accuracy,
        "esi_within_one_accuracy": s2.esi_within_one_accuracy,
        "high_acuity_sensitivity": s2.high_acuity_sensitivity,
        "high_acuity_specificity": s2.high_acuity_specificity,
        "undertriage_rate": s2.undertriage_rate,
        "overtriage_rate": s2.overtriage_rate,
        "latency_mean_ms": s2.latency_mean_ms,
        "latency_median_ms": s2.latency_median_ms,
        "latency_p95_ms": s2.latency_p95_ms,
        "per_esi_accuracy": s2.per_esi_accuracy,
        "mean_confidence": s2.mean_confidence,
    }
    y_true, y_pred = phase2_esi_pairs(results)
    return {
        "phase": 2,
        **base,
        **ordinal_report(y_true, y_pred),
        "triageagent": triageagent_metrics_from_results(results, phase=2),
        "advanced": advanced_block(results, y_true, y_pred, phase=2),
    }


def _print_aggregate(report: Dict[str, Any]) -> None:
    phase = report["phase"]
    _print_section(f"PHASE {phase} - AGGREGATE")
    print(
        f"Scenarios: {report['successful_runs']}/{report['total_scenarios']} successful"
    )
    if report["failed_runs"] > 0:
        print(f"  {report['failed_runs']} failed runs")
    print("\nESI Classification")
    print(f"  Exact Match:    {report['esi_exact_accuracy']:.1%}")
    print(f"  Within +/-1:    {report['esi_within_one_accuracy']:.1%}")
    if phase == 1:
        print("\nPriority Classification (HIGH vs LOW)")
        print(f"  Accuracy:       {report['priority_accuracy']:.1%}")
        print(f"  Sensitivity:    {report['high_priority_sensitivity']:.1%}")
        print(f"  Specificity:    {report['high_priority_specificity']:.1%}")
    else:
        print("\nHigh-Acuity Detection (ESI 1-2)")
        print(f"  Sensitivity:    {report['high_acuity_sensitivity']:.1%}")
        print(f"  Specificity:    {report['high_acuity_specificity']:.1%}")
        print(f"\nMean confidence:  {report['mean_confidence']:.2f}")
    print("\nTriage Safety (>=2 ESI levels)")
    print(f"  Undertriage:    {report['undertriage_rate']:.1%}")
    print(f"  Overtriage:     {report['overtriage_rate']:.1%}")
    print("\nLatency (ms)")
    print(f"  Mean:           {report['latency_mean_ms']:.0f}")
    print(f"  Median:         {report['latency_median_ms']:.0f}")
    print(f"  P95:            {report['latency_p95_ms']:.0f}")
    per_esi = report.get("per_esi_accuracy") or {}
    if per_esi:
        print("\nPer-ESI Accuracy")
        for esi, acc in per_esi.items():
            print(f"  ESI {esi}:         {acc:.1%}")


def _print_ordinal(report: Dict[str, Any]) -> None:
    _print_section(f"PHASE {report['phase']} - ORDINAL / KAPPA")
    print(f"ESI MAE:                  {report.get('esi_mae', 0):.4f}")
    print(f"ESI RMSE:                 {report.get('esi_rmse', 0):.4f}")
    print(f"Cohen's kappa:            {report.get('cohen_kappa', 0):.4f}")
    print(f"Linear weighted kappa:    {report.get('linear_weighted_kappa', 0):.4f}")
    print(f"Quadratic weighted kappa: {report.get('quadratic_weighted_kappa', 0):.4f}")
    cm = report.get("confusion_matrix_esi", {})
    if cm:
        print("\nConfusion matrix (rows = true ESI, cols = predicted ESI)")
        header = "      " + "  ".join(f"{c:>3d}" for c in ESI_LABELS)
        print(header)
        for true_esi in ESI_LABELS:
            row = cm.get(true_esi, {})
            cells = "  ".join(f"{row.get(p, 0):>3d}" for p in ESI_LABELS)
            print(f"  {true_esi}:  {cells}")


def _print_triageagent(report: Dict[str, Any]) -> None:
    ta = report.get("triageagent", {}) or {}
    _print_section(
        f"PHASE {report['phase']} - TRIAGEAGENT (Wang et al., 2024 Appendix D.2)"
    )
    print(f"N: {ta.get('n', 0)}")
    print(
        f"Total discordance:        {ta.get('total_discordance_rate', 0):.4f}  "
        f"({ta.get('total_discordance_count', 0)} cases)"
    )
    print(
        f"Undertriage:              {ta.get('undertriage_rate', 0):.4f}  "
        f"({ta.get('undertriage_count', 0)} cases)"
    )
    print(
        f"Significant undertriage:  {ta.get('significant_undertriage_rate', 0):.4f}  "
        f"({ta.get('significant_undertriage_count', 0)} cases)"
    )
    print(
        f"Overtriage:               {ta.get('overtriage_rate', 0):.4f}  "
        f"({ta.get('overtriage_count', 0)} cases)"
    )
    print(
        f"Significant overtriage:   {ta.get('significant_overtriage_rate', 0):.4f}  "
        f"({ta.get('significant_overtriage_count', 0)} cases)"
    )


def _print_advanced(report: Dict[str, Any]) -> None:
    adv = report.get("advanced", {}) or {}
    _print_section(f"PHASE {report['phase']} - ADVANCED (sklearn)")
    print(
        f"Quadratic-weighted kappa (sklearn):  "
        f"{adv.get('quadratic_weighted_kappa_sklearn', float('nan')):.4f}"
    )
    print(f"Macro F1 (ESI 1-5):                  {adv.get('macro_f1', float('nan')):.4f}")
    per_class = adv.get("per_class_f1") or {}
    if per_class:
        print("\nPer-class F1")
        for cls, val in sorted(per_class.items()):
            print(f"  {cls}:  {val:.4f}")
    if report["phase"] == 1:
        auc = adv.get("auc_roc_priority", float("nan"))
        n = adv.get("priority_n", 0)
        auc_str = (
            f"{auc:.4f}"
            if auc is not None and not (isinstance(auc, float) and math.isnan(auc))
            else "N/A"
        )
        print(f"\nAUC-ROC (HIGH vs LOW priority, n={n}):  {auc_str}")


def print_full_report(report: Dict[str, Any]) -> None:
    """Print every metric section for a single phase."""
    _print_aggregate(report)
    _print_ordinal(report)
    _print_triageagent(report)
    _print_advanced(report)
    print("\n" + "=" * 60)
