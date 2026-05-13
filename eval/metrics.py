"""Unified metrics for ED-Triage-Agent evaluations.

Library + CLI. The CLI takes a single results file and a phase (1 or 2)
and prints every metric we report for that phase:

  - Aggregate summary (accuracy, sensitivity/specificity, latency, per-ESI).
  - Paper-style ordinal metrics (MAE, RMSE, Cohen's kappa, linear/quadratic
    weighted kappa, confusion matrix).
  - TRIAGEAGENT (Wang et al., EMNLP 2024) Appendix D.2 rates.
  - sklearn-backed metrics (quadratic-weighted kappa, per-class F1, and
    AUC-ROC for the binary HIGH/LOW priority task in Phase 1).

Examples (from repository root):

  python -m eval.metrics results/phase1_results.json --phase 1
  python -m eval.metrics results/phase2_results.json --phase 2
  python -m eval.metrics results/phase2_results.json --phase 2 \
      --heatmap figures/cm_phase2.png --json-out results/phase2_metrics.json

``scikit-learn`` and ``matplotlib`` are imported lazily inside the advanced
section so that the importable summary / paper / TRIAGEAGENT helpers stay
dependency-light.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from eval.schemas import EvaluationResult, EvaluationSummary
from eval.shared.io import load_evaluation_results
from eval.shared.metric_utils import (
    exact_and_within_one,
    latency_stats,
    per_class_accuracy,
    two_level_rates,
)
from eval.shared.triage import phase1_esi_pairs, phase2_esi_pairs

ESI_LABELS: List[int] = [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Aggregate summaries (Phase 1: PAA / Phase 2: TCA)
# ---------------------------------------------------------------------------

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
    tp = sum(1 for r in successful if r.ground_truth_priority == "HIGH" and r.predicted_priority == "HIGH")
    fn = sum(1 for r in successful if r.ground_truth_priority == "HIGH" and r.predicted_priority == "LOW")
    tn = sum(1 for r in successful if r.ground_truth_priority == "LOW" and r.predicted_priority == "LOW")
    fp = sum(1 for r in successful if r.ground_truth_priority == "LOW" and r.predicted_priority == "HIGH")

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
    confidences = [r.phase2_confidence for r in successful if r.phase2_confidence is not None]
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


# ---------------------------------------------------------------------------
# Pretty-printers for the aggregate summaries (used by run_evaluation.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# TRIAGEAGENT (Wang et al., EMNLP 2024 Findings) - Appendix D.2
# ---------------------------------------------------------------------------
# - Total discordance: predicted ESI != ground truth.
# - Undertriage:  predicted label *greater* than true label.
# - Overtriage:   predicted label *less* than true label.
# - Significant undertriage: true ESI in {1, 2}, predicted in {3, 4, 5}.
# - Significant overtriage:  true ESI in {2, 3, 4}, predicted == 1.

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


# ---------------------------------------------------------------------------
# Paper-aligned ordinal metrics (MAE / RMSE / Cohen's kappa / weighted kappa)
# ---------------------------------------------------------------------------

def _confusion_matrix(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> Dict[int, Dict[int, int]]:
    cm: Dict[int, Dict[int, int]] = {}
    for t, p in zip(y_true, y_pred):
        cm.setdefault(t, {})
        cm[t][p] = cm[t].get(p, 0) + 1
    return cm


def _mae_rmse(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> Tuple[float, float]:
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0
    mae = sum(abs(a - b) for a, b in zip(y_true, y_pred)) / n
    mse = sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / n
    return mae, math.sqrt(mse)


def cohen_kappa_multiclass(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> float:
    """Unweighted Cohen's kappa for nominal agreement (ESI 1-5)."""
    n = len(y_true)
    if n == 0:
        return 0.0
    labels = sorted(set(y_true) | set(y_pred))
    k = len(labels)
    idx = {c: i for i, c in enumerate(labels)}
    po = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n
    row = [0] * k
    col = [0] * k
    for t, p in zip(y_true, y_pred):
        row[idx[t]] += 1
        col[idx[p]] += 1
    pe = sum(row[i] * col[i] for i in range(k)) / (n * n)
    if math.isclose(1.0 - pe, 0.0):
        return 1.0 if math.isclose(po, 1.0) else 0.0
    return (po - pe) / (1.0 - pe)


def weighted_kappa(
    y_true: Sequence[int], y_pred: Sequence[int], *, kind: str
) -> float:
    """Weighted Cohen's kappa on the ordinal index of sorted observed labels.

    kind: 'linear' -> w_ij = |i-j| / (k-1); 'quadratic' -> w_ij = ((i-j)/(k-1))^2
    """
    n = len(y_true)
    if n == 0:
        return 0.0
    labels = sorted(set(y_true) | set(y_pred))
    k = len(labels)
    if k == 1:
        return 1.0
    ridx = {c: i for i, c in enumerate(labels)}

    o_mat = [[0.0] * k for _ in range(k)]
    for t, p in zip(y_true, y_pred):
        o_mat[ridx[t]][ridx[p]] += 1.0
    for i in range(k):
        for j in range(k):
            o_mat[i][j] /= n

    row_m = [sum(o_mat[i]) for i in range(k)]
    col_m = [sum(o_mat[i][j] for i in range(k)) for j in range(k)]
    e_mat = [[row_m[i] * col_m[j] for j in range(k)] for i in range(k)]

    def w(i: int, j: int) -> float:
        d = abs(i - j) / (k - 1)
        if kind == "linear":
            return d
        if kind == "quadratic":
            return d * d
        raise ValueError(f"unknown kind {kind!r}")

    num = sum(w(i, j) * o_mat[i][j] for i in range(k) for j in range(k))
    den = sum(w(i, j) * e_mat[i][j] for i in range(k) for j in range(k))
    if math.isclose(den, 0.0):
        return 1.0 if math.isclose(num, 0.0) else 0.0
    return 1.0 - num / den


def _ordinal_report(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> Dict[str, Any]:
    mae, rmse = _mae_rmse(y_true, y_pred)
    return {
        "esi_mae": mae,
        "esi_rmse": rmse,
        "cohen_kappa": cohen_kappa_multiclass(y_true, y_pred),
        "linear_weighted_kappa": weighted_kappa(y_true, y_pred, kind="linear"),
        "quadratic_weighted_kappa": weighted_kappa(y_true, y_pred, kind="quadratic"),
        "confusion_matrix_esi": _confusion_matrix(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Advanced metrics (sklearn-backed) + confusion-matrix heatmap
# ---------------------------------------------------------------------------

def _priority_pairs(
    results: List[EvaluationResult],
) -> Tuple[List[int], List[int], List[float]]:
    """Phase 1 HIGH/LOW priority pairs and confidence proxies for AUC-ROC."""
    ok = [r for r in results if r.success and r.predicted_priority is not None]
    y_true = [1 if r.ground_truth_priority == "HIGH" else 0 for r in ok]
    y_pred = [1 if r.predicted_priority == "HIGH" else 0 for r in ok]
    y_score = [
        r.confidence if r.confidence is not None else float(r.predicted_priority == "HIGH")
        for r in ok
    ]
    return y_true, y_pred, y_score


def _advanced_block(
    results: List[EvaluationResult],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    phase: int,
) -> Dict[str, Any]:
    """sklearn-backed metrics (lazy import keeps base helpers dep-free)."""
    from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

    block: Dict[str, Any] = {}
    block["quadratic_weighted_kappa_sklearn"] = (
        float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
        if len(y_true) >= 2 else float("nan")
    )

    if y_true:
        macro = float(
            f1_score(y_true, y_pred, labels=ESI_LABELS, average="macro", zero_division=0)
        )
        per_class = f1_score(
            y_true, y_pred, labels=ESI_LABELS, average=None, zero_division=0
        )
        block["macro_f1"] = macro
        block["per_class_f1"] = {
            f"ESI_{lbl}": float(v) for lbl, v in zip(ESI_LABELS, per_class)
        }
    else:
        block["macro_f1"] = float("nan")
        block["per_class_f1"] = {}

    if phase == 1:
        y_true_bin, _, y_score = _priority_pairs(results)
        block["auc_roc_priority"] = (
            float(roc_auc_score(y_true_bin, y_score))
            if len(set(y_true_bin)) >= 2 else float("nan")
        )
        block["priority_n"] = len(y_true_bin)
    return block


def _save_confusion_heatmap(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int],
    title: str,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(
            cm.sum(axis=1, keepdims=True) == 0,
            0,
            cm / cm.sum(axis=1, keepdims=True),
        )

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Recall (row-normalised)")

    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"ESI {lbl}" for lbl in labels], fontsize=11)
    ax.set_yticklabels([f"ESI {lbl}" for lbl in labels], fontsize=11)
    ax.set_xlabel("Predicted ESI", fontsize=12)
    ax.set_ylabel("True ESI", fontsize=12)
    ax.set_title(title, fontsize=13, pad=12)

    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=12, color=color, fontweight="bold")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\nConfusion matrix heatmap saved -> {output_path}")


# ---------------------------------------------------------------------------
# Phase reports (everything in one dict)
# ---------------------------------------------------------------------------

def report_phase1(results: List[EvaluationResult]) -> Dict[str, Any]:
    """All Phase 1 metrics: aggregate + ordinal + TRIAGEAGENT + sklearn."""
    base = calculate_metrics(results).model_dump()
    y_true, y_pred = phase1_esi_pairs(results)
    return {
        "phase": 1,
        **base,
        **_ordinal_report(y_true, y_pred),
        "triageagent": triageagent_metrics_from_results(results, phase=1),
        "advanced": _advanced_block(results, y_true, y_pred, phase=1),
    }


def report_phase2(results: List[EvaluationResult]) -> Dict[str, Any]:
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
        **_ordinal_report(y_true, y_pred),
        "triageagent": triageagent_metrics_from_results(results, phase=2),
        "advanced": _advanced_block(results, y_true, y_pred, phase=2),
    }


def _print_aggregate(report: Dict[str, Any]) -> None:
    phase = report["phase"]
    _print_section(f"PHASE {phase} - AGGREGATE")
    print(f"Scenarios: {report['successful_runs']}/{report['total_scenarios']} successful")
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
    _print_section(f"PHASE {report['phase']} - TRIAGEAGENT (Wang et al., 2024 Appendix D.2)")
    print(f"N: {ta.get('n', 0)}")
    print(f"Total discordance:        {ta.get('total_discordance_rate', 0):.4f}  ({ta.get('total_discordance_count', 0)} cases)")
    print(f"Undertriage:              {ta.get('undertriage_rate', 0):.4f}  ({ta.get('undertriage_count', 0)} cases)")
    print(f"Significant undertriage:  {ta.get('significant_undertriage_rate', 0):.4f}  ({ta.get('significant_undertriage_count', 0)} cases)")
    print(f"Overtriage:               {ta.get('overtriage_rate', 0):.4f}  ({ta.get('overtriage_count', 0)} cases)")
    print(f"Significant overtriage:   {ta.get('significant_overtriage_rate', 0):.4f}  ({ta.get('significant_overtriage_count', 0)} cases)")


def _print_advanced(report: Dict[str, Any]) -> None:
    adv = report.get("advanced", {}) or {}
    _print_section(f"PHASE {report['phase']} - ADVANCED (sklearn)")
    print(f"Quadratic-weighted kappa (sklearn):  {adv.get('quadratic_weighted_kappa_sklearn', float('nan')):.4f}")
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute every evaluation metric for a results file (Phase 1 or Phase 2)."
    )
    parser.add_argument("results_file", type=Path,
                        help="Path to phase1 / phase2 results JSON")
    parser.add_argument("--phase", type=int, choices=[1, 2], required=True,
                        help="Which phase the results file corresponds to")
    parser.add_argument("--heatmap", type=Path, default=None, metavar="FILE.png",
                        help="Also save a confusion-matrix heatmap to this path")
    parser.add_argument("--json-out", type=Path, default=None,
                        help="Also write the full report JSON to this path")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress human-readable output (useful with --json-out)")
    args = parser.parse_args()

    if not args.results_file.exists():
        sys.exit(f"error: file not found: {args.results_file}")

    results = load_evaluation_results(args.results_file)
    report = report_phase1(results) if args.phase == 1 else report_phase2(results)

    if not args.quiet:
        print(f"Source: {args.results_file}")
        print_full_report(report)

    if args.heatmap is not None:
        if args.phase == 1:
            y_true, y_pred = phase1_esi_pairs(results)
            label = "Phase 1 (PAA)"
        else:
            y_true, y_pred = phase2_esi_pairs(results)
            label = "Phase 2 (TCA)"
        if y_true:
            _save_confusion_heatmap(
                y_true, y_pred, ESI_LABELS,
                title=f"ESI Confusion Matrix - {label}  (n={len(y_true)})",
                output_path=args.heatmap,
            )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2))
        if not args.quiet:
            print(f"\nWrote {args.json_out}")
        elif args.json_out is None:
            pass


if __name__ == "__main__":
    main()
