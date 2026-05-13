"""Reusable building blocks for evaluation metrics."""
from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Dict, Iterable, Sequence, Tuple


def latency_stats(latencies: Sequence[float]) -> Tuple[float, float, float]:
    """Return (mean, median, p95). Zero-valued latencies are ignored.

    P95 falls back to the single available sample when only one positive
    measurement is provided (matches the original metrics behaviour).
    """
    positive = [ms for ms in latencies if ms > 0]
    if not positive:
        return 0.0, 0.0, 0.0
    mean = statistics.mean(positive)
    median = statistics.median(positive)
    if len(positive) > 1:
        p95 = sorted(positive)[int(len(positive) * 0.95)]
    else:
        p95 = positive[0]
    return mean, median, p95


def per_class_accuracy(
    ground_truth: Iterable[int], predicted: Iterable[int]
) -> Dict[int, float]:
    """Accuracy bucketed by ground-truth class."""
    per: Dict[int, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    for gt, pred in zip(ground_truth, predicted):
        per[gt]["total"] += 1
        if pred == gt:
            per[gt]["correct"] += 1
    return {
        cls: (data["correct"] / data["total"]) if data["total"] else 0.0
        for cls, data in sorted(per.items())
    }


def exact_and_within_one(
    ground_truth: Sequence[int], predicted: Sequence[int]
) -> Tuple[float, float]:
    """Return (exact-match, within-1) accuracy fractions over the provided rows."""
    n = len(ground_truth)
    if n == 0:
        return 0.0, 0.0
    exact = sum(1 for gt, pred in zip(ground_truth, predicted) if gt == pred) / n
    near = sum(1 for gt, pred in zip(ground_truth, predicted) if abs(gt - pred) <= 1) / n
    return exact, near


def two_level_rates(
    ground_truth: Sequence[int], predicted: Sequence[int]
) -> Tuple[float, float]:
    """Undertriage / overtriage rates using a >=2 ESI-level threshold."""
    n = len(ground_truth)
    if n == 0:
        return 0.0, 0.0
    under = sum(1 for gt, pred in zip(ground_truth, predicted) if pred - gt >= 2) / n
    over = sum(1 for gt, pred in zip(ground_truth, predicted) if gt - pred >= 2) / n
    return under, over
