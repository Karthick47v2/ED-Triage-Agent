"""Wilson score confidence intervals for binomial proportion metrics."""
from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence, Tuple

from eval.schemas import EvaluationResult
from eval.shared.triage import phase1_esi_pairs, phase2_esi_pairs


def wilson_score_interval(
    successes: int,
    n: int,
    *,
    confidence: float = 0.95,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Wilson score interval for a binomial proportion.

    Returns (point, ci_low, ci_high). If n == 0, returns (None, None, None).
    """
    if n <= 0:
        return None, None, None
    if successes < 0 or successes > n:
        raise ValueError(f"successes must be in [0, n]; got {successes}, n={n}")

    # Normal critical value for two-sided CI (95% -> 1.959963...)
    # Keep dependency-free: approximate via erfinv via math for common levels,
    # otherwise fall back to 1.96.
    z = _z_from_confidence(confidence)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (
        z
        * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))
        / denom
    )
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return float(p), float(lo), float(hi)


def _z_from_confidence(confidence: float) -> float:
    """Two-sided normal critical value."""
    if abs(confidence - 0.95) < 1e-12:
        return 1.959963984540054
    if abs(confidence - 0.90) < 1e-12:
        return 1.6448536269514722
    if abs(confidence - 0.99) < 1e-12:
        return 2.5758293035489004
    # Fallback approximation via inverse erf
    alpha = 1.0 - confidence
    # P(|Z| <= z) = confidence => Phi(z) = 1 - alpha/2
    # z = sqrt(2) * erfinv(2p - 1)
    p = 1.0 - alpha / 2.0
    return math.sqrt(2.0) * _erfinv(2.0 * p - 1.0)


def _erfinv(x: float) -> float:
    """Approximate inverse error function (Winitzki)."""
    if x <= -1.0:
        return float("-inf")
    if x >= 1.0:
        return float("inf")
    a = 0.147
    sign = 1.0 if x >= 0 else -1.0
    ln = math.log(1.0 - x * x)
    first = 2.0 / (math.pi * a) + ln / 2.0
    return sign * math.sqrt(math.sqrt(first * first - ln / a) - first)


def wilson_metrics_phase1(
    results: Sequence[EvaluationResult],
    *,
    confidence: float = 0.95,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Wilson CIs for Phase 1 proportion metrics (not kappa/F1/AUC)."""
    ok = [r for r in results if r.success and r.predicted_esi is not None]
    n = len(ok)
    out: Dict[str, Dict[str, Optional[float]]] = {}
    if n == 0:
        return out

    y_true, y_pred = phase1_esi_pairs(ok)
    exact_k = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    within_k = sum(1 for t, p in zip(y_true, y_pred) if abs(t - p) <= 1)
    under_k = sum(1 for t, p in zip(y_true, y_pred) if p - t >= 2)
    over_k = sum(1 for t, p in zip(y_true, y_pred) if t - p >= 2)

    for key, k in (
        ("esi_exact_accuracy", exact_k),
        ("esi_within_one_accuracy", within_k),
        ("undertriage_rate", under_k),
        ("overtriage_rate", over_k),
    ):
        p, lo, hi = wilson_score_interval(k, n, confidence=confidence)
        out[key] = {"point": p, "ci_low": lo, "ci_high": hi, "n": float(n), "k": float(k)}

    pri_k = sum(1 for r in ok if r.predicted_priority == r.ground_truth_priority)
    p, lo, hi = wilson_score_interval(pri_k, n, confidence=confidence)
    out["priority_accuracy"] = {
        "point": p, "ci_low": lo, "ci_high": hi, "n": float(n), "k": float(pri_k)
    }

    tp = sum(
        1
        for r in ok
        if r.ground_truth_priority == "HIGH" and r.predicted_priority == "HIGH"
    )
    fn = sum(
        1
        for r in ok
        if r.ground_truth_priority == "HIGH" and r.predicted_priority == "LOW"
    )
    tn = sum(
        1
        for r in ok
        if r.ground_truth_priority == "LOW" and r.predicted_priority == "LOW"
    )
    fp = sum(
        1
        for r in ok
        if r.ground_truth_priority == "LOW" and r.predicted_priority == "HIGH"
    )
    n_pos, n_neg = tp + fn, tn + fp
    p, lo, hi = wilson_score_interval(tp, n_pos, confidence=confidence)
    out["high_priority_sensitivity"] = {
        "point": p, "ci_low": lo, "ci_high": hi, "n": float(n_pos), "k": float(tp)
    }
    p, lo, hi = wilson_score_interval(tn, n_neg, confidence=confidence)
    out["high_priority_specificity"] = {
        "point": p, "ci_low": lo, "ci_high": hi, "n": float(n_neg), "k": float(tn)
    }

    # TRIAGEAGENT proportions (same denominator n)
    ta_pairs = [
        ("ta_total_discordance_rate", sum(1 for t, p in zip(y_true, y_pred) if t != p)),
        ("ta_undertriage_rate", sum(1 for t, p in zip(y_true, y_pred) if p > t)),
        ("ta_overtriage_rate", sum(1 for t, p in zip(y_true, y_pred) if p < t)),
        (
            "ta_significant_undertriage_rate",
            sum(1 for t, p in zip(y_true, y_pred) if t in (1, 2) and p in (3, 4, 5)),
        ),
        (
            "ta_significant_overtriage_rate",
            sum(1 for t, p in zip(y_true, y_pred) if t in (2, 3, 4) and p == 1),
        ),
    ]
    for key, k in ta_pairs:
        p, lo, hi = wilson_score_interval(k, n, confidence=confidence)
        out[key] = {"point": p, "ci_low": lo, "ci_high": hi, "n": float(n), "k": float(k)}

    # Per-ESI accuracy (class-conditional)
    for esi in range(1, 6):
        idxs = [i for i, t in enumerate(y_true) if t == esi]
        n_c = len(idxs)
        k_c = sum(1 for i in idxs if y_pred[i] == esi)
        p, lo, hi = wilson_score_interval(k_c, n_c, confidence=confidence)
        out[f"per_esi_accuracy_esi_{esi}"] = {
            "point": p, "ci_low": lo, "ci_high": hi, "n": float(n_c), "k": float(k_c)
        }
    return out


def wilson_metrics_phase2(
    results: Sequence[EvaluationResult],
    *,
    confidence: float = 0.95,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Wilson CIs for Phase 2 proportion metrics (not kappa/F1)."""
    ok = [r for r in results if r.success and r.phase2_predicted_esi is not None]
    n = len(ok)
    out: Dict[str, Dict[str, Optional[float]]] = {}
    if n == 0:
        return out

    y_true, y_pred = phase2_esi_pairs(ok)
    exact_k = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    within_k = sum(1 for t, p in zip(y_true, y_pred) if abs(t - p) <= 1)
    under_k = sum(1 for t, p in zip(y_true, y_pred) if p - t >= 2)
    over_k = sum(1 for t, p in zip(y_true, y_pred) if t - p >= 2)

    for key, k in (
        ("esi_exact_accuracy", exact_k),
        ("esi_within_one_accuracy", within_k),
        ("undertriage_rate", under_k),
        ("overtriage_rate", over_k),
    ):
        p, lo, hi = wilson_score_interval(k, n, confidence=confidence)
        out[key] = {"point": p, "ci_low": lo, "ci_high": hi, "n": float(n), "k": float(k)}

    tp = sum(1 for t, p in zip(y_true, y_pred) if t <= 2 and p <= 2)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t <= 2 and p >= 3)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t >= 3 and p >= 3)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t >= 3 and p <= 2)
    n_pos, n_neg = tp + fn, tn + fp
    p, lo, hi = wilson_score_interval(tp, n_pos, confidence=confidence)
    out["high_acuity_sensitivity"] = {
        "point": p, "ci_low": lo, "ci_high": hi, "n": float(n_pos), "k": float(tp)
    }
    p, lo, hi = wilson_score_interval(tn, n_neg, confidence=confidence)
    out["high_acuity_specificity"] = {
        "point": p, "ci_low": lo, "ci_high": hi, "n": float(n_neg), "k": float(tn)
    }

    ta_pairs = [
        ("ta_total_discordance_rate", sum(1 for t, p in zip(y_true, y_pred) if t != p)),
        ("ta_undertriage_rate", sum(1 for t, p in zip(y_true, y_pred) if p > t)),
        ("ta_overtriage_rate", sum(1 for t, p in zip(y_true, y_pred) if p < t)),
        (
            "ta_significant_undertriage_rate",
            sum(1 for t, p in zip(y_true, y_pred) if t in (1, 2) and p in (3, 4, 5)),
        ),
        (
            "ta_significant_overtriage_rate",
            sum(1 for t, p in zip(y_true, y_pred) if t in (2, 3, 4) and p == 1),
        ),
    ]
    for key, k in ta_pairs:
        p, lo, hi = wilson_score_interval(k, n, confidence=confidence)
        out[key] = {"point": p, "ci_low": lo, "ci_high": hi, "n": float(n), "k": float(k)}

    for esi in range(1, 6):
        idxs = [i for i, t in enumerate(y_true) if t == esi]
        n_c = len(idxs)
        k_c = sum(1 for i in idxs if y_pred[i] == esi)
        p, lo, hi = wilson_score_interval(k_c, n_c, confidence=confidence)
        out[f"per_esi_accuracy_esi_{esi}"] = {
            "point": p, "ci_low": lo, "ci_high": hi, "n": float(n_c), "k": float(k_c)
        }
    return out


def format_wilson(
    entry: Mapping[str, Optional[float]],
    *,
    as_percent: bool = True,
    digits: int = 2,
) -> str:
    point = entry.get("point")
    lo = entry.get("ci_low")
    hi = entry.get("ci_high")
    if point is None:
        return "N/A"
    if as_percent:
        if lo is None or hi is None:
            return f"{100.0 * point:.{digits}f}%"
        return (
            f"{100.0 * point:.{digits}f}% "
            f"[{100.0 * lo:.{digits}f}%, {100.0 * hi:.{digits}f}%]"
        )
    if lo is None or hi is None:
        return f"{point:.4f}"
    return f"{point:.4f} [{lo:.4f}, {hi:.4f}]"
