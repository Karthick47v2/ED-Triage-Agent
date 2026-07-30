"""Case-level bootstrap confidence intervals for evaluation metrics."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from eval.metrics.aggregate import ESI_LABELS
from eval.metrics.ordinal import ordinal_report
from eval.metrics.triageagent import compute_triageagent_metrics
from eval.schemas import EvaluationResult
from eval.shared.metric_utils import (
    exact_and_within_one,
    per_class_accuracy,
    two_level_rates,
)
from eval.shared.triage import phase1_esi_pairs, phase2_esi_pairs


def _oriented_auc(results: Sequence[EvaluationResult]) -> float:
    """AUC-ROC with confidence oriented toward HIGH priority."""
    from sklearn.metrics import roc_auc_score

    ok = [r for r in results if r.success and r.predicted_priority is not None]
    if len(ok) < 2:
        return float("nan")
    y_true = [1 if r.ground_truth_priority == "HIGH" else 0 for r in ok]
    if len(set(y_true)) < 2:
        return float("nan")
    y_score = []
    for r in ok:
        conf = r.confidence if r.confidence is not None else 1.0
        if r.predicted_priority == "HIGH":
            y_score.append(float(conf))
        else:
            y_score.append(float(1.0 - conf))
    return float(roc_auc_score(y_true, y_score))


def _raw_auc(results: Sequence[EvaluationResult]) -> float:
    from sklearn.metrics import roc_auc_score

    ok = [r for r in results if r.success and r.predicted_priority is not None]
    if len(ok) < 2:
        return float("nan")
    y_true = [1 if r.ground_truth_priority == "HIGH" else 0 for r in ok]
    if len(set(y_true)) < 2:
        return float("nan")
    y_score = [
        float(r.confidence)
        if r.confidence is not None
        else float(r.predicted_priority == "HIGH")
        for r in ok
    ]
    return float(roc_auc_score(y_true, y_score))


def _macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    from sklearn.metrics import f1_score

    if not y_true:
        return float("nan")
    return float(
        f1_score(y_true, y_pred, labels=ESI_LABELS, average="macro", zero_division=0)
    )


def scalar_metrics_phase1(results: Sequence[EvaluationResult]) -> Dict[str, float]:
    """Point metrics used for Phase 1 bootstrap / paper-style tables."""
    ok = [r for r in results if r.success and r.predicted_esi is not None]
    if not ok:
        raise ValueError("No successful Phase 1 rows")
    y_true, y_pred = phase1_esi_pairs(ok)
    exact, within = exact_and_within_one(y_true, y_pred)
    under, over = two_level_rates(y_true, y_pred)
    ordinal = ordinal_report(y_true, y_pred)
    ta = compute_triageagent_metrics(y_true, y_pred)

    priority_matches = sum(
        1 for r in ok if r.predicted_priority == r.ground_truth_priority
    )
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

    return {
        "esi_exact_accuracy": float(exact),
        "esi_within_one_accuracy": float(within),
        "quadratic_weighted_kappa": float(ordinal["quadratic_weighted_kappa"]),
        "macro_f1": _macro_f1(y_true, y_pred),
        "priority_accuracy": float(priority_matches / len(ok)),
        "high_priority_sensitivity": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "high_priority_specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "auc_roc_priority_raw": _raw_auc(ok),
        "auc_roc_priority": _oriented_auc(ok),
        "undertriage_rate": float(under),
        "overtriage_rate": float(over),
        "ta_total_discordance_rate": float(ta["total_discordance_rate"]),
        "ta_undertriage_rate": float(ta["undertriage_rate"]),
        "ta_overtriage_rate": float(ta["overtriage_rate"]),
        "ta_significant_undertriage_rate": float(ta["significant_undertriage_rate"]),
        "ta_significant_overtriage_rate": float(ta["significant_overtriage_rate"]),
        "esi_mae": float(ordinal["esi_mae"]),
        "esi_rmse": float(ordinal["esi_rmse"]),
        "cohen_kappa": float(ordinal["cohen_kappa"]),
        "linear_weighted_kappa": float(ordinal["linear_weighted_kappa"]),
        **{
            f"per_esi_accuracy_esi_{k}": float(v)
            for k, v in per_class_accuracy(y_true, y_pred).items()
        },
    }


def scalar_metrics_phase2(results: Sequence[EvaluationResult]) -> Dict[str, float]:
    """Point metrics used for Phase 2 bootstrap / paper-style tables."""
    ok = [r for r in results if r.success and r.phase2_predicted_esi is not None]
    if not ok:
        raise ValueError("No successful Phase 2 rows")
    y_true, y_pred = phase2_esi_pairs(ok)
    exact, within = exact_and_within_one(y_true, y_pred)
    under, over = two_level_rates(y_true, y_pred)
    ordinal = ordinal_report(y_true, y_pred)
    ta = compute_triageagent_metrics(y_true, y_pred)

    tp = sum(1 for gt, pred in zip(y_true, y_pred) if gt <= 2 and pred <= 2)
    fn = sum(1 for gt, pred in zip(y_true, y_pred) if gt <= 2 and pred >= 3)
    tn = sum(1 for gt, pred in zip(y_true, y_pred) if gt >= 3 and pred >= 3)
    fp = sum(1 for gt, pred in zip(y_true, y_pred) if gt >= 3 and pred <= 2)
    confs = [r.phase2_confidence for r in ok if r.phase2_confidence is not None]
    mean_conf = float(np.mean(confs)) if confs else float("nan")

    return {
        "esi_exact_accuracy": float(exact),
        "esi_within_one_accuracy": float(within),
        "quadratic_weighted_kappa": float(ordinal["quadratic_weighted_kappa"]),
        "macro_f1": _macro_f1(y_true, y_pred),
        "high_acuity_sensitivity": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "high_acuity_specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "undertriage_rate": float(under),
        "overtriage_rate": float(over),
        "mean_confidence": mean_conf,
        "ta_total_discordance_rate": float(ta["total_discordance_rate"]),
        "ta_undertriage_rate": float(ta["undertriage_rate"]),
        "ta_overtriage_rate": float(ta["overtriage_rate"]),
        "ta_significant_undertriage_rate": float(ta["significant_undertriage_rate"]),
        "ta_significant_overtriage_rate": float(ta["significant_overtriage_rate"]),
        "esi_mae": float(ordinal["esi_mae"]),
        "esi_rmse": float(ordinal["esi_rmse"]),
        "cohen_kappa": float(ordinal["cohen_kappa"]),
        "linear_weighted_kappa": float(ordinal["linear_weighted_kappa"]),
        **{
            f"per_esi_accuracy_esi_{k}": float(v)
            for k, v in per_class_accuracy(y_true, y_pred).items()
        },
    }


def bootstrap_metrics(
    results: Sequence[EvaluationResult],
    metric_fn: Callable[[Sequence[EvaluationResult]], Mapping[str, float]],
    *,
    n_boot: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Percentile bootstrap CIs over case-level resamples.

    Returns mapping:
      metric -> {point, ci_low, ci_high, n_boot_valid}
    """
    results = list(results)
    n = len(results)
    if n == 0:
        raise ValueError("No results to bootstrap")

    point = dict(metric_fn(results))
    keys = list(point.keys())
    samples: Dict[str, List[float]] = {k: [] for k in keys}
    rng = np.random.default_rng(seed)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot = [results[int(i)] for i in idx]
        vals = metric_fn(boot)
        for k in keys:
            v = float(vals.get(k, float("nan")))
            if not np.isnan(v):
                samples[k].append(v)

    alpha = (1.0 - ci) / 2.0
    lo_q, hi_q = 100.0 * alpha, 100.0 * (1.0 - alpha)
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for k in keys:
        arr = np.asarray(samples[k], dtype=float)
        point_v = float(point[k]) if not np.isnan(point[k]) else None
        if arr.size == 0:
            out[k] = {
                "point": point_v,
                "ci_low": None,
                "ci_high": None,
                "n_boot_valid": 0.0,
            }
            continue
        out[k] = {
            "point": point_v,
            "ci_low": float(np.percentile(arr, lo_q)),
            "ci_high": float(np.percentile(arr, hi_q)),
            "n_boot_valid": float(arr.size),
        }
    return out


def format_with_ci(
    entry: Mapping[str, Optional[float]],
    *,
    as_percent: bool = False,
    digits: int = 4,
) -> str:
    """Format ``value [lo, hi]`` for display / CSV cells."""
    point = entry.get("point")
    lo = entry.get("ci_low")
    hi = entry.get("ci_high")
    if point is None:
        return "N/A"
    if as_percent:
        pct_digits = max(digits - 2, 0)
        point_s = f"{100.0 * point:.{pct_digits}f}%"
        if lo is None or hi is None:
            return point_s
        return (
            f"{point_s} "
            f"[{100.0 * lo:.{pct_digits}f}%, "
            f"{100.0 * hi:.{pct_digits}f}%]"
        )
    point_s = f"{point:.{digits}f}"
    if lo is None or hi is None:
        return point_s
    return f"{point_s} [{lo:.{digits}f}, {hi:.{digits}f}]"


def print_bootstrap_table(
    title: str,
    boot: Mapping[str, Mapping[str, Optional[float]]],
    *,
    percent_keys: Optional[Sequence[str]] = None,
) -> None:
    percent_keys = set(percent_keys or [])
    absolute_keys = {
        "quadratic_weighted_kappa",
        "macro_f1",
        "auc_roc_priority",
        "auc_roc_priority_raw",
        "esi_mae",
        "esi_rmse",
        "cohen_kappa",
        "linear_weighted_kappa",
        "mean_confidence",
    }
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    for key, entry in boot.items():
        as_pct = (
            key in percent_keys
            or key.endswith("_accuracy")
            or key.endswith("_rate")
            or key.endswith("_sensitivity")
            or key.endswith("_specificity")
        )
        if key in absolute_keys or key.startswith("per_esi_accuracy_"):
            # per-esi shown as percent anyway
            if key.startswith("per_esi_accuracy_"):
                as_pct = True
            else:
                as_pct = False
        print(f"  {key:42s}  {format_with_ci(entry, as_percent=as_pct)}")
