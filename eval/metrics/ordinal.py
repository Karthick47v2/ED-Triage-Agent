"""Paper-aligned ordinal metrics (MAE / RMSE / Cohen's kappa / weighted kappa)."""
from __future__ import annotations

import math
from typing import Any, Dict, Sequence, Tuple


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


def ordinal_report(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> Dict[str, Any]:
    mae, rmse = _mae_rmse(y_true, y_pred)
    return {
        "esi_mae": mae,
        "esi_rmse": rmse,
        "cohen_kappa": cohen_kappa_multiclass(y_true, y_pred),
        "linear_weighted_kappa": weighted_kappa(y_true, y_pred, kind="linear"),
        "quadratic_weighted_kappa": weighted_kappa(
            y_true, y_pred, kind="quadratic"
        ),
        "confusion_matrix_esi": _confusion_matrix(y_true, y_pred),
    }
