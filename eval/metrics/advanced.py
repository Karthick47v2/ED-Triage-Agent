"""sklearn-backed metrics and confusion-matrix heatmap (lazy imports)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from eval.metrics.aggregate import ESI_LABELS
from eval.schemas import EvaluationResult


def _priority_pairs(
    results: List[EvaluationResult],
) -> Tuple[List[int], List[int], List[float]]:
    """Phase 1 HIGH/LOW priority pairs and confidence proxies for AUC-ROC."""
    ok = [r for r in results if r.success and r.predicted_priority is not None]
    y_true = [1 if r.ground_truth_priority == "HIGH" else 0 for r in ok]
    y_pred = [1 if r.predicted_priority == "HIGH" else 0 for r in ok]
    y_score = [
        r.confidence
        if r.confidence is not None
        else float(r.predicted_priority == "HIGH")
        for r in ok
    ]
    return y_true, y_pred, y_score


def advanced_block(
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
        if len(y_true) >= 2
        else float("nan")
    )

    if y_true:
        macro = float(
            f1_score(
                y_true, y_pred, labels=ESI_LABELS, average="macro", zero_division=0
            )
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
            if len(set(y_true_bin)) >= 2
            else float("nan")
        )
        block["priority_n"] = len(y_true_bin)
    return block


def save_confusion_heatmap(
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
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=12,
                color=color,
                fontweight="bold",
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\nConfusion matrix heatmap saved -> {output_path}")
