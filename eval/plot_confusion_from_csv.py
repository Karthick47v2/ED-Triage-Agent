"""Generate ESI confusion-matrix heatmaps (Phase 1 + Phase 2) from minimal CSVs.

Matches the side-by-side style used in evaluation figures.

Examples:
  # Practice/competency remapped CSVs (n=60)
  python -m eval.plot_confusion_from_csv \\
    --phase1 "results - orig/metrics_phase1_minimal.csv" \\
    --phase2 "results - orig/metrics_phase2_minimal.csv" \\
    --out "results - orig/esi_confusion_matrices.png"

  # CM216 expanded CSVs (n=216)
  python -m eval.plot_confusion_from_csv \\
    --phase1 "results - orig/cm216/metrics_phase1_minimal.csv" \\
    --phase2 "results - orig/cm216/metrics_phase2_minimal.csv" \\
    --out "results - orig/cm216/esi_confusion_matrices.png"
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics.aggregate import ESI_LABELS


def _load_csv(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _pairs_from_csv(
    rows: Sequence[dict],
    pred_col: str,
) -> Tuple[List[int], List[int]]:
    y_true: List[int] = []
    y_pred: List[int] = []
    for row in rows:
        if not _truthy(row.get("success", True)):
            continue
        gt = row.get("ground_truth_esi")
        pred = row.get(pred_col)
        if gt in (None, "", "None") or pred in (None, "", "None"):
            continue
        y_true.append(int(float(gt)))
        y_pred.append(int(float(pred)))
    if not y_true:
        raise ValueError(f"No usable rows for predicted column '{pred_col}'")
    return y_true, y_pred


def _confusion_counts(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int] = ESI_LABELS,
) -> List[List[int]]:
    index = {lbl: i for i, lbl in enumerate(labels)}
    cm = [[0 for _ in labels] for _ in labels]
    for t, p in zip(y_true, y_pred):
        if t in index and p in index:
            cm[index[t]][index[p]] += 1
    return cm


def save_side_by_side_heatmaps(
    cm_phase1: Sequence[Sequence[int]],
    cm_phase2: Sequence[Sequence[int]],
    *,
    output_path: Path,
    labels: Sequence[int] = ESI_LABELS,
    title1: str = "ESI Confusion Matrix - Phase 1",
    title2: str = "ESI Confusion Matrix - Phase 2",
    dpi: int = 150,
) -> None:
    """Save a two-panel figure colored by row-normalized recall, annotated with counts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    def _draw(ax, cm, title: str) -> None:
        arr = np.asarray(cm, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            row_sums = arr.sum(axis=1, keepdims=True)
            norm = np.divide(arr, row_sums, out=np.zeros_like(arr), where=row_sums != 0)

        ax.imshow(norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
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
                color = "white" if norm[i, j] > 0.55 else "black"
                ax.text(
                    j,
                    i,
                    str(int(arr[i, j])),
                    ha="center",
                    va="center",
                    fontsize=12,
                    color=color,
                    fontweight="bold",
                )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    _draw(axes[0], cm_phase1, title1)
    _draw(axes[1], cm_phase2, title2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved side-by-side confusion matrices -> {output_path}")


def save_single_heatmap(
    cm: Sequence[Sequence[int]],
    *,
    output_path: Path,
    title: str,
    labels: Sequence[int] = ESI_LABELS,
    dpi: int = 150,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    arr = np.asarray(cm, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = arr.sum(axis=1, keepdims=True)
        norm = np.divide(arr, row_sums, out=np.zeros_like(arr), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(6, 5.2))
    ax.imshow(norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
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
            color = "white" if norm[i, j] > 0.55 else "black"
            ax.text(
                j,
                i,
                str(int(arr[i, j])),
                ha="center",
                va="center",
                fontsize=12,
                color=color,
                fontweight="bold",
            )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Phase 1 / Phase 2 ESI confusion matrices from minimal CSVs."
    )
    parser.add_argument(
        "--phase1",
        type=Path,
        required=True,
        help="Phase 1 minimal CSV (needs ground_truth_esi, predicted_esi)",
    )
    parser.add_argument(
        "--phase2",
        type=Path,
        required=True,
        help="Phase 2 minimal CSV (needs ground_truth_esi, phase2_predicted_esi)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results - orig/esi_confusion_matrices.png"),
        help="Output path for the side-by-side PNG",
    )
    parser.add_argument(
        "--also-individual",
        action="store_true",
        help="Also write phase1_cm.png and phase2_cm.png next to --out",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    if not args.phase1.exists():
        sys.exit(f"error: Phase 1 CSV not found: {args.phase1}")
    if not args.phase2.exists():
        sys.exit(f"error: Phase 2 CSV not found: {args.phase2}")

    y1_true, y1_pred = _pairs_from_csv(_load_csv(args.phase1), "predicted_esi")
    y2_true, y2_pred = _pairs_from_csv(_load_csv(args.phase2), "phase2_predicted_esi")
    cm1 = _confusion_counts(y1_true, y1_pred)
    cm2 = _confusion_counts(y2_true, y2_pred)

    n1 = sum(sum(r) for r in cm1)
    n2 = sum(sum(r) for r in cm2)
    print(f"Phase 1 n={n1}, Phase 2 n={n2}")

    save_side_by_side_heatmaps(
        cm1,
        cm2,
        output_path=args.out,
        title1="ESI Confusion Matrix - Phase 1",
        title2="ESI Confusion Matrix - Phase 2",
        dpi=args.dpi,
    )

    if args.also_individual:
        p1_out = args.out.with_name("phase1_cm.png")
        p2_out = args.out.with_name("phase2_cm.png")
        save_single_heatmap(
            cm1,
            output_path=p1_out,
            title="ESI Confusion Matrix - Phase 1",
            dpi=args.dpi,
        )
        save_single_heatmap(
            cm2,
            output_path=p2_out,
            title="ESI Confusion Matrix - Phase 2",
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
