"""Reliability diagram: binned mean confidence vs empirical accuracy (Phase 1 or 2)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from eval.schemas import EvaluationResult
from eval.shared.io import load_evaluation_results


def _pairs_phase1(rows: List[EvaluationResult]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for r in rows:
        if not r.success or r.predicted_esi is None or r.confidence is None:
            continue
        if not (0.0 <= r.confidence <= 1.0):
            continue
        correct = 1.0 if r.predicted_esi == r.ground_truth_esi else 0.0
        out.append((float(r.confidence), correct))
    return out


def _pairs_phase2(rows: List[EvaluationResult]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for r in rows:
        if not r.success or r.phase2_predicted_esi is None or r.phase2_confidence is None:
            continue
        if not (0.0 <= r.phase2_confidence <= 1.0):
            continue
        correct = 1.0 if r.phase2_predicted_esi == r.ground_truth_esi else 0.0
        out.append((float(r.phase2_confidence), correct))
    return out


def _bin_means(
    pairs: List[Tuple[float, float]], n_bins: int
) -> Tuple[List[float], List[float], List[int]]:
    """Return (bin_center, mean_accuracy, count) per bin with at least one sample."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    sums_c = [0.0] * n_bins
    sums_y = [0.0] * n_bins
    counts = [0] * n_bins
    for conf, y in pairs:
        # conf in [0,1]; last bin includes 1.0
        idx = min(int(conf * n_bins), n_bins - 1)
        sums_c[idx] += conf
        sums_y[idx] += y
        counts[idx] += 1
    centers: List[float] = []
    accs: List[float] = []
    ns: List[int] = []
    for i in range(n_bins):
        if counts[i] == 0:
            continue
        centers.append(sums_c[i] / counts[i])
        accs.append(sums_y[i] / counts[i])
        ns.append(counts[i])
    return centers, accs, ns


def plot_reliability(
    pairs: List[Tuple[float, float]],
    out_path: Path,
    title: str,
    n_bins: int = 10,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not pairs:
        raise ValueError("No (confidence, correctness) pairs to plot")

    x_mean, y_acc, _ = _bin_means(pairs, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Perfect calibration")
    ax.plot(x_mean, y_acc, marker="o", label="Empirical accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean confidence (bin)")
    ax.set_ylabel("Fraction correct (exact ESI)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reliability diagram from evaluation results JSON")
    parser.add_argument("--input", required=True, help="Phase 1 or Phase 2 results JSON")
    parser.add_argument(
        "--output",
        required=True,
        help="Output image path (.png) or prefix (we append _reliability.png)",
    )
    parser.add_argument("--phase", type=int, choices=(1, 2), default=1)
    parser.add_argument("--bins", type=int, default=10, help="Number of equal-width confidence bins")
    args = parser.parse_args()

    inp = Path(args.input)
    rows = load_evaluation_results(inp)
    if args.phase == 1:
        pairs = _pairs_phase1(rows)
        title = "Reliability (Phase 1, exact ESI)"
    else:
        pairs = _pairs_phase2(rows)
        title = "Reliability (Phase 2, exact ESI)"

    outp = Path(args.output)
    if outp.suffix.lower() != ".png":
        outp = outp.with_suffix(".png")

    plot_reliability(pairs, outp, title=title, n_bins=args.bins)
    print(f"Wrote {outp}")


if __name__ == "__main__":
    main()
