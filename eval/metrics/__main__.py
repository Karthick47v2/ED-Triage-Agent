"""CLI: python -m eval.metrics RESULTS.json --phase {1,2}"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.metrics.aggregate import ESI_LABELS
from eval.metrics.advanced import save_confusion_heatmap
from eval.metrics.report import print_full_report, report_phase1, report_phase2
from eval.shared.io import load_evaluation_results
from eval.shared.triage import phase1_esi_pairs, phase2_esi_pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute every evaluation metric for a results file (Phase 1 or Phase 2).\n\n"
            "Examples:\n"
            "  python -m eval.metrics results/phase1_results.json --phase 1\n"
            "  python -m eval.metrics results/phase2_results.json --phase 2 \\\n"
            "      --heatmap figures/cm_phase2.png --json-out results/phase2_metrics.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "results_file",
        type=Path,
        help="Path to phase1 / phase2 results JSON",
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        required=True,
        help="Which phase the results file corresponds to",
    )
    parser.add_argument(
        "--heatmap",
        type=Path,
        default=None,
        metavar="FILE.png",
        help="Also save a confusion-matrix heatmap to this path",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Also write the full report JSON to this path",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress human-readable output (useful with --json-out)",
    )
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
            save_confusion_heatmap(
                y_true,
                y_pred,
                ESI_LABELS,
                title=f"ESI Confusion Matrix - {label}  (n={len(y_true)})",
                output_path=args.heatmap,
            )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2))
        if not args.quiet:
            print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
