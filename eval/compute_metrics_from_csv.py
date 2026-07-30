"""Compute full Phase 1 / Phase 2 metrics (incl. TRIAGEAGENT + bootstrap + Wilson CIs).

Usage:
  python -m eval.compute_metrics_from_csv \\
    --phase1 "results - orig/metrics_phase1_minimal.csv" \\
    --phase2 "results - orig/metrics_phase2_minimal.csv" \\
    --bootstrap 1000 \\
    --json-out "results - orig/all_metrics_from_csv.json" \\
    --summary-csv "results - orig/all_metrics_summary.csv" \\
    --bootstrap-csv "results - orig/all_metrics_bootstrap.csv" \\
    --wilson-csv "results - orig/all_metrics_wilson.csv"
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics.bootstrap import (
    bootstrap_metrics,
    format_with_ci,
    print_bootstrap_table,
    scalar_metrics_phase1,
    scalar_metrics_phase2,
)
from eval.metrics.report import print_full_report, report_phase1, report_phase2
from eval.metrics.wilson import (
    format_wilson,
    wilson_metrics_phase1,
    wilson_metrics_phase2,
)
from eval.schemas import EvaluationResult
from eval.shared.triage import derive_priority


Dataset = str  # "combined" | "practice" | "competency" | custom label

PERCENT_KEYS = {
    "esi_exact_accuracy",
    "esi_within_one_accuracy",
    "priority_accuracy",
    "high_priority_sensitivity",
    "high_priority_specificity",
    "high_acuity_sensitivity",
    "high_acuity_specificity",
    "undertriage_rate",
    "overtriage_rate",
    "ta_total_discordance_rate",
    "ta_undertriage_rate",
    "ta_overtriage_rate",
    "ta_significant_undertriage_rate",
    "ta_significant_overtriage_rate",
}


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _opt_int(value: Any) -> Optional[int]:
    if value in (None, "", "None"):
        return None
    return int(float(value))


def _opt_float(value: Any) -> Optional[float]:
    if value in (None, "", "None"):
        return None
    return float(value)


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _filter_dataset(rows: List[Dict[str, str]], dataset: str) -> List[Dict[str, str]]:
    """Filter rows for a named split.

    Built-ins:
      - combined: all rows
      - practice / competency: filter by substring in ``source_file``
    Any other label (e.g. ``cm216``): all rows (single-dataset mode).
    """
    if dataset in ("combined",):
        return rows
    if dataset in ("practice", "competency"):
        return [row for row in rows if dataset in row.get("source_file", "")]
    # Custom single-dataset label: use every row.
    return rows


def _dataset_columns(datasets: Sequence[str]) -> List[str]:
    return list(datasets)


def phase1_rows_to_results(rows: List[Dict[str, str]]) -> List[EvaluationResult]:
    results: List[EvaluationResult] = []
    for row in rows:
        gt_esi = _opt_int(row["ground_truth_esi"])
        if gt_esi is None:
            raise ValueError(f"Missing ground_truth_esi in row: {row}")
        gt_priority = row.get("ground_truth_priority") or derive_priority(gt_esi)
        results.append(
            EvaluationResult(
                scenario_number=_opt_int(row.get("scenario_number")) or 0,
                ground_truth_esi=gt_esi,
                ground_truth_priority=gt_priority,  # type: ignore[arg-type]
                predicted_esi=_opt_int(row.get("predicted_esi")),
                predicted_priority=row.get("predicted_priority") or None,  # type: ignore[arg-type]
                confidence=_opt_float(row.get("confidence")),
                phase1_latency_ms=_opt_float(row.get("phase1_latency_ms")) or 0.0,
                success=_truthy(row.get("success", True)),
            )
        )
    return results


def phase2_rows_to_results(rows: List[Dict[str, str]]) -> List[EvaluationResult]:
    results: List[EvaluationResult] = []
    for row in rows:
        gt_esi = _opt_int(row["ground_truth_esi"])
        if gt_esi is None:
            raise ValueError(f"Missing ground_truth_esi in row: {row}")
        results.append(
            EvaluationResult(
                scenario_number=_opt_int(row.get("scenario_number")) or 0,
                ground_truth_esi=gt_esi,
                ground_truth_priority=derive_priority(gt_esi),  # type: ignore[arg-type]
                phase2_predicted_esi=_opt_int(row.get("phase2_predicted_esi")),
                phase2_confidence=_opt_float(row.get("phase2_confidence")),
                phase2_latency_ms=_opt_float(row.get("phase2_latency_ms")) or 0.0,
                predicted_esi=_opt_int(row.get("predicted_esi")),
                predicted_priority=row.get("predicted_priority") or None,  # type: ignore[arg-type]
                confidence=_opt_float(row.get("confidence")),
                phase1_latency_ms=_opt_float(row.get("phase1_latency_ms")) or 0.0,
                success=_truthy(row.get("success", True)),
            )
        )
    return results


def _flatten_report(report: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    adv = report.get("advanced") or {}
    ta = report.get("triageagent") or {}
    per_esi = report.get("per_esi_accuracy") or {}
    per_f1 = adv.get("per_class_f1") or {}

    row: Dict[str, Any] = {
        "dataset": dataset,
        "phase": report["phase"],
        "total_scenarios": report["total_scenarios"],
        "successful_runs": report["successful_runs"],
        "failed_runs": report["failed_runs"],
        "esi_exact_accuracy": report["esi_exact_accuracy"],
        "esi_within_one_accuracy": report["esi_within_one_accuracy"],
        "esi_mae": report.get("esi_mae"),
        "esi_rmse": report.get("esi_rmse"),
        "cohen_kappa": report.get("cohen_kappa"),
        "linear_weighted_kappa": report.get("linear_weighted_kappa"),
        "quadratic_weighted_kappa": report.get("quadratic_weighted_kappa"),
        "quadratic_weighted_kappa_sklearn": adv.get("quadratic_weighted_kappa_sklearn"),
        "macro_f1": adv.get("macro_f1"),
        "undertriage_rate": report["undertriage_rate"],
        "overtriage_rate": report["overtriage_rate"],
        "latency_mean_ms": report["latency_mean_ms"],
        "latency_median_ms": report["latency_median_ms"],
        "latency_p95_ms": report["latency_p95_ms"],
        "ta_n": ta.get("n"),
        "ta_total_discordance_rate": ta.get("total_discordance_rate"),
        "ta_undertriage_rate": ta.get("undertriage_rate"),
        "ta_overtriage_rate": ta.get("overtriage_rate"),
        "ta_significant_undertriage_rate": ta.get("significant_undertriage_rate"),
        "ta_significant_overtriage_rate": ta.get("significant_overtriage_rate"),
    }

    if report["phase"] == 1:
        row.update(
            {
                "priority_accuracy": report.get("priority_accuracy"),
                "high_priority_sensitivity": report.get("high_priority_sensitivity"),
                "high_priority_specificity": report.get("high_priority_specificity"),
                "auc_roc_priority": adv.get("auc_roc_priority"),
                "mean_confidence": None,
            }
        )
    else:
        row.update(
            {
                "priority_accuracy": None,
                "high_priority_sensitivity": report.get("high_acuity_sensitivity"),
                "high_priority_specificity": report.get("high_acuity_specificity"),
                "auc_roc_priority": None,
                "mean_confidence": report.get("mean_confidence"),
            }
        )

    for esi in range(1, 6):
        row[f"per_esi_accuracy_esi_{esi}"] = per_esi.get(esi, per_esi.get(str(esi)))
        row[f"per_class_f1_esi_{esi}"] = per_f1.get(f"ESI_{esi}")
    return row


PHASE1_PAPER_METRICS = [
    ("esi_exact_accuracy", True),
    ("esi_within_one_accuracy", True),
    ("quadratic_weighted_kappa", False),
    ("macro_f1", False),
    ("priority_accuracy", True),
    ("high_priority_sensitivity", True),
    ("high_priority_specificity", True),
    ("auc_roc_priority", False),
    ("ta_total_discordance_rate", True),
    ("ta_undertriage_rate", True),
    ("ta_significant_undertriage_rate", True),
    ("ta_overtriage_rate", True),
    ("ta_significant_overtriage_rate", True),
]
PHASE2_PAPER_METRICS = [
    ("esi_exact_accuracy", True),
    ("esi_within_one_accuracy", True),
    ("quadratic_weighted_kappa", False),
    ("macro_f1", False),
    ("high_acuity_sensitivity", True),
    ("high_acuity_specificity", True),
    ("ta_total_discordance_rate", True),
    ("ta_undertriage_rate", True),
    ("ta_significant_undertriage_rate", True),
    ("ta_overtriage_rate", True),
    ("ta_significant_overtriage_rate", True),
]
WILSON_PAPER_METRICS = {
    1: [
        "esi_exact_accuracy",
        "esi_within_one_accuracy",
        "priority_accuracy",
        "high_priority_sensitivity",
        "high_priority_specificity",
        "ta_total_discordance_rate",
        "ta_undertriage_rate",
        "ta_significant_undertriage_rate",
        "ta_overtriage_rate",
        "ta_significant_overtriage_rate",
    ],
    2: [
        "esi_exact_accuracy",
        "esi_within_one_accuracy",
        "high_acuity_sensitivity",
        "high_acuity_specificity",
        "ta_total_discordance_rate",
        "ta_undertriage_rate",
        "ta_significant_undertriage_rate",
        "ta_overtriage_rate",
        "ta_significant_overtriage_rate",
    ],
}


def _bootstrap_rows(
    dataset: str,
    phase: int,
    boot: Dict[str, Dict[str, Optional[float]]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metric, entry in boot.items():
        as_pct = metric in PERCENT_KEYS
        rows.append(
            {
                "dataset": dataset,
                "phase": phase,
                "metric": metric,
                "method": "bootstrap",
                "point": entry.get("point"),
                "ci_low": entry.get("ci_low"),
                "ci_high": entry.get("ci_high"),
                "n": entry.get("n_boot_valid"),
                "display": format_with_ci(entry, as_percent=as_pct),
            }
        )
    return rows


def _wilson_rows(
    dataset: str,
    phase: int,
    wilson: Dict[str, Dict[str, Optional[float]]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metric, entry in wilson.items():
        rows.append(
            {
                "dataset": dataset,
                "phase": phase,
                "metric": metric,
                "method": "wilson",
                "point": entry.get("point"),
                "ci_low": entry.get("ci_low"),
                "ci_high": entry.get("ci_high"),
                "n": entry.get("n"),
                "k": entry.get("k"),
                "display": format_wilson(entry, as_percent=True),
            }
        )
    return rows


def _paper_style_wide(
    all_boot: Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]],
    datasets: Sequence[str],
) -> List[Dict[str, str]]:
    """One row per metric, columns = datasets."""
    out: List[Dict[str, str]] = []
    for phase, order in ((1, PHASE1_PAPER_METRICS), (2, PHASE2_PAPER_METRICS)):
        for metric, as_pct in order:
            row = {"phase": str(phase), "metric": metric, "method": "bootstrap"}
            for dataset in datasets:
                entry = all_boot[dataset][f"phase{phase}"][metric]
                row[dataset] = format_with_ci(entry, as_percent=as_pct)
            out.append(row)
    return out


def _wilson_paper_style_wide(
    all_wilson: Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]],
    datasets: Sequence[str],
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for phase, metrics in WILSON_PAPER_METRICS.items():
        for metric in metrics:
            row = {"phase": str(phase), "metric": metric, "method": "wilson"}
            for dataset in datasets:
                entry = all_wilson[dataset][f"phase{phase}"][metric]
                row[dataset] = format_wilson(entry, as_percent=True)
            out.append(row)
    return out


def _print_wilson_table(
    title: str,
    wilson: Dict[str, Dict[str, Optional[float]]],
) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    for metric, entry in wilson.items():
        print(f"  {metric:42s}  {format_wilson(entry, as_percent=True)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute all eval metrics from Phase 1 / Phase 2 minimal CSVs, "
            "with optional bootstrap CIs."
        )
    )
    parser.add_argument(
        "--phase1",
        type=Path,
        default=Path("results - orig/metrics_phase1_minimal.csv"),
        help="Phase 1 minimal metrics CSV",
    )
    parser.add_argument(
        "--phase2",
        type=Path,
        default=Path("results - orig/metrics_phase2_minimal.csv"),
        help="Phase 2 minimal metrics CSV",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("results - orig/all_metrics_from_csv.json"),
        help="Write full nested metrics JSON here",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("results - orig/all_metrics_summary.csv"),
        help="Write flat point-estimate summary CSV here",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resamples (0 to disable)",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="RNG seed for bootstrap",
    )
    parser.add_argument(
        "--bootstrap-csv",
        type=Path,
        default=Path("results - orig/all_metrics_bootstrap.csv"),
        help="Long-form bootstrap results CSV",
    )
    parser.add_argument(
        "--bootstrap-table-csv",
        type=Path,
        default=Path("results - orig/all_metrics_bootstrap_table.csv"),
        help="Wide paper-style bootstrap table CSV",
    )
    parser.add_argument(
        "--wilson-csv",
        type=Path,
        default=Path("results - orig/all_metrics_wilson.csv"),
        help="Long-form Wilson score CI results CSV",
    )
    parser.add_argument(
        "--wilson-table-csv",
        type=Path,
        default=Path("results - orig/all_metrics_wilson_table.csv"),
        help="Wide paper-style Wilson CI table CSV",
    )
    parser.add_argument(
        "--no-wilson",
        action="store_true",
        help="Skip Wilson score intervals",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["combined", "practice", "competency"],
        help=(
            "Dataset labels to compute. Built-ins: combined / practice / competency. "
            "For a single expanded CM CSV set, use e.g. --datasets cm216"
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only write files; skip printing reports",
    )
    args = parser.parse_args()

    if not args.phase1.exists():
        sys.exit(f"error: Phase 1 CSV not found: {args.phase1}")
    if not args.phase2.exists():
        sys.exit(f"error: Phase 2 CSV not found: {args.phase2}")

    phase1_rows = _load_csv(args.phase1)
    phase2_rows = _load_csv(args.phase2)
    datasets: List[str] = list(args.datasets)

    all_reports: Dict[str, Dict[str, Any]] = {}
    summary_rows: List[Dict[str, Any]] = []
    bootstrap_long: List[Dict[str, Any]] = []
    wilson_long: List[Dict[str, Any]] = []
    all_boot: Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]] = {}
    all_wilson: Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]] = {}

    for dataset in datasets:
        p1_rows = _filter_dataset(phase1_rows, dataset)
        p2_rows = _filter_dataset(phase2_rows, dataset)
        if not p1_rows or not p2_rows:
            print(f"warning: skipping empty dataset '{dataset}'")
            continue
        p1 = phase1_rows_to_results(p1_rows)
        p2 = phase2_rows_to_results(p2_rows)
        r1 = report_phase1(p1)
        r2 = report_phase2(p2)
        all_reports[dataset] = {"phase1": r1, "phase2": r2}
        summary_rows.append(_flatten_report(r1, dataset))
        summary_rows.append(_flatten_report(r2, dataset))

        if not args.quiet:
            print(f"\n########## DATASET: {dataset.upper()} ##########")
            print_full_report(r1)
            print_full_report(r2)

        if not args.no_wilson:
            w1 = wilson_metrics_phase1(p1)
            w2 = wilson_metrics_phase2(p2)
            all_wilson[dataset] = {"phase1": w1, "phase2": w2}
            all_reports[dataset]["phase1_wilson"] = w1
            all_reports[dataset]["phase2_wilson"] = w2
            wilson_long.extend(_wilson_rows(dataset, 1, w1))
            wilson_long.extend(_wilson_rows(dataset, 2, w2))
            if not args.quiet:
                _print_wilson_table(
                    f"{dataset.upper()} PHASE 1 — Wilson 95% CI", w1
                )
                _print_wilson_table(
                    f"{dataset.upper()} PHASE 2 — Wilson 95% CI", w2
                )

        if args.bootstrap > 0:
            if not args.quiet:
                print(
                    f"\nBootstrapping {dataset} (B={args.bootstrap}, "
                    f"seed={args.bootstrap_seed})..."
                )
            b1 = bootstrap_metrics(
                p1,
                scalar_metrics_phase1,
                n_boot=args.bootstrap,
                seed=args.bootstrap_seed,
            )
            b2 = bootstrap_metrics(
                p2,
                scalar_metrics_phase2,
                n_boot=args.bootstrap,
                seed=args.bootstrap_seed,
            )
            all_boot[dataset] = {"phase1": b1, "phase2": b2}
            all_reports[dataset]["phase1_bootstrap"] = b1
            all_reports[dataset]["phase2_bootstrap"] = b2
            bootstrap_long.extend(_bootstrap_rows(dataset, 1, b1))
            bootstrap_long.extend(_bootstrap_rows(dataset, 2, b2))

            if not args.quiet:
                print_bootstrap_table(
                    f"{dataset.upper()} PHASE 1 — bootstrap point [95% CI]",
                    b1,
                    percent_keys=PERCENT_KEYS,
                )
                print_bootstrap_table(
                    f"{dataset.upper()} PHASE 2 — bootstrap point [95% CI]",
                    b2,
                    percent_keys=PERCENT_KEYS,
                )

    if not summary_rows:
        sys.exit("error: no datasets produced results")

    # Only keep datasets that actually ran
    datasets = [d for d in datasets if d in all_reports]

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(all_reports, indent=2, default=str), encoding="utf-8")

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nWrote {args.json_out}")
    print(f"Wrote {args.summary_csv}")

    if wilson_long:
        args.wilson_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "dataset",
            "phase",
            "metric",
            "method",
            "point",
            "ci_low",
            "ci_high",
            "n",
            "k",
            "display",
        ]
        with args.wilson_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(wilson_long)
        print(f"Wrote {args.wilson_csv}")

        wide_w = _wilson_paper_style_wide(all_wilson, datasets)
        args.wilson_table_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.wilson_table_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["phase", "metric", "method", *datasets],
            )
            writer.writeheader()
            writer.writerows(wide_w)
        print(f"Wrote {args.wilson_table_csv}")

    if args.bootstrap > 0 and bootstrap_long:
        args.bootstrap_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.bootstrap_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(bootstrap_long[0].keys()))
            writer.writeheader()
            writer.writerows(bootstrap_long)
        print(f"Wrote {args.bootstrap_csv}")

        wide = _paper_style_wide(all_boot, datasets)
        args.bootstrap_table_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.bootstrap_table_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["phase", "metric", "method", *datasets],
            )
            writer.writeheader()
            writer.writerows(wide)
        print(f"Wrote {args.bootstrap_table_csv}")


if __name__ == "__main__":
    main()
