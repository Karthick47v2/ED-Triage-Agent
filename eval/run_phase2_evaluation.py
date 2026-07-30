"""Phase 2 evaluation: CRA Phase 2 -> TCA with pre-extracted vitals and physical exam."""
from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from ed_triage.common.schemas import PhysicalExam, VitalSigns
from ed_triage.orchestration.phase2 import run_phase2_pipeline
from eval.scenario_age import merge_intake_and_scenario_age
from eval.schemas import EvaluationResult, EvaluationScenario
from eval.shared.io import load_json_file, save_json_file
from eval.shared.metric_utils import exact_and_within_one

load_dotenv()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("Phase2-Evaluation")
logger.setLevel(logging.INFO)

warnings.filterwarnings(
    "ignore",
    message=r"Pydantic serializer warnings:",
    category=UserWarning,
    module=r"pydantic\.main",
)


def load_vital_signs(json_path: str) -> Dict[int, VitalSigns]:
    return {
        item["scenario_number"]: VitalSigns(**item["vital_signs"])
        for item in load_json_file(json_path)
    }


def load_physical_exams(json_path: str) -> Dict[int, PhysicalExam]:
    return {
        item["scenario_number"]: PhysicalExam(physical_exam=item["physical_exam"])
        for item in load_json_file(json_path)
    }


def load_phase1_results(json_path: str) -> Dict[int, EvaluationResult]:
    data = load_json_file(json_path)
    return {r["scenario_number"]: EvaluationResult(**r) for r in data.get("results", [])}


def load_scenarios(json_path: str) -> Dict[int, EvaluationScenario]:
    return {
        s["scenario_number"]: EvaluationScenario(**s)
        for s in load_json_file(json_path)
    }


def run_single_scenario_phase2(
    phase1_result: EvaluationResult,
    vital_signs: VitalSigns,
    physical_exam: PhysicalExam,
    scenario: Optional[EvaluationScenario] = None,
    verbose: bool = False,
) -> EvaluationResult:
    """Run Phase 2 evaluation for one scenario."""
    scenario_num = phase1_result.scenario_number
    if verbose:
        print(f"\n{'=' * 60}\nScenario {scenario_num} - Phase 2\n{'=' * 60}")

    result = phase1_result.model_copy()
    result.vital_signs = vital_signs
    result.physical_exam = physical_exam

    if not phase1_result.success or phase1_result.intake_data is None:
        result.success = False
        result.error_message = (
            f"Phase 1 failed or missing intake_data: {phase1_result.error_message}"
        )
        return result

    try:
        start = time.time()
        intake = phase1_result.intake_data
        age_years, age_months, age_days = merge_intake_and_scenario_age(
            intake.age_years, intake.age_months, intake.age_days, scenario
        )
        tca_result = run_phase2_pipeline(
            intake_summary=intake,
            vital_signs=vital_signs,
            physical_exam=physical_exam,
            age_years=age_years,
            age_months=age_months,
            age_days=age_days,
        )
        result.phase2_latency_ms = (time.time() - start) * 1000
        result.tca_result = tca_result
        result.phase2_predicted_esi = tca_result.final_esi
        result.phase2_confidence = tca_result.confidence
        result.phase2_rationale = tca_result.rationale
        result.success = True

        if verbose:
            print(f"Ground Truth ESI: {result.ground_truth_esi}")
            print(f"Phase 1 ESI: {result.predicted_esi}")
            print(f"Phase 2 ESI: {tca_result.final_esi} (Confidence: {tca_result.confidence:.2f})")
            print(f"Latency: {result.phase2_latency_ms:.0f}ms")
    except Exception as exc:
        logger.error("Scenario %d Phase 2 failed: %s", scenario_num, exc)
        result.success = False
        result.error_message = str(exc)
    return result


def run_phase2_evaluation(
    phase1_results_path: str,
    vital_signs_path: str,
    physical_exam_path: str,
    output_path: Optional[str] = None,
    verbose: bool = False,
    limit: Optional[int] = None,
    scenarios_path: Optional[str] = None,
) -> List[EvaluationResult]:
    logger.info("Loading Phase 1 results...")
    phase1_results = load_phase1_results(phase1_results_path)
    scenarios_by_num: Dict[int, EvaluationScenario] = (
        load_scenarios(scenarios_path) if scenarios_path else {}
    )
    logger.info("Loading vital signs and physical exams...")
    vital_signs_data = load_vital_signs(vital_signs_path)
    physical_exam_data = load_physical_exams(physical_exam_path)

    scenario_nums = sorted(phase1_results.keys())
    if limit:
        scenario_nums = scenario_nums[:limit]

    logger.info("Running Phase 2 on %d scenarios...", len(scenario_nums))
    results: List[EvaluationResult] = []
    for i, scenario_num in enumerate(scenario_nums, 1):
        logger.info("Processing scenario %d (%d/%d)", scenario_num, i, len(scenario_nums))
        result = run_single_scenario_phase2(
            phase1_result=phase1_results[scenario_num],
            vital_signs=vital_signs_data.get(scenario_num, VitalSigns()),
            physical_exam=physical_exam_data.get(scenario_num, PhysicalExam()),
            scenario=scenarios_by_num.get(scenario_num),
            verbose=verbose,
        )
        results.append(result)

    if output_path:
        save_json_file(output_path, {
            "phase": 2,
            "total_scenarios": len(results),
            "successful": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "results": [r.model_dump() for r in results],
        })
        logger.info("Results saved to %s", output_path)
    return results


def _print_quick_summary(results: List[EvaluationResult]) -> None:
    successful = [
        r for r in results if r.success and r.phase2_predicted_esi is not None
    ]
    if not successful:
        return
    y_true = [r.ground_truth_esi for r in successful]
    y_pred = [r.phase2_predicted_esi for r in successful]
    exact, within = exact_and_within_one(y_true, y_pred)
    print(f"\n{'=' * 60}\nPHASE 2 EVALUATION SUMMARY\n{'=' * 60}")
    print(f"Scenarios: {len(successful)}/{len(results)} successful")
    print(f"ESI Exact Match: {exact:.1%}")
    print(f"ESI Within +/-1: {within:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 Evaluation Runner")
    parser.add_argument("--phase1-results", default="results/phase1_results.json",
                        help="Path to Phase 1 results JSON")
    parser.add_argument("--vital-signs", default="vital_signs_extracted.json",
                        help="Path to vital signs JSON")
    parser.add_argument("--physical-exam", default="physical_exam_extracted.json",
                        help="Path to physical exam JSON")
    parser.add_argument("--output", "-o", default="results/phase2_results.json")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--limit", "-l", type=int, help="Limit number of scenarios")
    parser.add_argument(
        "--scenarios",
        default=None,
        help="Optional ETEK scenarios JSON (structured age fields merge with intake for TCA)",
    )
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results = run_phase2_evaluation(
        phase1_results_path=args.phase1_results,
        vital_signs_path=args.vital_signs,
        physical_exam_path=args.physical_exam,
        output_path=args.output,
        verbose=args.verbose,
        limit=args.limit,
        scenarios_path=args.scenarios,
    )
    _print_quick_summary(results)


if __name__ == "__main__":
    main()
