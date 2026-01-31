"""Phase 2 evaluation: CRA Phase 2 → TCA with pre-extracted vitals and physical exam."""
import os
import sys
import json
import time
import logging
import argparse
from typing import List, Optional, Dict, Any
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))
from eval.schemas import EvaluationScenario, EvaluationResult
from ed_triage.iia.schema import IntakeSummary
from ed_triage.common.schemas import VitalSigns, PhysicalExam
from ed_triage.graph_phase2 import run_phase2

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("Phase2-Evaluation")
logger.setLevel(logging.INFO)


def derive_priority(esi_level: int) -> str:
    """Convert ESI level to binary priority."""
    return "HIGH" if esi_level <= 3 else "LOW"


def load_vital_signs(json_path: str) -> Dict[int, VitalSigns]:
    """Load vital signs from extracted JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return {
        item["scenario_number"]: VitalSigns(**item["vital_signs"])
        for item in data
    }


def load_physical_exams(json_path: str) -> Dict[int, PhysicalExam]:
    """Load physical exams from extracted JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return {
        item["scenario_number"]: PhysicalExam(physical_exam=item["physical_exam"])
        for item in data
    }


def load_phase1_results(json_path: str) -> Dict[int, EvaluationResult]:
    """Load Phase 1 results to get intake_data."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return {
        r["scenario_number"]: EvaluationResult(**r)
        for r in data.get("results", [])
    }


def run_single_scenario_phase2(
    phase1_result: EvaluationResult,
    vital_signs: VitalSigns,
    physical_exam: PhysicalExam,
    verbose: bool = False
) -> EvaluationResult:
    """Run Phase 2 evaluation for one scenario."""
    scenario_num = phase1_result.scenario_number
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Scenario {scenario_num} - Phase 2")
        print(f"{'='*60}")
    result = phase1_result.model_copy()
    result.vital_signs = vital_signs
    result.physical_exam = physical_exam
    if not phase1_result.success or phase1_result.intake_data is None:
        result.error_message = f"Phase 1 failed or missing intake_data: {phase1_result.error_message}"
        result.success = False
        return result
    
    try:
        start_time = time.time()
        tca_result = run_phase2(
            intake_data=phase1_result.intake_data,
            vital_signs=vital_signs,
            physical_exam=physical_exam
        )
        
        end_time = time.time()
        result.phase2_latency_ms = (end_time - start_time) * 1000
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
            
    except Exception as e:
        logger.error(f"Scenario {scenario_num} Phase 2 failed: {e}")
        result.success = False
        result.error_message = str(e)
        
    return result


def run_phase2_evaluation(
    phase1_results_path: str,
    vital_signs_path: str,
    physical_exam_path: str,
    output_path: Optional[str] = None,
    verbose: bool = False,
    limit: Optional[int] = None
) -> List[EvaluationResult]:
    """Run Phase 2 evaluation on all scenarios."""
    logger.info("Loading Phase 1 results...")
    phase1_results = load_phase1_results(phase1_results_path)
    
    logger.info("Loading vital signs...")
    vital_signs_data = load_vital_signs(vital_signs_path)
    
    logger.info("Loading physical exams...")
    physical_exam_data = load_physical_exams(physical_exam_path)
    scenario_nums = sorted(phase1_results.keys())
    if limit:
        scenario_nums = scenario_nums[:limit]
    
    logger.info(f"Running Phase 2 on {len(scenario_nums)} scenarios...")
    
    results = []
    for i, scenario_num in enumerate(scenario_nums, 1):
        logger.info(f"Processing scenario {scenario_num} ({i}/{len(scenario_nums)})")
        
        phase1_result = phase1_results[scenario_num]
        vital_signs = vital_signs_data.get(scenario_num, VitalSigns())
        physical_exam = physical_exam_data.get(scenario_num, PhysicalExam())
        
        result = run_single_scenario_phase2(
            phase1_result=phase1_result,
            vital_signs=vital_signs,
            physical_exam=physical_exam,
            verbose=verbose
        )
        results.append(result)
    if output_path:
        output_data = {
            "phase": 2,
            "total_scenarios": len(results),
            "successful": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "results": [r.model_dump() for r in results]
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Evaluation Runner")
    parser.add_argument(
        "--phase1-results",
        default="results/phase1_results.json",
        help="Path to Phase 1 results JSON"
    )
    parser.add_argument(
        "--vital-signs",
        default="vital_signs_extracted.json",
        help="Path to vital signs JSON"
    )
    parser.add_argument(
        "--physical-exam",
        default="physical_exam_extracted.json",
        help="Path to physical exam JSON"
    )
    parser.add_argument(
        "--output", "-o",
        default="results/phase2_results.json",
        help="Output path for results"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--limit", "-l", type=int, help="Limit number of scenarios")
    
    args = parser.parse_args()
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = run_phase2_evaluation(
        phase1_results_path=args.phase1_results,
        vital_signs_path=args.vital_signs,
        physical_exam_path=args.physical_exam,
        output_path=args.output,
        verbose=args.verbose,
        limit=args.limit
    )
    successful = [r for r in results if r.success and r.phase2_predicted_esi is not None]
    if successful:
        exact_matches = sum(1 for r in successful if r.phase2_predicted_esi == r.ground_truth_esi)
        within_one = sum(1 for r in successful if abs(r.phase2_predicted_esi - r.ground_truth_esi) <= 1)
        
        print(f"\n{'='*60}")
        print("PHASE 2 EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Scenarios: {len(successful)}/{len(results)} successful")
        print(f"ESI Exact Match: {exact_matches}/{len(successful)} ({100*exact_matches/len(successful):.1f}%)")
        print(f"ESI Within ±1: {within_one}/{len(successful)} ({100*within_one/len(successful):.1f}%)")


if __name__ == "__main__":
    main()
