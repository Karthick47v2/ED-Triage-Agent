"""Phase 1 evaluation: run ETEK scenarios through IIA → CRA → PAA with simulated patient agents."""
import os
import sys
import json
import time
import logging
import argparse
from typing import List, Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from eval.schemas import EvaluationScenario, EvaluationResult
from eval.patient_agent import PatientAgent
from eval.metrics import calculate_metrics, print_summary

from ed_triage.graph import build_graph

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("Evaluation")
logger.setLevel(logging.INFO)
MAX_TURNS = 15


def derive_priority(esi_level: int) -> str:
    """Convert ESI level to binary priority."""
    return "HIGH" if esi_level <= 3 else "LOW"


def load_scenarios(json_path: str) -> List[EvaluationScenario]:
    """Load ETEK scenarios from JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return [EvaluationScenario(**s) for s in data]


def run_single_scenario(
    scenario: EvaluationScenario,
    graph,
    verbose: bool = False
) -> EvaluationResult:
    """Run one ETEK scenario through Phase 1 pipeline."""
    logger.info(f"Running scenario {scenario.scenario_number}...")
    result = EvaluationResult(
        scenario_number=scenario.scenario_number,
        ground_truth_esi=scenario.category,
        ground_truth_priority=derive_priority(scenario.category)
    )
    
    start_time = time.time()
    
    try:
        patient = PatientAgent(scenario.description)
        config = {"configurable": {"thread_id": f"eval_{scenario.scenario_number}"}}
        turns = 0
        conversation_complete = False
        transcript = []
        initial_patient_msg = patient.respond("What brings you to the emergency department today?")
        transcript.append({"role": "patient", "content": initial_patient_msg})
        
        state_update = {"messages": [HumanMessage(content=initial_patient_msg)]}
        
        while not conversation_complete and turns < MAX_TURNS:
            turns += 1
            
            if verbose:
                print(f"  --- Invoking Nurse (IIA) ---")
            nurse_response = None
            final_state = None
            for event in graph.stream(state_update, config):
                for key, value in event.items():
                    if key == "interviewer":
                        if value.get("messages"):
                            nurse_response = value["messages"][-1].content
                    elif key == "paa":
                        final_state = value
            
            if nurse_response:
                display_response = nurse_response.replace("[CONVERSATION_END]", "").strip()
                transcript.append({"role": "nurse", "content": display_response})
                if verbose:
                    print(f"  Nurse: {display_response}")
                if "[CONVERSATION_END]" in nurse_response:
                    conversation_complete = True
                else:
                    if verbose:
                        print(f"  --- Calling Patient Agent ---")
                    
                    # Get patient response
                    patient_reply = patient.respond(display_response)
                    transcript.append({"role": "patient", "content": patient_reply})
                    
                    if verbose:
                        print(f"  Patient: {patient_reply}")
                    
                    state_update = {"messages": [HumanMessage(content=patient_reply)]}
            snapshot = graph.get_state(config)
            if snapshot.values.get("paa_result"):
                conversation_complete = True
                final_state = snapshot.values
        end_time = time.time()
        result.phase1_latency_ms = (end_time - start_time) * 1000
        result.interview_turns = turns
        result.transcript = transcript
        
        if final_state and final_state.get("paa_result"):
            paa = final_state["paa_result"]
            result.predicted_esi = paa.tentative_esi
            result.predicted_priority = paa.priority_score
            result.confidence = paa.confidence
            
        if final_state and final_state.get("intake_data"):
            intake = final_state["intake_data"]
            result.emergency_detected = intake.emergency_detected
            result.chief_complaint = intake.chief_complaint
            result.intake_data = intake
            
        if final_state and final_state.get("cra_result"):
            result.cra_result = final_state["cra_result"]
            
        result.success = True
        logger.info(f"  ✓ Scenario {scenario.scenario_number}: Predicted ESI={result.predicted_esi}, "
                   f"Priority={result.predicted_priority} (GT: {result.ground_truth_esi})")
        
    except Exception as e:
        result.success = False
        result.error_message = str(e)
        logger.error(f"  ✗ Scenario {scenario.scenario_number} failed: {e}")
    
    return result


def run_evaluation(
    scenarios: List[EvaluationScenario],
    output_path: Optional[str] = None,
    verbose: bool = False,
    limit: Optional[int] = None
) -> List[EvaluationResult]:
    """Run Phase 1 evaluation on all scenarios."""
    logger.info("Building evaluation graph...")
    graph = build_graph()
    
    if limit:
        scenarios = scenarios[:limit]
    
    logger.info(f"Running {len(scenarios)} scenarios...")
    results = []
    
    for i, scenario in enumerate(scenarios):
        print(f"[{i+1}/{len(scenarios)}] Scenario {scenario.scenario_number}", end="")
        result = run_single_scenario(scenario, graph, verbose=verbose)
        results.append(result)
        
        status = "✓" if result.success else "✗"
        match = "=" if result.predicted_esi == result.ground_truth_esi else "≠"
        print(f" {status} ESI: {result.predicted_esi} {match} {result.ground_truth_esi}")
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_data = {
            "results": [r.model_dump() for r in results],
            "metadata": {
                "total_scenarios": len(scenarios),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Evaluation Runner")
    parser.add_argument("--scenario", type=int, help="Run single scenario by number")
    parser.add_argument("--all", action="store_true", help="Run all scenarios")
    parser.add_argument("--limit", type=int, help="Limit number of scenarios")
    parser.add_argument("--output", type=str, default="results/phase1_results.json",
                       help="Output path for results JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--scenarios-file", type=str, 
                       default="scenarios_with_answers.json",
                       help="Path to scenarios JSON")
    
    args = parser.parse_args()
    load_dotenv()
    scenarios = load_scenarios(args.scenarios_file)
    logger.info(f"Loaded {len(scenarios)} scenarios")
    
    if args.scenario:
        scenario = next((s for s in scenarios if s.scenario_number == args.scenario), None)
        if not scenario:
            print(f"Scenario {args.scenario} not found")
            return
        
        graph = build_graph()
        result = run_single_scenario(scenario, graph, verbose=True)
        print(f"\nResult: {result.model_dump_json(indent=2)}")
        
    elif args.all or args.limit:
        results = run_evaluation(
            scenarios,
            output_path=args.output,
            verbose=args.verbose,
            limit=args.limit
        )
        try:
            summary = calculate_metrics(results)
            print_summary(summary)
            summary_path = args.output.replace(".json", "_summary.json")
            with open(summary_path, "w") as f:
                f.write(summary.model_dump_json(indent=2))
            print(f"\nSummary saved to {summary_path}")
        except ValueError as e:
            print(f"\nCould not calculate metrics: {e}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
