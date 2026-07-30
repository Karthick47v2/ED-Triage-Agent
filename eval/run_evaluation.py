"""Phase 1 evaluation: run ETEK scenarios through IIA -> CRA -> PAA with simulated patient agents."""
from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional

# Allow running as ``python eval/run_evaluation.py`` in addition to ``-m``.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from ed_triage.cra.agent import run_cra
from ed_triage.common.message_sanitize import message_content_to_plain_text
from ed_triage.iia.agent import extraction_node
from ed_triage.orchestration.phase1 import build_phase1_graph
from ed_triage.paa.agent import run_paa
from eval.metrics import calculate_metrics, print_summary
from eval.patient_agent import PatientAgent
from eval.schemas import EvaluationResult, EvaluationScenario
from eval.shared.io import load_json_file, save_json_file
from eval.shared.result_builder import apply_phase1_state
from eval.shared.triage import derive_priority

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("Evaluation")
logger.setLevel(logging.INFO)

# LangChain + Pydantic 2.12 emit warnings for structured-output ``parsed`` field.
warnings.filterwarnings(
    "ignore",
    message=r"Pydantic serializer warnings:",
    category=UserWarning,
    module=r"pydantic\.main",
)
MAX_TURNS = 15


def load_scenarios(json_path: str) -> List[EvaluationScenario]:
    return [EvaluationScenario(**s) for s in load_json_file(json_path)]


def _salvage_phase1_outputs(snapshot_values: dict) -> dict:
    """Run extractor/CRA/PAA on a checkpoint that has messages but no PAA result."""
    state = dict(snapshot_values)
    if not state.get("messages"):
        return state
    if not state.get("intake_data"):
        state.update(extraction_node(state))
    if state.get("intake_data") and not state.get("cra_result"):
        state["cra_result"] = run_cra(intake_summary=state["intake_data"])
    if state.get("intake_data") and state.get("cra_result") and not state.get("paa_result"):
        state["paa_result"] = run_paa(
            intake_summary=state["intake_data"],
            cra_result=state["cra_result"],
        )
    return state


def run_single_scenario(
    scenario: EvaluationScenario,
    graph,
    profile_id: int = 1,
    verbose: bool = False,
) -> EvaluationResult:
    """Run one ETEK scenario through the Phase 1 pipeline."""
    logger.info("Running scenario %d...", scenario.scenario_number)
    result = EvaluationResult(
        scenario_number=scenario.scenario_number,
        ground_truth_esi=scenario.category,
        ground_truth_priority=derive_priority(scenario.category),
    )

    start = time.time()
    try:
        patient = PatientAgent(scenario.description, profile_id=profile_id)
        config = {"configurable": {"thread_id": f"eval_{scenario.scenario_number}"}}

        transcript = []
        initial_patient_msg = patient.respond(
            "What brings you to the emergency department today?"
        )
        transcript.append({"role": "patient", "content": initial_patient_msg})

        state_update = {"messages": [HumanMessage(content=initial_patient_msg)]}
        final_state = None
        turns = 0
        conversation_complete = False

        while not conversation_complete and turns < MAX_TURNS:
            turns += 1
            if verbose:
                print("  --- Invoking Nurse (IIA) ---")
            nurse_response = None
            for event in graph.stream(state_update, config):
                interviewer_update = event.get("interviewer")
                if interviewer_update and interviewer_update.get("messages"):
                    nurse_response = message_content_to_plain_text(
                        interviewer_update["messages"][-1].content
                    )
                if "paa" in event:
                    final_state = event["paa"]

            if nurse_response:
                display = nurse_response.replace("[CONVERSATION_END]", "").strip()
                transcript.append({"role": "nurse", "content": display})
                if verbose:
                    print(f"  Nurse: {display}")
                if "[CONVERSATION_END]" in nurse_response:
                    conversation_complete = True
                else:
                    patient_reply = patient.respond(display)
                    transcript.append({"role": "patient", "content": patient_reply})
                    if verbose:
                        print(f"  Patient: {patient_reply}")
                    state_update = {"messages": [HumanMessage(content=patient_reply)]}

            snapshot = graph.get_state(config)
            if snapshot.values.get("paa_result"):
                conversation_complete = True
                final_state = snapshot.values

        # Salvage: model may not have emitted [CONVERSATION_END] before MAX_TURNS.
        if not final_state or not final_state.get("paa_result"):
            final_state = _salvage_phase1_outputs(graph.get_state(config).values)

        result.phase1_latency_ms = (time.time() - start) * 1000
        result.interview_turns = turns
        result.transcript = transcript
        apply_phase1_state(result, final_state)

        if not result.success:
            result.error_message = (
                f"Phase 1 did not complete within {MAX_TURNS} interview turns."
            )
            logger.error("  Scenario %d did not produce a final triage decision.", scenario.scenario_number)
        else:
            logger.info(
                "  Scenario %d: ESI=%s priority=%s (GT %d)",
                scenario.scenario_number,
                result.predicted_esi,
                result.predicted_priority,
                result.ground_truth_esi,
            )
    except Exception as exc:
        result.success = False
        result.error_message = str(exc)
        logger.error("  Scenario %d failed: %s", scenario.scenario_number, exc)

    return result


def run_evaluation(
    scenarios: List[EvaluationScenario],
    output_path: Optional[str] = None,
    profile_id: int = 1,
    verbose: bool = False,
    limit: Optional[int] = None,
) -> List[EvaluationResult]:
    """Run Phase 1 evaluation on all scenarios."""
    logger.info("Building evaluation graph...")
    graph = build_phase1_graph()
    if limit:
        scenarios = scenarios[:limit]
    logger.info("Running %d scenarios...", len(scenarios))

    results: List[EvaluationResult] = []
    for i, scenario in enumerate(scenarios, 1):
        print(f"[{i}/{len(scenarios)}] Scenario {scenario.scenario_number}", end="")
        result = run_single_scenario(scenario, graph, profile_id=profile_id, verbose=verbose)
        results.append(result)
        status = "OK" if result.success else "FAIL"
        match = "=" if result.predicted_esi == result.ground_truth_esi else "!="
        print(f" {status} ESI: {result.predicted_esi} {match} {result.ground_truth_esi}")

    if output_path:
        save_json_file(output_path, {
            "results": [r.model_dump() for r in results],
            "metadata": {
                "total_scenarios": len(scenarios),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        })
        logger.info("Results saved to %s", output_path)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 Evaluation Runner")
    parser.add_argument("--scenario", type=int, help="Run a single scenario by number")
    parser.add_argument("--all", action="store_true", help="Run all scenarios")
    parser.add_argument("--limit", type=int, help="Limit number of scenarios")
    parser.add_argument("--output", type=str, default="results/phase1_results.json",
                        help="Output path for results JSON")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--scenarios-file", type=str,
                        default="eval/practice_cases.json",
                        help="Path to scenarios JSON")
    parser.add_argument("--patient-profile", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Patient persona profile (1-4)")
    args = parser.parse_args()
    load_dotenv()

    scenarios = load_scenarios(args.scenarios_file)
    logger.info("Loaded %d scenarios", len(scenarios))

    if args.scenario:
        scenario = next(
            (s for s in scenarios if s.scenario_number == args.scenario), None
        )
        if not scenario:
            print(f"Scenario {args.scenario} not found")
            return
        graph = build_phase1_graph()
        result = run_single_scenario(scenario, graph, profile_id=args.patient_profile, verbose=True)
        print(f"\nResult: {result.model_dump_json(indent=2)}")
        return

    if args.all or args.limit:
        results = run_evaluation(
            scenarios,
            output_path=args.output,
            profile_id=args.patient_profile,
            verbose=args.verbose,
            limit=args.limit,
        )
        try:
            summary = calculate_metrics(results)
            print_summary(summary)
            summary_path = args.output.replace(".json", "_summary.json")
            save_json_file(summary_path, summary.model_dump())
            print(f"\nSummary saved to {summary_path}")
        except ValueError as exc:
            print(f"\nCould not calculate metrics: {exc}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
