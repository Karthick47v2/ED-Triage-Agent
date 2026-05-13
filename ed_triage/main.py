"""Interactive runner for the Phase 1 pipeline (IIA -> CRA -> PAA)."""
import logging
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from ed_triage.common.message_sanitize import message_content_to_plain_text
from ed_triage.orchestration.phase1 import build_phase1_graph

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("Triage-System")
logger.setLevel(logging.INFO)


def print_dashboard(state) -> None:
    intake = state.get("intake_data")
    cra = state.get("cra_result")
    paa = state.get("paa_result")

    print("\n" + "=" * 60)
    print("                ED TRIAGE DASHBOARD")
    print("=" * 60 + "\n")

    if paa:
        print(f"PRIORITY: {paa.priority_score} (ESI {paa.tentative_esi})")
        print(f"Confidence: {paa.confidence}")
        print(f"Reasoning: {paa.reasoning}\n")

    if cra:
        print("-" * 30)
        print("CLINICAL REASONING")
        print("-" * 30)
        print(f"Clinical Concept: {cra.clinical_explanation[:200]}...")
        print("Differential Diagnoses:")
        for diag in cra.differential_diagnoses:
            print(f"  - {diag.condition} ({diag.likelihood})")
        print("References:")
        for ref in cra.esi_handbook_references[:2]:
            print(f"  * {ref[:100]}...")
        print()

    if intake:
        print("-" * 30)
        print("INTAKE SUMMARY")
        print("-" * 30)
        print(f"Chief Complaint: {intake.chief_complaint}")
        if intake.emergency_detected:
            print(f"!!! EMERGENCY FLAG: {intake.emergency_reason} !!!")
        print(f"HPI: {', '.join(s.name for s in intake.hpi)}")

    print("=" * 60 + "\n")


def main() -> None:
    load_dotenv()
    print("Initializing ED Triage System (IIA -> CRA -> PAA)...")
    graph = build_phase1_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    print("\nSystem Ready. Type 'quit' to exit.")
    print("-" * 50)

    while True:
        try:
            user_input = input("\nPatient: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        state_update = {"messages": [HumanMessage(content=user_input)]}
        for event in graph.stream(state_update, config):
            interviewer_update = event.get("interviewer")
            if interviewer_update and interviewer_update.get("messages"):
                last_msg = interviewer_update["messages"][-1]
                display_text = message_content_to_plain_text(
                    last_msg.content
                ).replace("[CONVERSATION_END]", "")
                print(f"\nNurse AI: {display_text}")

        snapshot = graph.get_state(config)
        if snapshot.values.get("paa_result"):
            print_dashboard(snapshot.values)
            print("Session Complete. Returning to standby.")
            break


if __name__ == "__main__":
    main()
