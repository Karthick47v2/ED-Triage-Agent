import logging
import uuid

try:
    import readline
except ImportError:
    pass

from langchain_core.messages import HumanMessage
from ed_triage.graph import build_graph
from dotenv import load_dotenv

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("Triage-System")
logger.setLevel(logging.INFO)
logging.getLogger("IIA-Agent").setLevel(logging.DEBUG)
logging.getLogger("CRA-Agent").setLevel(logging.INFO)
logging.getLogger("PAA-Agent").setLevel(logging.INFO)


def print_dashboard(state):
    intake = state.get("intake_data")
    cra = state.get("cra_result")
    paa = state.get("paa_result")
    
    print("\n" + "="*60)
    print("                ED TRIAGE DASHBOARD")
    print("="*60 + "\n")
    
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
        print("\n")
        
    if intake:
        print("-" * 30)
        print("INTAKE SUMMARY")
        print("-" * 30)
        print(f"Chief Complaint: {intake.chief_complaint}")
        if intake.emergency_detected:
            print(f"!!! EMERGENCY FLAG: {intake.emergency_reason} !!!")
        print(f"HPI: {', '.join([s.name for s in intake.hpi])}")

    print("="*60 + "\n")

def main():
    load_dotenv()
    print("Initializing ED Triage System (IIA -> CRA -> PAA)...")
    graph = build_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "messages": [],
        "intake_data": None,
        "cra_result": None,
        "paa_result": None
    }
    print("\nSystem Ready. Type 'quit' to exit.")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nPatient: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
            
        if user_input.lower() in ["quit", "exit"]:
            break
        state_update = {"messages": [HumanMessage(content=user_input)]}
        if not initial_state["messages"]:
            initial_state["messages"] = [HumanMessage(content=user_input)]
        for event in graph.stream(state_update, config):
            for key, value in event.items():
                if key == "interviewer" and value.get("messages"):
                    last_msg = value["messages"][-1]
                    display_text = last_msg.content.replace("[CONVERSATION_END]", "")
                    print(f"\nNurse AI: {display_text}")
                elif key == "paa":
                    pass
        snapshot = graph.get_state(config)
        if snapshot.values.get("paa_result"):
            print_dashboard(snapshot.values)
            print("Session Complete. Returning to standby.")
            break

if __name__ == "__main__":
    main()
