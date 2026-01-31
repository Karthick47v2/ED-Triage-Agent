import os
import sys
import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("IIA-Main")
logger.setLevel(logging.DEBUG)
logging.getLogger("IIA-Agent").setLevel(logging.DEBUG)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from ed_triage.iia.agent import build_graph
from ed_triage.iia.schema import IntakeSummary

def main():
    load_dotenv()
    logger.info("Initializing Intake Interview Agent (IIA)...")
    required_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_IIA_DEPLOYMENT_NAME"]
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        logger.error(f"Missing environment variables: {missing}")
        return

    graph = build_graph()
    state = {"messages": [], "intake_data": None}
    print("\n--- ED Triage Intake (Type 'q' to quit) ---")
    print("Nurse AI: Hello. What brings you to the ED today?")
    
    while True:
        user_input = input("\nPatient: ")
        if user_input.lower() in ["q", "quit", "exit"]:
            break
            
        state["messages"].append(HumanMessage(content=user_input))
        logger.debug(f"Invoking graph with {len(state['messages'])} messages.")
        result = graph.invoke(state)
        state["messages"] = result["messages"]
        if "intake_data" in result:
            state["intake_data"] = result["intake_data"]

        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.content:
            display_content = last_message.content.replace("[CONVERSATION_END]", "").strip()
            if display_content:
                print(f"Nurse AI: {display_content}")

        current_data = state.get("intake_data")
        if current_data:
            print("\n--- Interview & Extraction Complete ---")
            if current_data.emergency_detected:
                print(f"!!! EMERGENCY FLAGGED !!!\nReason: {current_data.emergency_reason}")
            print("Summary Extracted:")
            print(current_data.model_dump_json(indent=2, exclude_none=True))
            break

if __name__ == "__main__":
    main()
