"""Standalone runner for the Intake Interview Agent (IIA)."""
import os
import logging
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from ed_triage.common.message_sanitize import message_content_to_plain_text
from ed_triage.iia.agent import build_graph

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("IIA-Main")
logger.setLevel(logging.DEBUG)
logging.getLogger("IIA-Agent").setLevel(logging.DEBUG)


def main():
    load_dotenv()
    required_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_IIA_DEPLOYMENT_NAME",
    ]
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        logger.error("Missing environment variables: %s", missing)
        return

    graph = build_graph()
    state = {"messages": [], "intake_data": None}
    print("\n--- ED Triage Intake (Type 'q' to quit) ---")
    print("Nurse AI: Hello. What brings you to the ED today?")

    while True:
        user_input = input("\nPatient: ")
        if user_input.lower() in ("q", "quit", "exit"):
            break

        state["messages"].append(HumanMessage(content=user_input))
        result = graph.invoke(state)
        state["messages"] = result["messages"]
        if "intake_data" in result:
            state["intake_data"] = result["intake_data"]

        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.content:
            display_content = message_content_to_plain_text(
                last_message.content).replace("[CONVERSATION_END]", "").strip()
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
