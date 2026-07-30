"""Interactive runner for the Phase 1 pipeline (IIA -> CRA -> PAA)."""

from __future__ import annotations

import argparse
import logging
import uuid
from typing import Any, Mapping

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from ed_triage.common.message_sanitize import message_content_to_plain_text
from ed_triage.orchestration.phase1 import build_phase1_graph


LOGGER = logging.getLogger("Triage-System")

EXIT_COMMANDS = {"quit", "exit", "q"}
CONVERSATION_END_MARKER = "[CONVERSATION_END]"
SEPARATOR = "=" * 60
SUBSEPARATOR = "-" * 30

NODE_LABELS = {
    "interviewer": "IIA interview",
    "extractor": "IIA extraction",
    "cra": "CRA clinical reasoning",
    "paa": "PAA priority assessment",
}

# Only these namespaces get INFO/DEBUG; root stays WARNING so httpx/openai/etc. stay quiet.
TRIAGE_LOGGERS = (
    "Triage-System",
    "IIA-Agent",
    "PAA-Agent",
    "TCA-Agent",
    "ed_triage",
)


def configure_logging(*, verbose: bool) -> None:
    """Enable logs for this project's agents only."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s [%(name)s] %(message)s",
        force=True,
    )
    level = logging.DEBUG if verbose else logging.INFO
    for name in TRIAGE_LOGGERS:
        logging.getLogger(name).setLevel(level)


def truncate(value: Any, limit: int) -> str:
    """Convert a value to text and truncate it only when necessary."""
    text = str(value or "")
    return text if len(text) <= limit else f"{text[:limit].rstrip()}..."


def print_dashboard(state: Mapping[str, Any], *, verbose: bool = False) -> None:
    """Print the final triage results."""
    intake = state.get("intake_data")
    cra = state.get("cra_result")
    paa = state.get("paa_result")

    lines = [
        "",
        SEPARATOR,
        "                ED TRIAGE DASHBOARD",
        SEPARATOR,
        "",
    ]

    if paa is not None:
        lines.extend(
            [
                f"PRIORITY: {paa.priority_score} "
                f"(ESI {paa.tentative_esi})",
                f"High priority flag: {paa.is_high_priority}",
                f"Confidence: {paa.confidence}",
                f"Reasoning: {paa.reasoning}",
                "",
            ]
        )

    if cra is not None:
        lines.extend(
            [
                SUBSEPARATOR,
                "CLINICAL REASONING",
                SUBSEPARATOR,
                f"Suggested ESI: {cra.suggested_esi_level}",
                f"Clinical Concept: "
                f"{truncate(cra.clinical_explanation, 10_000 if verbose else 200)}",
                "Differential Diagnoses:",
            ]
        )

        for diagnosis in cra.differential_diagnoses:
            lines.append(
                f"  - {diagnosis.condition} ({diagnosis.likelihood})"
            )
            if verbose:
                lines.append(f"      {diagnosis.reasoning}")

        if cra.risk_factors:
            lines.append("Risk Factors:")
            lines.extend(f"  - {factor}" for factor in cra.risk_factors)

        if cra.critical_findings:
            lines.append("Critical Findings:")
            lines.extend(f"  - {finding}" for finding in cra.critical_findings)

        lines.append("References:")
        refs = (
            cra.esi_handbook_references
            if verbose
            else cra.esi_handbook_references[:2]
        )
        lines.extend(
            f"  * {truncate(reference, 10_000 if verbose else 100)}"
            for reference in refs
        )
        lines.append("")

    if intake is not None:
        lines.extend(
            [
                SUBSEPARATOR,
                "INTAKE SUMMARY",
                SUBSEPARATOR,
                f"Chief Complaint: {intake.chief_complaint}",
            ]
        )

        age_parts = []
        if intake.age_years is not None:
            age_parts.append(f"{intake.age_years}y")
        if intake.age_months is not None:
            age_parts.append(f"{intake.age_months}mo")
        if intake.age_days is not None:
            age_parts.append(f"{intake.age_days}d")
        if age_parts:
            lines.append(f"Age: {' '.join(age_parts)}")

        if intake.emergency_detected:
            lines.append(
                f"!!! EMERGENCY FLAG: {intake.emergency_reason} !!!"
            )

        hpi_names = ", ".join(symptom.name for symptom in intake.hpi)
        lines.append(f"HPI: {hpi_names or 'Not provided'}")

        if verbose and intake.hpi:
            for symptom in intake.hpi:
                lines.append(f"  - {symptom.model_dump_json()}")

        if verbose:
            if intake.medical_history:
                lines.append(
                    f"History: {', '.join(intake.medical_history)}"
                )
            if intake.medications:
                lines.append(f"Meds: {', '.join(intake.medications)}")
            if intake.allergies:
                lines.append(f"Allergies: {', '.join(intake.allergies)}")

    if verbose:
        lines.extend([SUBSEPARATOR, "FULL STRUCTURED OUTPUTS", SUBSEPARATOR])
        if intake is not None:
            lines.append("intake_data:")
            lines.append(intake.model_dump_json(indent=2))
        if cra is not None:
            lines.append("cra_result:")
            lines.append(cra.model_dump_json(indent=2))
        if paa is not None:
            lines.append("paa_result:")
            lines.append(paa.model_dump_json(indent=2))

    lines.extend([SEPARATOR, ""])
    print("\n".join(lines))


def display_interviewer_update(event: Mapping[str, Any]) -> None:
    """Print the newest interviewer message from a graph event."""
    interviewer_update = event.get("interviewer")

    if not interviewer_update:
        return

    messages = interviewer_update.get("messages") or []
    if not messages:
        return

    content = message_content_to_plain_text(messages[-1].content)
    display_text = content.removesuffix(CONVERSATION_END_MARKER).strip()

    if display_text:
        print(f"\nNurse AI: {display_text}")


def display_pipeline_progress(event: Mapping[str, Any]) -> None:
    """Announce non-chat graph nodes so decisions are visible mid-run."""
    for node_name, label in NODE_LABELS.items():
        if node_name in event and node_name != "interviewer":
            print(f"\n[{label}] done")
            LOGGER.info("Graph node finished: %s", node_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive Phase 1 triage (IIA -> CRA -> PAA).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG logs plus full structured intake/CRA/PAA dumps.",
    )
    return parser.parse_args()


def main() -> None:
    """Run one interactive triage session."""
    args = parse_args()
    load_dotenv()
    configure_logging(verbose=args.verbose)

    print("Initializing ED Triage System (IIA -> CRA -> PAA)...")
    if args.verbose:
        print("Verbose mode: DEBUG agent logs + full JSON at the end.")
    else:
        print("Logging: INFO (use -v for DEBUG and full JSON dumps).")

    graph = build_phase1_graph()
    config: RunnableConfig = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
        }
    }

    print("\nSystem Ready. Type 'quit' to exit.")
    print("-" * 50)

    while True:
        try:
            user_input = input("\nPatient: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            return

        if user_input.casefold() in EXIT_COMMANDS:
            print("Exiting...")
            return

        if not user_input:
            continue

        state_update = {
            "messages": [HumanMessage(content=user_input)],
        }

        try:
            for event in graph.stream(
                state_update,
                config=config,
                stream_mode="updates",
            ):
                display_interviewer_update(event)
                display_pipeline_progress(event)
        except Exception:
            LOGGER.exception("Phase 1 graph execution failed.")
            print(
                "\nThe triage system encountered an error. "
                "Please try again or contact support."
            )
            return

        state = graph.get_state(config).values

        if state.get("paa_result") is not None:
            print_dashboard(state, verbose=args.verbose)
            print("Session complete. Returning to standby.")
            return


if __name__ == "__main__":
    main()
