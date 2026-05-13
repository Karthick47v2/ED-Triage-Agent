"""Evaluation module for ED-Triage-Agent.

``PatientAgent`` is loaded lazily so lightweight metric tools can run without
pulling in LangChain / Azure dependencies.
"""
from typing import Any

from eval.schemas import EvaluationScenario, EvaluationResult, EvaluationSummary
from eval.metrics import calculate_metrics, print_summary

__all__ = [
    "EvaluationScenario",
    "EvaluationResult",
    "EvaluationSummary",
    "PatientAgent",
    "calculate_metrics",
    "print_summary",
]


def __getattr__(name: str) -> Any:
    if name == "PatientAgent":
        from eval.patient_agent import PatientAgent

        return PatientAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
