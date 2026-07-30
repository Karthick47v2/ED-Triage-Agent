"""Format vital-sign assessments for LLM prompts."""

from __future__ import annotations

from collections.abc import Sequence

from ed_triage.common.vital_assessment.tables import (
    GAP_STATUSES,
    VitalAssessment,
    VitalSignsAssessment,
    VitalStatus,
)


_INCOMPLETE_VITALS_CAVEAT = (
    "**Incomplete vital set:** Overall status and ESI recommendation reflect "
    "**only** measurements assessed against configured thresholds. Missing or "
    "not-assessed vitals are **not** evidence of normal physiology."
)


def _format_unknown_vitals(
    vitals: Sequence[VitalAssessment],
) -> str:
    items = [
        f"- **{vital.name}:** {vital.reasoning}"
        for vital in vitals
        if vital.status is VitalStatus.UNKNOWN
    ]

    if not items:
        return ""

    return "\n".join([
        "### VITALS NOT RECORDED",
        "",
        *items,
    ])


def _format_not_assessed_vitals(
    vitals: Sequence[VitalAssessment],
) -> str:
    items = [
        (
            f"- **{vital.name}:** {vital.value} {vital.unit} — "
            f"{vital.reasoning}"
        )
        for vital in vitals
        if vital.status is VitalStatus.NOT_ASSESSED
    ]

    if not items:
        return ""

    return "\n".join([
        "### VITALS RECORDED BUT NOT AGE-ASSESSED",
        "",
        *items,
    ])


def _append_findings(
    lines: list[str],
    heading: str,
    findings: Sequence[str],
) -> None:
    """Append a non-empty findings section."""
    if findings:
        lines.extend([
            heading,
            *(f"- {finding}" for finding in findings),
            "",
        ])


def format_assessment_for_llm(
    assessment: VitalSignsAssessment,
) -> str:
    """Format a complete deterministic vital-sign assessment for an LLM."""
    vitals = assessment.vitals

    lines = [
        "## VITAL SIGNS ASSESSMENT (Deterministic Tool Output)",
        "",
        f"**Overall Status: {assessment.overall_status.value.upper()}**",
        f"**ESI Recommendation: {assessment.esi_recommendation}**",
        "",
    ]

    if assessment.has_incomplete_vitals:
        lines.extend([
            _INCOMPLETE_VITALS_CAVEAT,
            "",
        ])

    _append_findings(
        lines,
        "### CRITICAL FINDINGS",
        assessment.critical_findings,
    )
    _append_findings(
        lines,
        "### HIGH-RISK FINDINGS",
        assessment.high_risk_findings,
    )
    _append_findings(
        lines,
        "### ABNORMAL FINDINGS",
        assessment.abnormal_findings,
    )

    lines.append("### INDIVIDUAL VITAL ASSESSMENTS")

    lines.extend(
        (
            f"- {vital.name}: {vital.value} {vital.unit} → "
            f"{vital.status.value.upper()}"
        )
        for vital in vitals
        if vital.status not in GAP_STATUSES
    )

    additional_sections = (
        _format_unknown_vitals(vitals),
        _format_not_assessed_vitals(vitals),
    )

    for section in additional_sections:
        if section:
            lines.extend(["", section])

    return "\n".join(lines)
